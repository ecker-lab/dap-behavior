# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import shutil

from tqdm import tqdm

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    # Our slurm setup only exposes one cuda for each task, so nothing to do here
    pass
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


import wandb
from yacs.config import CfgNode

from dap_behavior.jepa.src.datasets.data_manager import init_data
from dap_behavior.jepa.src.masks.random_tube import MaskCollator as TubeMaskCollator
from dap_behavior.jepa.src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from dap_behavior.jepa.src.masks.utils import apply_masks
from dap_behavior.jepa.src.utils.distributed import init_distributed, AllReduce, broadcast_string
from dap_behavior.jepa.src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from dap_behavior.jepa.src.utils.tensors import repeat_interleave_batch

from dap_behavior.jepa.app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from dap_behavior.jepa.app.vjepa.transforms import make_transforms


# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

def setup_train(project_id, save_dir, cfg_name, resume_id=None):

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    run_id = f"{cfg_name}-{timestamp}" if resume_id is None else resume_id

    ckpt_dir = f"{save_dir}/checkpoints/{project_id}/{run_id}/"

    while True:
        try:
            os.makedirs(ckpt_dir)
            break
        except FileExistsError:
            run_id = f"{cfg_name}-{timestamp}-{random.randint(0, 10000)}"
            ckpt_dir = f"{save_dir}/checkpoints/{project_id}/{run_id}/"
            logging.warning(f"Checkpoint directory {ckpt_dir} already exists. Trying again with a different run_id.")

    return run_id, ckpt_dir


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    fname = args.get('fname')
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    finetune = cfgs_meta.get('finetune', False)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    

    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)
    use_caching = cfgs_data.get('use_caching', False)

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)
    crop_to_bboxes = cfgs_data_aug.get('crop_to_bboxes', True)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff')

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # Copy datasets to node-local storage and create csv files for dataloaders
    # Note: This currently only works if all data is used on all nodes

    # ----------------------------------------------------------------------- #

    
    video_base_dir = dataset_paths["data_path"]
    dataset_paths = [dataset_paths["label_path"]]

    cache_dir = None if not use_caching else os.environ.get('LOCAL_TMPDIR', "/local/")

    if isinstance(save_every_freq, int):
        if save_every_freq > 0:
            save_epochs = list(range(0, num_epochs, save_every_freq))
        else:
            save_epochs = []
    elif isinstance(save_every_freq, list):
        save_epochs = save_every_freq


    # ----------------------------------------------------------------------- #

    if rank == 0:
        run_id, ckpt_dir = setup_train(tag, folder, Path(fname).stem, resume_id=None)

        os.makedirs(folder, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        run = wandb.init(
            project=tag,
            name=run_id,
            config=args,
            dir=folder,
        )
    else:
        ckpt_dir = None

    
    
    logger.info(f'MASTER_ADDR: {os.environ["MASTER_ADDR"]}')
    logger.info(f'MASTER_PORT: {os.environ["MASTER_PORT"]}')
    logger.info(f'RANK: {rank}')
    logger.info(f'WORLD_SIZE: {world_size}')
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Cuda devices visible: {torch.cuda.device_count()}")

    # -- set device
    if not torch.cuda.is_available():
        logger.error('CUDA not available')
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:0')
        torch.cuda.set_device(device)

    # broadcast rank 0 ckpt_dir to all other ranks
    ckpt_dir = broadcast_string(rank, ckpt_dir, src=0, device=device)

    # -- log/checkpointing paths
    log_file = os.path.join(ckpt_dir, f'r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(ckpt_dir, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False
    if finetune:
        load_path = os.path.join(folder, r_file)
        if not os.path.exists(load_path):
            logger.error(f'Finetuning: {load_path} does not exist')
            return

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'enc-grad-norm'),
        ('%.5f', 'pred-grad-norm'),
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    target_encoder = copy.deepcopy(encoder)

    if model_name.startswith("hiera"):
        assert encoder.backbone.mask_unit_size[1] == encoder.backbone.mask_unit_size[2]
        assert encoder.backbone.patch_stride[1] == encoder.backbone.patch_stride[2]
        mask_unit_size = encoder.backbone.mask_unit_size[1]*encoder.backbone.patch_stride[1]
        assert mask_unit_size == 32, f"{mask_unit_size=}"
        # For hiera models, mask_unit_size is not equal to Transformer patch size, as the transformer patch size varies throughtout the model
    else:
        mask_unit_size = patch_size

    # -- make data transforms
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=mask_unit_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=mask_unit_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         batch_size=batch_size,
         training=True,
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=datasets_weights,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=folder if log_resource_util_data else None,
         video_base_path=video_base_dir,
         crop_to_bboxes=crop_to_bboxes,
         cache_dir=cache_dir,)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    if use_caching:
        if world_size > 8:
            logger.error("Prewarming the cache for multi-node will not work properly for now")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            indices = range(rank, len(unsupervised_loader.dataset), world_size)
            transfered_bytes = 0
            for ret in (bar := tqdm(executor.map(lambda i: unsupervised_loader.dataset.cache_sample(i), indices), total=len(unsupervised_loader.dataset)/world_size, desc="Caching samples (0MB)", disable=None)):
                transfered_bytes += ret
                bar.set_description(f"Caching samples ({transfered_bytes / 1024**2:.1f}MB)")

        end_time = time.time()

        logger.info(f"Rank {rank} finished caching samples in {end_time - start_time:.1f}s, {transfered_bytes / 1024**2 / (end_time-start_time):.1f}MB/s")

        torch.distributed.barrier()
        logger.info(f"Finished caching samples for rank {rank}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    if finetune:
        # Difference to laod_model: do not load optimizer state and start from epoch 0
        (
            encoder,
            predictor,
            target_encoder,
            _,
            _,
            _,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None)

    def save_checkpoint(epoch, path, inference_only=False):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict() if not inference_only else None,
            'predictor': predictor.state_dict() if not inference_only else None,
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    logger.info('Initializing loader...')
    loader = iter(unsupervised_loader)

    epoch = 0
    loss_meter = AverageMeter()
    #save_checkpoint(0, os.path.join(ckpt_dir, f'untrained.pth.tar'))

    if skip_batches > 0:
        logger.info(f'Skip {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f'Skip {itr}/{skip_batches} batches')
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)

    samples_seen = 0

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception as e:
                if not isinstance(e, StopIteration):
                    logger.warning(f'Exception refreshing data loader: {e.__class__.__name__}: {e}')
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)
            clips, masks_enc, masks_pred = load_clips()

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred, concat=False)
                        #print("h0 shape", h[0].shape)
                        return h

                def forward_context(c, h):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    z = encoder(c, masks_enc)
                    #print("s0 shape", z[0].shape)
                    z = predictor(z, h, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # Step 1. Forward
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z = forward_context(clips, h)
                    loss_jepa = loss_fn(z, h)  # jepa prediction loss
                    pstd_z = reg_fn(z)  # predictor variance across patches
                    loss_reg += torch.mean(F.relu(1.-pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg

                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )
            (loss, loss_jepa, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            samples_seen += len(clips) * world_size

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)
                
                if rank == 0:
                    log_dict ={
                        "epoch": epoch + 1,
                        #"itr": itr,
                        "loss": loss,
                        #"loss-jepa": loss_jepa,
                        #"reg-loss": loss_reg,
                        #"enc-grad-norm": grad_stats.global_norm,
                        #"pred-grad-norm": grad_stats_pred.global_norm,
                        "gpu-time(ms)": gpu_etime_ms,
                        "wall-time(ms)": iter_elapsed_time_ms,
                        "samples-seen": samples_seen,
                        "learning-rate": _new_lr,
                        "weight-decay": _new_wd,
                    }

                    if reg_coeff > 0:
                        # if reg_coeff is 0, then the loss only consists of the jepa loss
                        log_dict["reg-loss"] = loss_reg
                        log_dict["loss-jepa"] = loss_jepa

                    log_dict.update({f'avg/mask-{i}': m.avg for i, m in enumerate(mask_meters)})
                    log_dict.update({
                        "avg/loss": loss_meter.avg,
                        "avg/input-var": input_var_meter.avg,
                        "avg/input-var-min": input_var_min_meter.avg,
                        "avg/gpu-time": gpu_time_meter.avg,
                        "avg/wall-time": wall_time_meter.avg,
                    })
                    log_dict["cuda-memory"] = torch.cuda.max_memory_allocated() / 1024.0**2

                    log_dict.update({
                        "optim-stats/first_moment": optim_stats.get('exp_avg').avg,
                        "optim-stats/first_moment_min": optim_stats.get('exp_avg').min,
                        "optim-stats/first_moment_max": optim_stats.get('exp_avg').max,
                        "optim-stats/second_moment": optim_stats.get('exp_avg_sq').avg,
                        "optim-stats/second_moment_min": optim_stats.get('exp_avg_sq').min,
                        "optim-stats/second_moment_max": optim_stats.get('exp_avg_sq').max,
                    })

                    log_dict.update({
                        "grad-stats/enc-first-layer": grad_stats.first_layer,
                        "grad-stats/enc-last-layer": grad_stats.last_layer,
                        "grad-stats/enc-min": grad_stats.min,
                        "grad-stats/enc-max": grad_stats.max,
                        "grad-stats/enc-global-norm": grad_stats.global_norm,
                    })

                    log_dict.update({
                        "grad-stats/pred-first-layer": grad_stats_pred.first_layer,
                        "grad-stats/pred-last-layer": grad_stats_pred.last_layer,
                        "grad-stats/pred-min": grad_stats_pred.min,
                        "grad-stats/pred-max": grad_stats_pred.max,
                        "grad-stats/pred-global-norm": grad_stats_pred.global_norm,
                    })

                    run.log(log_dict)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           jepa_loss_meter.avg,
                           reg_loss_meter.avg,
                           input_var_meter.avg,
                           input_var_min_meter.avg,
                           '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.0**2,
                           gpu_time_meter.avg,
                           wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch + 1, itr,
                               optim_stats.get('exp_avg').avg,
                               optim_stats.get('exp_avg').min,
                               optim_stats.get('exp_avg').max,
                               optim_stats.get('exp_avg_sq').avg,
                               optim_stats.get('exp_avg_sq').min,
                               optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max,
                               grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats_pred.first_layer,
                               grad_stats_pred.last_layer,
                               grad_stats_pred.min,
                               grad_stats_pred.max,
                               grad_stats_pred.global_norm))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint
        logger.info('avg. loss %.3f' % loss_meter.avg)
        # -- Save Last
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if epoch in save_epochs:
                save_every_file = f'{tag}-e{epoch}.pth.tar'
                save_every_path = os.path.join(ckpt_dir, save_every_file)
                save_checkpoint(epoch + 1, save_every_path, inference_only=True)#

    if cache_dir is not None:
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info(f"Removed cache dir {cache_dir}")

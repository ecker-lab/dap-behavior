# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
import time
from multiprocessing import Process
import secrets

from dap_behavior.utils.misc import timestamp

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    #os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    # our slurm setup exposes only one GPU per process
    pass
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import dap_behavior.jepa.src.models.vision_transformer as vit
from dap_behavior.jepa.src.models.attentive_pooler import AttentiveClassifier
from dap_behavior.jepa.src.datasets.data_manager import (
    init_data,
)
from dap_behavior.jepa.src.utils.distributed import (
    init_distributed,
    AllReduce
)
from dap_behavior.jepa.src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from dap_behavior.jepa.src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from dap_behavior.jepa.evals.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation,
    FrameAggregation
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = secrets.randbits(32)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    train_data_path = args_data.get('dataset_train')
    val_data_path = args_data.get('dataset_val')
    test_data_path = args_data.get('dataset_test', None)
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    num_workers = args_data.get('num_workers', 12)
    copy_local = args_data.get('copy_local', [])

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    val_batch_size = args_opt.get('val_batch_size', batch_size)
    repetitions_per_epoch = args_opt.get('repetitions_per_epoch', 1)


    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    args_head = args_eval.get('head', dict())
    complete_block = args_head.get('complete_block', True)

    args_eval["global_seed"] = _GLOBAL_SEED
    print("SEED", _GLOBAL_SEED)

    # ----------------------------------------------------------------------- #

    WAIT_FOR_PROCESSES = []

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    logger.info(f'MASTER_ADDR: {os.environ["MASTER_ADDR"]}')
    logger.info(f'MASTER_PORT: {os.environ["MASTER_PORT"]}')
    logger.info(f'RANK: {rank}')
    logger.info(f'WORLD_SIZE: {world_size}')
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Cuda devices visible: {torch.cuda.device_count()}")

    # -- log/checkpointing paths
    folder_base = os.path.join(pretrain_folder, 'video_classification_frozen/')

    os.makedirs(folder_base, exist_ok=True)

    # ensure that the folder is unique
    while True:

        if eval_tag is not None:
            folder = os.path.join(folder_base, f"{eval_tag}_{timestamp(mode='short_seconds')}/")

        try:
            os.makedirs(folder, exist_ok=False)
            break
        except OSError:
            sleep_time = 2
            logger.info(f"Folder {folder} already exists, waiting for {sleep_time} seconds...")
            time.sleep(sleep_time)

    ckp_stem = ckp_fname.split('.')[0]
    log_file = os.path.join(folder, f'{ckp_stem}_r{rank}.csv')
    config_file = os.path.join(folder, f'{ckp_stem}_r{rank}.yaml')
    latest_path = os.path.join(folder, f'{ckp_stem}-latest.pth.tar')

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'train_acc'),
                               ('%.5f', 'val_acc'),
                               ('%.5f', 'test_acc'))
        
    with open(config_file, 'w') as f:
        f.write(pprint.pformat(args_eval))
        
    # ----------------------------------------------------------------------- #

    # ----------------------------------------------------------------------- #

    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
        complete_block=complete_block,
    ).to(device)

    train_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=[train_data_path["label_path"]],
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_segments=eval_num_segments if attend_across_segments else 1,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        repetitions_per_epoch=repetitions_per_epoch,
        video_base_path=train_data_path["data_path"],)
    val_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=[val_data_path["label_path"]],
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=val_batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        video_base_path=val_data_path["data_path"],)
    if test_data_path is not None:
        test_loader = make_dataloader(
            dataset_type=dataset_type,
            root_path=[test_data_path["label_path"]],
            resolution=resolution,
            frames_per_clip=eval_frames_per_clip,
            frame_step=eval_frame_step,
            num_segments=eval_num_segments,
            eval_duration=eval_duration,
            num_views_per_segment=eval_num_views_per_segment,
            allow_segment_overlap=True,
            batch_size=val_batch_size,
            world_size=world_size,
            rank=rank,
            training=False,
            num_workers=num_workers,
            video_base_path=test_data_path["data_path"],)
    else:
        test_loader = None
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)
    classifier = DistributedDataParallel(classifier, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            torch.save(save_dict, os.path.join(folder, f'{ckp_stem}-epoch{epoch:02d}.pth.tar'))

    val_accs = []

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_acc = run_one_epoch(
            device=device,
            training=True,
            num_temporal_views=eval_num_segments if attend_across_segments else 1,
            attend_across_segments=attend_across_segments,
            num_spatial_views=1,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16)

        val_logit_path = os.path.join(folder, f'{ckp_stem}_val_logits_epoch{(epoch + 1):02d}_r{rank}')

        val_acc = run_one_epoch(
            device=device,
            training=False,
            num_temporal_views=eval_num_segments,
            attend_across_segments=attend_across_segments,
            num_spatial_views=eval_num_views_per_segment,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            save_raw_scores=val_logit_path)
        
        if val_loader.dataset.eval_func is not None:
            #val_loader.dataset.eval_func(val_logit_path + '_indices.npy')
            WAIT_FOR_PROCESSES.append(Process(target=val_loader.dataset.eval_func, args=(val_logit_path + '_indices.npy',)))
            WAIT_FOR_PROCESSES[-1].start()
        
        if test_loader is not None:
            path = os.path.join(folder, f'{ckp_stem}_test_logits_epoch{(epoch + 1):02d}_r{rank}')

            test_acc = run_one_epoch(
                device=device,
                training=False,
                num_temporal_views=eval_num_segments,
                attend_across_segments=attend_across_segments,
                num_spatial_views=eval_num_views_per_segment,
                encoder=encoder,
                classifier=classifier,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=test_loader,
                use_bfloat16=use_bfloat16,
                save_raw_scores=path)
            
            logger.info(f'Saved test logits to {path}')

            if val_loader.dataset.eval_func is not None:
                #val_loader.dataset.eval_func(val_logit_path + '_indices.npy')
                WAIT_FOR_PROCESSES.append(Process(target=test_loader.dataset.eval_func, args=(path + '_indices.npy',)))
                WAIT_FOR_PROCESSES[-1].start()
        else:
            test_acc = 0.0

        val_accs.append(val_acc)

        logger.info('[%5d] train: %.3f%% val: %.3f%% test: %.3f%%' % (epoch + 1, train_acc, val_acc, test_acc))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc, test_acc)
        save_checkpoint(epoch + 1)

    if rank == 0:
        best_val_acc_idx = np.argmax(val_accs)
        best_val_acc = val_accs[best_val_acc_idx]
        logger.info(f'Best validation accuracy: {best_val_acc} at epoch {best_val_acc_idx + 1}')
        result_file = os.path.join(folder, f'{ckp_stem}_best_val_acc{best_val_acc:.5f}_epoch{best_val_acc_idx + 1}.txt')
        with open(result_file, 'w') as f:
            f.write(f'{best_val_acc:.5f} at epoch {best_val_acc_idx + 1}')

    # -- wait for processes to finish
    for p in WAIT_FOR_PROCESSES:
        p.join()
        if p.exitcode != 0:
            logger.warning(f"Process {p} exited with code {p.exitcode}")


def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
    save_raw_scores=None
):
    
    classifier.train(mode=training)
    if data_loader.dataset.multi_label:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:   
        criterion = torch.nn.CrossEntropyLoss()

    dataset_len = len(data_loader.dataset)
    logger.info(f"Dataset length: {dataset_len}")

    print(f"Using loss criterion: {criterion}")

    scores = None
    indices = None
    write_to_disk_idx = 0

    top1_meter = AverageMeter()
    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)
            info = data[3]

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    if attend_across_segments:
                        outputs = [classifier(o) for o in outputs]
                    else:
                        outputs = [[classifier(ost) for ost in os] for os in outputs]
            if training:
                if attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os] for os in outputs]

        if save_raw_scores is not None:
            o = torch.stack(outputs, dim=1)
            i = info['index'].detach().cpu().numpy()

            if scores is None:
                scores = np.memmap(save_raw_scores + f'_scores.npy', dtype=np.float32, mode='w+', shape=(dataset_len, *o.shape[1:]))
                indices = np.memmap(save_raw_scores + f'_indices.npy', dtype=np.int32, mode='w+', shape=(dataset_len,))

            scores[write_to_disk_idx:write_to_disk_idx + len(o)] = o.detach().cpu().numpy()
            indices[write_to_disk_idx:write_to_disk_idx + len(o)] = i
            write_to_disk_idx += len(o)

        # Compute loss
        if attend_across_segments:
            loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
        else:
            loss = sum([sum([criterion(ost, labels) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
        with torch.no_grad():
            if attend_across_segments:
                outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            else:
                outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])

            if data_loader.dataset.multi_label:
                #top1_acc = float(AllReduce.apply(-loss.detach()))
                top1_meter.update(-loss.item())
            else:
                top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
                top1_acc = float(AllReduce.apply(top1_acc))
                top1_meter.update(top1_acc)

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info(("TRAIN" if training else "VAL") + ' [%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))
            
    if save_raw_scores is not None:
        scores.flush()
        indices.flush()
        del scores
        del indices
    
    return top1_meter.avg


def load_checkpoint(
    device,
    r_path,
    classifier,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['classifier']

        if 'module.' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            logging.info('Removing module. from keys in pretrained_dict')

        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return classifier, opt, scaler, epoch


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    repetitions_per_epoch=1,
    **kwargs
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
        repetitions_per_epoch=repetitions_per_epoch,
        **kwargs)
    return data_loader


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    logger.info(f'Loaded pretrained encoder from {pretrained}...')
    return encoder


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler

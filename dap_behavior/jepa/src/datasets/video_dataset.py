# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from functools import partial
import json
import os
import pathlib
import shutil
import warnings
from tqdm import tqdm

from logging import getLogger

import cv2
import numpy as np
import pandas as pd
from filelock import FileLock

from decord import VideoReader, cpu

import torch



_GLOBAL_SEED = 0
logger = getLogger()

def load_video_cv2(fname, all_indices):
    cap = cv2.VideoCapture(fname)

    buffer = []

    for i in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f'Error reading frame {i} from {fname=}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(frame)
    buffer = np.stack(buffer, axis=0)

    return buffer

def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
    repetitions_per_epoch=1,
    **kwargs,
):
    from dap_behavior.jepa.src.datasets.utils.weighted_sampler import DistributedWeightedSampler

    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform,
        **kwargs,)
    
    if False:
        dataset_ = torch.utils.data.Subset(dataset, range(0, 100))
        print(f'Using subset of {len(dataset_)} samples')
        dataset_.multi_label = dataset.multi_label
        dataset = dataset_

    logger.info('VideoDataset dataset created')
    if repetitions_per_epoch > 1:
        dataset = torch.utils.data.ConcatDataset([dataset] * repetitions_per_epoch)
    
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler

def _ava_eval_func(scores_path, data, annot_joint, data_path):

    try:

        from dap_behavior.eval.ava_map import load_ava_results, ava_map

        pred = load_ava_results(scores_path)
        results = ava_map(**pred, 
                    annot_join=annot_joint,
                    gt=data, 
                    label_file=data_path.parent / 'action_list.json',
                    exclude_file=data_path.parent / f"{data_path.stem}_excluded_timestamps.csv",)

        with open(scores_path.replace('_indices.npy', '_ava_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(f'AVA results for {data_path} saved to {scores_path.replace("_indices.npy", "_ava_results.json")}')

    except Exception as e:
        print(f'ERROR processing AVA results {scores_path=}: {e.__class__} {e}')
        #raise e

class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
        video_base_path=None,
        crop_increase_factor=0.25,
        temporal_crop_frames=None,
        cache_dir=None,
        crop_to_bboxes=True,
    ):
        """"

        Each data_path's entry must be an absolute path to a csv file. We interpret the all video paths in the csv file as relative to data_path's parent directory. If bboxes are available, they should be stored in a file with the same stem as the csv file but with a '_bboxes.npy' suffix. The bboxes file should contain a numpy array with the following fields:
        - bboxes: np.array of shape (N, 4) where N is the number of bboxes. Each bbox is represented as [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bbox.
        - sample_idx2bbox_idx: np.array of shape (M,) where M is the number of samples in the csv file. This array maps each sample to the corresponding bbox indices in the bboxes array. The i-th element of this array is the index in the bboxes array corresponding to the i-th sample in the csv file."""
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = max(frame_step, 1)
        self.center_frame_only = frame_step == 0
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if self.center_frame_only:
            print(f'Using center frame only, {self.frames_per_clip=}, {self.frame_step=}')
        
        
        self.crop_increase_factor = crop_increase_factor
        self.multi_label = False
        self.bbox_format = None # or "xyxy_relative" or "tlwh_absolute"
        self.eval_func = None # Custom evaluation function that is called during training


        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        self.samples = [] # list of video paths
        self.labels = [] # list of labels or a np.array of shape (N, num_classes) where N is the number of samples
        self.bboxes_list = [] # list of bboxes for each sample (list[list[list[float]]]) or empty
        self.time_intervals = [] # list of time intervals for each sample (list[tuple[int, int]]) or empty
        self.num_samples_per_dataset = []

        assert video_base_path is not None, "video_base_path must be set"

        self.cache_dir = pathlib.Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.base_dir = pathlib.Path(video_base_path)

        for data_path in self.data_paths:

            dataPath = pathlib.Path(data_path)

            if data_path.endswith('.ava.csv'):
                # Load AVA dataset
                data = pd.read_csv(data_path, 
                                   header=None, 
                                   delimiter=",", 
                                   names=["filename", "frame", "x_min", "y_min", "x_max", "y_max", "label", "entity_id"],
                                   dtype={"filename": str, "frame": int, "x_min": float, "y_min": float, "x_max": float, "y_max": float, "label": int, "entity_id": int})

                with open(dataPath.parent / 'action_list.json', 'r') as f:
                    action_list = json.load(f)

                print(f'Loaded {len(data)} rows from {data_path}')

                group_keys = ["filename", "frame", "entity_id", "x_min", "y_min", "x_max", "y_max"]

                annot_joint = data[group_keys].drop_duplicates()
                annot_joint = annot_joint.set_index(["filename", "frame", "entity_id"])

                duplicates_in_frames = annot_joint[annot_joint.index.duplicated(keep=False)]
                if len(duplicates_in_frames) > 0:
                    print("Warning: duplicates in gt_short", duplicates_in_frames.index.drop_duplicates())

                annot_joint = annot_joint[~annot_joint.index.duplicated(keep='last')]
                annot_joint = annot_joint.reset_index()

                print(f'Got {len(annot_joint)} samples from {data_path} after removing duplicates and merging multi-labels')

                self.eval_func = partial(_ava_eval_func, data=data, annot_joint=annot_joint, data_path=dataPath)

                assert len(self.data_paths) == 1, 'Multi-label datasets are only supported for single dataset for now'

                self.samples = annot_joint["filename"].to_list()

                action_label2idx = {action["id"]: i for i, action in enumerate(action_list)}

                annot_joint_ = annot_joint.reset_index().rename(columns={'index': 'orig_idx'})[['orig_idx', 'filename', 'frame', 'entity_id']]

                # 2. Perform a single merge on the join keys
                merged = annot_joint_.merge(
                    data[['filename', 'frame', 'entity_id', 'label']],
                    on=['filename', 'frame', 'entity_id'],
                    how='left'
                )

                # 3. Drop rows with no label match and map labels to indices
                merged = merged.dropna(subset=['label'])
                merged['action_idx'] = merged['label'].map(action_label2idx)
                merged = merged.dropna(subset=['action_idx'])
                merged['action_idx'] = merged['action_idx'].astype(int)

                # 4. Prepare the label matrix
                N = len(annot_joint)
                K = len(action_label2idx)
                self.labels = np.zeros((N, K), dtype=np.float32)

                # 5. Vectorized assignment
                rows = merged['orig_idx'].to_numpy(dtype=int)
                cols = merged['action_idx'].to_numpy(dtype=int)
                self.labels[rows, cols] = 1

                self.bboxes_list = [[row[["x_min", "y_min", "x_max", "y_max"]].to_list()] for _, row in annot_joint.iterrows()]

                if temporal_crop_frames is None:
                    temporal_crop_frames = self.frames_per_clip * self.frame_step

                self.time_intervals = [(row["frame"]-temporal_crop_frames//2, row["frame"]+temporal_crop_frames//2) for _, row in annot_joint.iterrows()]

                assert self.bbox_format is None or self.bbox_format == "xyxy_relative", f"conflicting bbox_format {self.bbox_format} already set"
                self.bbox_format = "xyxy_relative"

                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)
                self.multi_label = True

            elif data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ")
                # Convert relative paths to absolute paths
                data.iloc[:, 0] = data.iloc[:, 0]
                
                self.samples += list(data.values[:, 0])

                if len(data.values[0]) > 2:
                    self.multi_label = True
                    self.labels = data.values[:, 1:].astype(np.float32)
                    assert len(self.data_paths) == 1, 'Multi-label datasets are only supported for single dataset for now'
                else:
                    self.labels += list(data.values[:, 1])

                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

                bbox_path = dataPath.parent / (str(dataPath.stem) + '_bboxes.npz')
                if os.path.exists(bbox_path):
                    logger.info(f'Loading bounding boxes from {bbox_path}')
                    with np.load(bbox_path, allow_pickle=True) as data:
                        

                        sample_idx2bbox_idx = data['sample_idx2bbox_idx']
                        this_bboxes = data['bboxes']

                        for index in range(num_samples):
                            bbox_start_idx = sample_idx2bbox_idx[index]
                            bbox_end_idx = sample_idx2bbox_idx[index + 1] if index + 1 < len(sample_idx2bbox_idx) else len(this_bboxes)

                            this_bboxes_list = []
                            for bbox_idx in range(bbox_start_idx, bbox_end_idx):
                                this_bboxes_list.append(this_bboxes[bbox_idx].tolist())

                            self.bboxes_list.append(this_bboxes_list)

                    assert self.bbox_format is None or self.bbox_format == "xyxy_relative", f"conflicting bbox_format {self.bbox_format} already set"
                    self.bbox_format = "xyxy_relative"
                    logger.info(f'Loaded bounding boxes')
                    

            elif data_path.endswith(".json"):
                with open(data_path, "r") as f:
                    data = json.load(f)
                
                self.samples += [str(i["file_name"]) for i in data["images"]]
                self.labels += [i.get("category_id", 0) for i in data["images"]]

                num_samples = len(data["images"])
                self.num_samples_per_dataset.append(len(data["images"]))

                bbox_dict = defaultdict(list)
                for i in data["annotations"]:
                    bbox_dict[i["image_id"]].append(i["bbox"])

                this_bboxes_list = [bbox_dict[i["id"]] for i in data["images"]]
                self.bboxes_list += this_bboxes_list

                assert self.bbox_format is None or self.bbox_format == "tlwh_absolute", f"conflicting bbox_format {self.bbox_format} already set"
                self.bbox_format = "tlwh_absolute"

            else:
                raise ValueError(f'Unsupported file format: {data_path}')

            print(f'Loaded {num_samples} samples from {data_path}')

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        if not crop_to_bboxes:
            print("Forgetting bounding boxes, as bbox cropping is disabled")
            self.bboxes_list = []

        if self.bboxes_list:
            assert len(self.bboxes_list) == len(self.samples), f'{len(self.bboxes_list)=} {len(self.samples)=}'
        
    def cache_sample(self, index):
        sample = self.samples[index]
        cached_file = self.resolve_video(sample)  # caches the video
        # Return the size of the cached file
        return os.path.getsize(cached_file)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            try:
                if self.time_intervals:
                    interval = self.time_intervals[index]
                else:
                    interval = None
                buffer, clip_indices = self.loadvideo_decord(sample, interval)  # [T H W 3]
            except Exception as e:
                logger.warning(f'Error loading video {sample=}: {e.__class__} {e}')
                buffer = []
                clip_indices = []
            loaded_video = len(buffer) > 0
            if not loaded_video:
                logger.warning(f'Error loading video {sample=}, trying again')
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]
        
        if self.bboxes_list:
            try:
                sample_bboxes = self.bboxes_list[index]
            except Exception as e:
                logger.warning(f'Error loading bboxes for video {sample=}: {e.__class__} {e}, {index=}')
                raise e

            if len(sample_bboxes) > 0:
                bbox_idx = np.random.randint(len(sample_bboxes))
                sample_bbox = sample_bboxes[bbox_idx]
            else:
                sample_bbox = None

            if sample_bbox is not None:
                if self.bbox_format == "xyxy_relative":
                    width = sample_bbox[2] - sample_bbox[0]  # relative width
                    height = sample_bbox[3] - sample_bbox[1]  # relative height
                    width_increase = width * self.crop_increase_factor / 2
                    height_increase = height * self.crop_increase_factor / 2

                    # Compute expanded bbox coordinates while keeping them between 0 and 1
                    rel_x1 = max(0.0, sample_bbox[0] - width_increase)
                    rel_y1 = max(0.0, sample_bbox[1] - height_increase)
                    rel_x2 = min(1.0, sample_bbox[2] + width_increase)
                    rel_y2 = min(1.0, sample_bbox[3] + height_increase)

                    # Convert relative coordinates to absolute pixel values
                    img_height, img_width = buffer.shape[1:3]
                    abs_x1 = int(rel_x1 * img_width)
                    abs_y1 = int(rel_y1 * img_height)
                    abs_x2 = int(rel_x2 * img_width)
                    abs_y2 = int(rel_y2 * img_height)
                elif self.bbox_format == "tlwh_absolute":
                    width_increase = sample_bbox[2] * self.crop_increase_factor / 2
                    height_increase = sample_bbox[3] * self.crop_increase_factor / 2

                    # Compute expanded bbox coordinates in absolute coordinates
                    abs_x1 = int(max(0, sample_bbox[0] - width_increase))
                    abs_y1 = int(max(0, sample_bbox[1] - height_increase))
                    abs_x2 = int(min(buffer.shape[2], sample_bbox[0] + sample_bbox[2] + width_increase))
                    abs_y2 = int(min(buffer.shape[1], sample_bbox[1] + sample_bbox[3] + height_increase))
                else:
                    raise ValueError(f'Unsupported bbox format: {self.bbox_format}')
                
                buffer = buffer[:, abs_y1:abs_y2, abs_x1:abs_x2, :]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        if self.center_frame_only:
            # If center frame only, then take the center frame of each clip
            buffer = [
                np.repeat(clip[clip.shape[0] // 2:clip.shape[0] // 2+1], clip.shape[0]) 
                for clip in buffer]

        #print(f'Loaded video {sample=}, {label=}, {clip_indices=}, {index=}')

        return buffer, label, clip_indices, {'index': int(index)}
    
    def resolve_video(self, sample):

        if self.cache_dir is not None:
            #print(f'Caching video {sample=}')
            fname = self.cache_dir / sample
            if not os.path.exists(fname):
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with FileLock(str(fname) + '.lock'):
                    if not os.path.exists(fname):
                        tmp_path = str(fname) + '.tmp'
                        shutil.copy2(self.base_dir / sample, tmp_path)
                        os.rename(tmp_path, fname)

        else:
            fname = str(self.base_dir / sample)

        return str(fname)

    def loadvideo_decord(self, sample, interval=None):
        """ Load video content using Decord """

        fname = self.resolve_video(sample)

        if not os.path.exists(fname):
            logger.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            logger.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            logger.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception as e:
            logger.warn(f'Error loading video {fname=}, {e.__class__} {e}')
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                logger.warn(e)
        clip_len = int(fpc * fstp)

        if interval is not None:
            start_interval, end_interval = interval
            start_interval = max(0, start_interval)
            end_interval = min(len(vr), end_interval)
            sample_len = end_interval - start_interval
        else:
            sample_len = len(vr)

        if self.filter_short_videos and sample_len < clip_len:
            logger.warn(f'skipping video of length {sample_len} out of {len(vr)}: {fname=}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = sample_len // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, sample_len) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if sample_len > clip_len:
                        clip_step = (sample_len - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

            if interval is not None:
                # If interval is specified, then add the start_interval to all
                # indices to get the actual frame indices in the video
                all_indices = [i + start_interval for i in all_indices]

        try:
            buffer = vr.get_batch(all_indices).asnumpy()
        except Exception as e:
            buffer = load_video_cv2(fname, all_indices)
            logger.info(f'Error loading video {fname=}: {e.__class__} {e}, fallback to cv2')

        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)

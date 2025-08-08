import os
import json
from pathlib import Path

import torch
from fire import Fire
import pandas as pd
import numpy as np
from decord import VideoReader
import cv2
import torchvision.transforms.functional as F


CLIP_LEN = 16
GAP_THRESHOLD = 1

PANAF_IDX2ACTIONS = [
    "no_action",
    "walking",
    "standing",
    "sitting",
    "climbing_up",
    "hanging",
    "climbing_down",
    "running",
    "camera_interaction",
    "sitting_on_back",
]

PANAF_ACTIONS2IDX = {action: idx for idx, action in enumerate(PANAF_IDX2ACTIONS)}


class panaf500:

    def __init__(self, path) -> None:
        self.base_path = path
        self._splits = ["train", "validation", "test"]

        self.split_ids = {}

        for split_name in self._splits:
            files = os.listdir(os.path.join(self.base_path, "annotations", split_name))
            ids = [os.path.splitext(file)[0] for file in files if file.endswith(".json")]
            ids.sort()
            self.split_ids[split_name] = ids

    def list_split(self, split_name: str):
        return self.split_ids[split_name]

    def get_video_path(self, video_id: str):
        return os.path.join(self.base_path, "videos", f"{video_id}.mp4")

    def get_tracks_for_split(self, split_name, min_length_frames=0, gap_threshold=10):
        """
        get dictionary of tracks for all videos in a split. This skips videos with no tracks.

        Returns:
            tracks
                a list of tracks for each video in the split
            ids
                a list of video ids
        """

        tracks = []
        ids = []
        for video_id in self.list_split(split_name):
            this_tracks = self.get_tracks_for_video(video_id, min_length_frames, gap_threshold)

            if not this_tracks:
                # skip videos with no tracks
                continue

            ids.append(video_id)
            tracks.append(this_tracks)

        return tracks, ids

    def get_tracks_for_video(self, video_id, min_length_frames=0, gap_threshold=10):
        """ "
        get a track list for a single video

        The start_frame is 0-based, not 1-based as in the PanAf annotations

        Args:
            video_id: str
                The video id
            gap_threshold: int
                The maximum gap between two frames to consider them as part of the same track

        Returns:
            list of dict
                A list of tracks, each track is a dictionary containing the following keys:
                'ape_id': int
                    The ape id
                'start_frame': int
                    The frame id where the track starts
                'bboxes': torch.Tensor
                    The bounding boxes of the track (if there was a gap in the track, we continue the last bbox
                    and label until the monkey is registered again)
                'labels': torch.Tensor
                    The labels of the track
        """

        with open(
            os.path.join(
                self.base_path,
                "annotations",
                self.which_split(video_id),
                f"{video_id}.json",
            ),
            "r",
        ) as f:
            data = json.load(f)

        finished_tracks = []
        tracks = (
            {}
        )  # data structure: {ape_id: [(frame_id, bbox, behaviour) for each frame in track]}
        for frame in data["annotations"]:
            frame_id = int(frame["frame_id"])
            for det in frame["detections"]:
                if det["ape_id"] not in tracks:
                    tracks[det["ape_id"]] = []
                elif tracks[det["ape_id"]][-1][0] < frame_id - gap_threshold:
                    finished_tracks.append((det["ape_id"], tracks[det["ape_id"]]))
                    tracks[det["ape_id"]] = []
                else:
                    for id_ in range(tracks[det["ape_id"]][-1][0] + 1, frame_id):
                        # add dummy entries for missing frames
                        tracks[det["ape_id"]].append((id_, *tracks[det["ape_id"]][-1][1:]))
                tracks[det["ape_id"]].append(
                    (int(frame["frame_id"]), det["bbox"], det["behaviour"])
                )
        for ape_id, track in tracks.items():
            finished_tracks.append((ape_id, track))

        out_tracks = []
        for ape_id, track in finished_tracks:
            if len(track) < min_length_frames:
                continue

            bboxes = torch.empty((len(track), 4))
            labels = torch.empty((len(track),), dtype=torch.long)

            for idx, (frame_id, bbox, behaviour) in enumerate(track):
                bboxes[idx] = torch.tensor(bbox)
                labels[idx] = PANAF_ACTIONS2IDX[behaviour]

            track = {
                "ape_id": ape_id,
                "start_frame": track[0][0]
                - 1,  # PanAf starts counting from 1, but we use 0-based indexing
                "bboxes": bboxes,
                "labels": labels,
            }
            out_tracks.append(track)
        return out_tracks

    def which_split(self, video_id: str):
        for split in self._splits:
            if video_id in self.list_split(split):
                return split


def split_tracks_by_action(ids, tracks):

    flat_tracks = []

    for video_id, this_tracks in zip(ids, tracks):
        print(video_id, len(this_tracks))
        for track in this_tracks:

            previous_cut_idx = 0
            for idx in range(len(labels := track["labels"])):
                if idx + 1 >= len(labels) or labels[idx] != labels[idx + 1]:
                    # label chane, set a cut here
                    label = labels[idx]
                    assert (labels[previous_cut_idx : idx + 1] == label).all()
                    flat_tracks.append(
                        {
                            "video_id": video_id,
                            "ape_id": track["ape_id"],
                            "start_frame": track["start_frame"] + previous_cut_idx,
                            "label": label.item(),
                            "bboxes": track["bboxes"][previous_cut_idx : idx + 1],
                            "no_frames": idx + 1 - previous_cut_idx,
                        }
                    )
                    previous_cut_idx = idx + 1

    return flat_tracks


def split_track(track, split_len=64, stride=16):
    result = []
    for i in range(0, track["no_frames"] - split_len + 1, stride):
        result.append(
            {
                "video_id": track["video_id"],
                "ape_id": track["ape_id"],
                "start_frame": track["start_frame"] + i,
                "label": track["label"],
                "bboxes": track["bboxes"][i : i + split_len],
                "no_frames": split_len,
            }
        )
    return result


def get_enclosing_bbox(bboxes):
    # Convert bboxes tensor to numpy for easier computation
    bboxes_np = bboxes.numpy()

    # Get min/max coordinates across all bboxes
    x_min = bboxes_np[:, 0].min()
    y_min = bboxes_np[:, 1].min()
    x_max = bboxes_np[:, 2].max()
    y_max = bboxes_np[:, 3].max()

    # Increase size by 25% in each direction
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width * 0.25
    y_min -= height * 0.25
    x_max += width * 0.25
    y_max += height * 0.25

    return [x_min, y_min, x_max, y_max]


def process_video_segment(panaf, row, output_dir):
    # Construct video path
    video_path = panaf.get_video_path(row.video_id)

    # Create output directory structure
    out_path_video = Path(output_dir) / row.video_filename

    # Load video
    vr = VideoReader(video_path)

    # Extract frames
    frames = vr.get_batch(range(row.start_frame, row.start_frame + row.no_frames)).asnumpy()

    # Get center frame
    center_idx = len(frames) // 2
    center_frame = frames[center_idx]

    # Get bbox coordinates
    bbox = row.enclosing_bbox
    x1, y1, x2, y2 = map(int, bbox)

    # Get image dimensions
    height, width = frames[0].shape[:2]

    # Clip bbox coordinates to image boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_path_video), fourcc, 30, (x2 - x1, y2 - y1))

    # Process each frame
    for frame in frames:
        # Convert to torch tensor and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Crop frame
        cropped_frame = F.crop(frame_tensor, y1, x1, y2 - y1, x2 - x1)

        # Convert back to numpy and BGR format for OpenCV
        cropped_frame = (cropped_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cropped_frame_bgr = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)

        # Ensure there is no alpha channel in cropped_frame
        if cropped_frame.shape[2] == 4:
            cropped_frame = cropped_frame[:, :, :3]

        # Write frame
        video_writer.write(cropped_frame_bgr)

    # Release video writer
    video_writer.release()


def process_split(panaf, out_base, split):

    tracks, ids = panaf.get_tracks_for_split(split, min_length_frames=CLIP_LEN, gap_threshold=1)
    flat_tracks = split_tracks_by_action(ids, tracks)

    df = pd.DataFrame(flat_tracks)
    df["label_str"] = df["label"].map(lambda x: PANAF_IDX2ACTIONS[x])

    split_tracks = []

    for track in flat_tracks:
        if track["no_frames"] > CLIP_LEN:
            stride = CLIP_LEN  # if track["no_frames"] > 160 else 8
            split_tracks.extend(split_track(track, CLIP_LEN, stride))

    split_df = pd.DataFrame(split_tracks)
    split_df["label_str"] = split_df["label"].map(lambda x: PANAF_IDX2ACTIONS[x])

    split_df["enclosing_bbox"] = split_df["bboxes"].map(get_enclosing_bbox)
    split_df["video_filename"] = split_df.apply(
        lambda row: f"videos/{row.video_id}_{row.ape_id}_{row.start_frame}.mp4", axis=1
    )

    # Process all segments
    out_dir = Path(f"{out_base}/panaf500_paper_ar_{split}")
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    for idx, row in split_df.iterrows():
        if True:
            process_video_segment(panaf, row, out_dir)
            if idx % 100 == 0:
                print(f"Processed {idx} segments")

    split_df["label"] = split_df["label"] - 1

    split_df.to_csv(
        out_dir / f"panaf500_paper_ar_{split}.csv",
        index=False,
        header=False,
        sep=" ",
        columns=["video_filename", "label"],
    )


def main(panaf_path, out_base):

    panaf = panaf500(path=panaf_path)

    process_split(panaf, out_base, "validation")
    process_split(panaf, out_base, "train")
    process_split(panaf, out_base, "test")


if __name__ == "__main__":
    Fire(main)

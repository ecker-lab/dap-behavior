from functools import partial
import json
from multiprocessing import Pool
import multiprocessing
import os
import time
import cv2
import pandas as pd
from tqdm import tqdm


def extract_frames(
    output_path,
    rel_output_path,
    rel_video_path,
    stride_seconds,
    min_video_length_seconds,
    video_info,
):

    video_path = os.path.join(rel_video_path, video_info["path"])
    video_id = video_info["video_id"]
    width = video_info["width"]
    height = video_info["height"]

    if os.path.exists(f"{output_path}/{video_id:04d}.json"):
        # print(f"Skipping video {video_id} as it already has been processed")
        return

    out_infos = []
    bad_frames = []

    os.makedirs(output_path, exist_ok=True)

    def save():
        with open(f"{output_path}/{video_id:04d}.json", "w") as f:
            json.dump({"images": out_infos, "info": {"bad_frames": bad_frames}}, f, indent=4)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        bad_frames.append([video_id, -1, "Failed to open video"])
        print(f"Failed to open video {video_path}")
        save()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_frames = int(stride_seconds * fps)
    print(
        f"Extracting frames from video {video_path} with {frame_count} frames at {fps} fps and stride {stride_frames} frames"
    )

    if frame_count < 10 or frame_count < min_video_length_seconds * fps:
        bad_frames.append(
            [video_id, -1, f"Video is too short ({frame_count} frames)", frame_count]
        )
        print(f"Video {video_path} is too short ({frame_count} frames)")
        save()
        return

    frame_indices = list(range(0, frame_count, stride_frames))
    frame_indices.append(frame_count - 1)

    os.makedirs(output_path + f"/{video_id:04d}/", exist_ok=True)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_path, f"{video_id:04d}/{i:05d}.jpg"), frame)
            out_infos.append(
                {
                    "video_id": video_id,
                    "file_name": f"{rel_output_path}/{video_id:04d}/{i:05d}.jpg",
                    "video_path": video_info["path"],
                    "frame_index": i,
                    "frame_time": i / fps,
                    "width": width,
                    "height": height,
                }
            )
        else:
            print(f"Failed to read frame {i} of video {video_path}")
            bad_frames.append([video_id, i])
    cap.release()

    save()


def extract(
    *datasets, base_path, stride_seconds=1, num_workers=48, chunksize=5, min_video_length_seconds=2
):

    multiprocessing.set_start_method("spawn")

    print(f"Frame extraction with stride {stride_seconds} seconds")

    with Pool(num_workers) as p:

        for dataset in datasets:

            print(f"Extracting frames for dataset {dataset}")

            start_time = time.time()

            if not os.path.exists(f"{base_path}/pretrain/{dataset}/"):
                raise ValueError(f"Dataset {dataset} does not exist")

            func = partial(
                extract_frames,
                f"{base_path}/pretrain/{dataset}/frames",
                f"{dataset}/frames",
                base_path,
                stride_seconds,
                min_video_length_seconds,
            )

            videos = pd.read_csv(f"{base_path}/pretrain/{dataset}/videos.csv")

            video_infos = [t._asdict() for t in videos.itertuples()]

            for i in tqdm(
                p.imap_unordered(func, video_infos, chunksize=chunksize), total=len(video_infos)
            ):
                pass

            end_time = time.time()

            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.6f} seconds")


def merge(*datasets, base_path, ignore_missing=False):

    for dataset in datasets:

        print(f"Merging frames for dataset {dataset}")

        videos = pd.read_csv(f"{base_path}/pretrain/{dataset}/videos.csv")

        images = []
        bad_frames = []

        missing_files = False

        for video_id in tqdm(videos["video_id"]):
            if not os.path.exists(f"{base_path}/pretrain/{dataset}/frames/{video_id:04d}.json"):
                print(f"Video {video_id} is missing")
                missing_files = True
            else:
                with open(f"{base_path}/pretrain/{dataset}/frames/{video_id:04d}.json") as f:
                    data = json.load(f)
                    images += data["images"]
                    bad_frames += data["info"]["bad_frames"]

        if missing_files and not ignore_missing:
            raise ValueError("Some videos are missing")

        for idx, f in enumerate(images):
            f["id"] = idx

        out = {
            "info": {"no_frames": len(images), "no_videos": len(videos)},
            "images": images,
            "annotations": [],
            "categories": [{"id": 1, "name": "primate"}],
        }

        with open(f"{base_path}/pretrain/{dataset}/frames.json", "w") as f:
            json.dump(out, f, indent=4)

        with open(f"{base_path}/pretrain/{dataset}/bad_frames.json", "w") as f:
            json.dump(bad_frames, f, indent=4)


if __name__ == "__main__":
    import fire

    fire.Fire({"extract": extract, "merge": merge})

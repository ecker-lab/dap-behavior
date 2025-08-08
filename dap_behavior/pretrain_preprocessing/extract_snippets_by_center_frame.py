from functools import partial
import os
from pathlib import Path
import secrets
import subprocess
import time

import cv2
from tqdm import tqdm

from dap_behavior.utils.misc import load_json

FFMPEG_PATH = "ffmpeg"  # Adjust this if ffmpeg is not in your PATH

def extract_one_video(
    img_desc: dict, save_dir: Path, len_seconds: int, video_base_path: Path, threads: int
):
    video_path = video_base_path / img_desc["video_path"]
    start_time = img_desc["frame_time"] - len_seconds / 2
    end_time = img_desc["frame_time"] + len_seconds / 2

    save_path = (save_dir / f"{img_desc['file_name']}").with_suffix(".mp4")
    tmp_path = save_path.with_suffix(f".tmp.{secrets.token_hex(4)}.mp4")

    if save_path.exists():
        return img_desc["id"], "existed"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the FPS of the video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps == 0:
        fps = 24 # FPS extraction often fails for PanAf, but it is 24 for all videos

    print(f"Extracting {video_path} to {save_path}, fps: {fps}")

    ffmpeg_cmd = [
        FFMPEG_PATH,
        "-ss",
        str(start_time),
        "-i",
        str(video_path),
        "-to",
        str(len_seconds),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
    ]
    if fps > 35:
        ffmpeg_cmd += ["-vf", "fps=fps=30,setpts=PTS-STARTPTS"]
    ffmpeg_cmd += [
        "-threads",
        str(threads),
        "-an",  # disable audio
        "-y",
        str(tmp_path),
    ]

    print(" ".join(ffmpeg_cmd))

    ret = subprocess.run(
        ffmpeg_cmd,
        capture_output=True,
        text=True,
    )

    if ret.returncode != 0:
        print(ret.stderr)
        print(f"Failed to extract {video_path} to {save_path}")
        tmp_path.unlink(missing_ok=True)
    else:
        os.rename(tmp_path, save_path)

    return img_desc["id"], ret.returncode

def run_local(
    save_dir,
    path_to_json,
    video_base_path,
    len_seconds=3,
    n_samples=None,
    check_existence=True,
    n_workers=32,
    threads_per_worker=2,
):
    from multiprocessing import Pool

    images = load_json(path_to_json)["images"]

    print(f"Loaded {len(images)} entries")

    if check_existence:
        images = [i for i in images if not os.path.exists(os.path.join(save_dir, i["file_name"]))]

    print(f"Filtered to {len(images)} unprocessed entries")

    if n_samples is not None:
        images = images[:n_samples]

    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(
                extract_one_video,
                save_dir=Path(save_dir),
                len_seconds=len_seconds,
                video_base_path=Path(video_base_path),
                threads=threads_per_worker,
            ),
            images,
            chunksize=1,
        )):
            pass

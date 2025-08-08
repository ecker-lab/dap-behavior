import cv2
import pandas as pd
from pathlib import Path

video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".mts", ".webm")


def is_video_file(file_name):
    return file_name.lower().endswith(video_extensions)


def is_image_file(file_name):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    return file_name.lower().endswith(image_extensions)


def get_video_properties(folder_path, rel_path=None):
    video_data = []

    # Supported video extensions

    # Recursively iterate through all files
    for filepath in Path(folder_path).rglob("*"):
        if filepath.name.startswith("._"):
            continue

        if filepath.suffix.lower() in video_extensions:
            try:
                # Open video file
                cap = cv2.VideoCapture(str(filepath))

                # Get video properties
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Release video capture
                cap.release()

                # Add to list
                video_data.append(
                    {
                        "filename": filepath.name,
                        "path": (
                            str(filepath)
                            if rel_path is None
                            else str(filepath.relative_to(rel_path))
                        ),
                        "frame_count": frame_count,
                        "fps": fps,
                        "resolution": f"{width}x{height}",
                        "width": width,
                        "height": height,
                    }
                )

            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")

    # Create DataFrame
    df = pd.DataFrame(video_data)

    if df.empty:
        print("No video files found in the specified directory.")
        return df

    df["duration"] = df["frame_count"] / df["fps"]
    df["suffix"] = df["filename"].apply(lambda x: x.split(".")[-1].lower())
    print("total duration: ", df["duration"].sum() / 3600, " hours")
    return df

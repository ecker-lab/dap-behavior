import os
import json

from dap_behavior.pretrain_preprocessing.extract_snippets_by_center_frame import run_local
from dap_behavior.pretrain_preprocessing.utils import get_video_properties
from dap_behavior.pretrain_preprocessing.extract_frames import extract, merge
from dap_behavior.pretrain_preprocessing.gdino_coco import run_coco, nms
from dap_behavior.utils.misc import load_json


def main(dataset_name, dataset_video_source, base_path, stop_after=None):


    features_dir = base_path + f"pretrain/{dataset_name}/frame_features"

    if True:

        video_df = get_video_properties(dataset_video_source, base_path)

        video_df.reset_index(drop=True, inplace=True)
        video_df.reset_index(inplace=True)
        video_df.rename(columns={"index": "video_id"}, inplace=True)

        os.makedirs(base_path + f"pretrain/{dataset_name}", exist_ok=True)
        video_df.to_csv(base_path + f"pretrain/{dataset_name}/videos.csv", index=False)

        extract(dataset_name, base_path=base_path)
        merge(dataset_name, base_path=base_path)

    if True:
        os.makedirs(features_dir, exist_ok=True)
        run_coco(
            base_path + f"pretrain/{dataset_name}/frames.json",
            features_dir + "/gdino.json",
            prompt="monkey.primate.ape.",
            box_threshold=0.1,
            batch_size=20,
            base_path=base_path + f"pretrain/",
        )

    if True:

        nms(features_dir + "/gdino.json", features_dir + "/gdino_filtered.json", 0.5, 0.2)

    gdino = load_json(features_dir + "/gdino_filtered.json")

    images_with_dets = set(d["image_id"] for d in gdino["annotations"])

    for img in gdino["images"]:
        if img["id"] not in images_with_dets:
            # Skip images without detections
            continue
        img["file_name"] = "videos/" + img["file_name"].replace("/frames/", "/").replace(
            ".jpg", ".mp4"
        )

    with open(base_path + f"pretrain/{dataset_name}.json", "w") as f:
        json.dump(gdino, f, indent=4)

    run_local(
        base_path + f"/pretrain/",
        path_to_json=base_path + f"/pretrain/{dataset_name}.json",
        video_base_path=base_path,
        n_samples=stop_after,
        check_existence=True,
        n_workers=14,
        threads_per_worker=2,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

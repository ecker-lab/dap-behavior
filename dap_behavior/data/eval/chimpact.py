import json
import shutil

import pandas as pd
from fire import Fire


def process_split(base_path, split, downsample=True):
    gt = pd.read_csv(f"{base_path}/annotations/action/{split}_action.csv", header=None, sep=",")
    gt.columns = ["path", "frame", "x_min", "y_min", "x_max", "y_max", "label", "entity_id"]
    gt["filename"] = gt["path"].apply(lambda x: x.split("/")[-1])
    gt["path"] = gt["filename"] + ".mp4"

    if downsample:
        gt_ = gt.query("frame % 10 == 0")
    else:
        gt_ = gt
    gt_.drop(columns=["filename"]).to_csv(
        f"{base_path}/annotations/action/{split}_action{'_10' if downsample else ''}.ava.csv",
        header=None,
        index=None,
        sep=",",
    )


def prepare_chimpact(base_path):

    process_split(base_path, "val")
    process_split(base_path, "train")
    process_split(base_path, "test", downsample=False)

    action_list = [
        {"name": "moving", "id": 1, "label_type": "Locomotion"},
        {"name": "climbing", "id": 2, "label_type": "Locomotion"},
        {"name": "resting", "id": 3, "label_type": "Locomotion"},
        {"name": "sleeping", "id": 4, "label_type": "Locomotion"},
        {"name": "solitary object playing", "id": 5, "label_type": "Object Interaction"},
        {"name": "eating", "id": 6, "label_type": "Object Interaction"},
        {"name": "manipulating object", "id": 7, "label_type": "Object Interaction"},
        {"name": "grooming", "id": 8, "label_type": "Social Interaction"},
        {"name": "being groomed", "id": 9, "label_type": "Social Interaction"},
        {"name": "aggressing", "id": 10, "label_type": "Social Interaction"},
        {"name": "embracing", "id": 11, "label_type": "Social Interaction"},
        {"name": "begging", "id": 12, "label_type": "Social Interaction"},
        {"name": "being begged from", "id": 13, "label_type": "Social Interaction"},
        {"name": "taking object", "id": 14, "label_type": "Social Interaction"},
        {"name": "losing object", "id": 15, "label_type": "Social Interaction"},
        {"name": "carrying", "id": 16, "label_type": "Social Interaction"},
        {"name": "being carried", "id": 17, "label_type": "Social Interaction"},
        {"name": "nursing", "id": 18, "label_type": "Social Interaction"},
        {"name": "being nursed", "id": 19, "label_type": "Social Interaction"},
        {"name": "playing", "id": 20, "label_type": "Social Interaction"},
        {"name": "touching", "id": 21, "label_type": "Social Interaction"},
        {"name": "erection", "id": 22, "label_type": "Others"},
        {"name": "displaying", "id": 23, "label_type": "Others"},
    ]

    with open(f"{base_path}/annotations/action/action_list.json", "w") as fp:
        json.dump(action_list, fp)

    shutil.copy(
        f"{base_path}/annotations/action/val_action_excluded_timestamps.csv",
        f"{base_path}/annotations/action/val_action_10.ava_excluded_timestamps.csv",
    )
    shutil.copy(
        f"{base_path}/annotations/action/train_action_excluded_timestamps.csv",
        f"{base_path}/annotations/action/train_action_10.ava_excluded_timestamps.csv",
    )
    shutil.copy(
        f"{base_path}/annotations/action/test_action_excluded_timestamps.csv",
        f"{base_path}/annotations/action/test_action.ava_excluded_timestamps.csv",
    )


if __name__ == "__main__":
    Fire(prepare_chimpact)

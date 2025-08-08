from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
import json


class CocoDataset(Dataset):
    def __init__(
        self,
        json_file,
        transform=None,
        base_path="",
        skip_image_ids={},
        rank=0,
        world_size=1,
    ):
        with open(json_file, "r") as f:
            imgs = json.load(f)["images"]

            self.img_paths = [img for img in imgs if img["id"] not in skip_image_ids]
            self.img_paths = self.img_paths[rank::world_size]
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """

        Returns
        -------
        image: torch.Tensor
            The image in HWC format. This equals the format the GroundingDINO preprocessor expects.
        """
        img_info = self.img_paths[idx]
        path = Path(self.base_path) / img_info["file_name"]
        image = read_image(str(path)).permute(1, 2, 0)

        if image.shape[-1] > 3:  # get rid of alpha channel if present
            image = image[:, :, :3]

        if self.transform:
            image = self.transform(image)

        # do not have gt bboxes or labels
        return image, img_info

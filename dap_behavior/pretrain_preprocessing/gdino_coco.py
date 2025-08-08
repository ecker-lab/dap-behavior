
from collections import defaultdict
from functools import partial
import json
import os
from einops import rearrange
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.image_transforms import center_to_corners_format
from fire import Fire
from torchvision.transforms import functional as F

from dap_behavior.utils.coco_dataset import CocoDataset
from dap_behavior.utils.nms import nms_cpu


class ResizeTo1080p:
    def __init__(self):
        self.max_size = 1080

    def __call__(self, img):
        # Get current dimensions
        height, width = img.shape[:2]

        # Calculate scaling factor
        scale = self.max_size / max(width, height)

        if scale >= 1:
            return img

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image
        img = F.resize(rearrange(img, "h w c -> c h w"), (new_height, new_width), antialias=True)
        img = rearrange(img, "c h w -> h w c")
        return img


def gdino_preprocess_collate_fn(processor, prompt, batch):
    imgs, img_info = zip(*batch)
    inputs = processor(imgs, [prompt] * len(imgs), return_tensors="pt")
    inputs["_img_infos"] = list(img_info)

    return inputs


def merge_output_files(path_to_json, save_path, info={}):

    with open(save_path + ".nl", "r") as f:
        data = [json.loads(line) for line in tqdm(f.readlines())]

    with open(save_path + "_processed_images.txt", "r") as f:
        processed_images = [int(line) for line in f.readlines()]
        processed_images = set(processed_images)

    with open(path_to_json, "r") as f:
        coco_file = json.load(f)

    out_data = {
        "info": {
            **coco_file["info"],
            **info,
            "no_detections": len(data),
        },
        "images": [img for img in coco_file["images"] if img["id"] in processed_images],
        "annotations": data,
        "categories": coco_file["categories"],
    }

    if len(out_data["images"]) < len(coco_file["images"]):
        print(f"Missing {len(coco_file['images']) - len(out_data['images'])} images!")

    print(f"Produced {len(data)} bounding boxes.")

    with open(save_path, "w") as f:
        json.dump(out_data, f)

    print("Done!")


@torch.no_grad()
def nms(path_to_json, save_path, iou_threshold, score_threshold, device="cpu"):
    """
    Apply non-maximum suppression to a COCO format model output.
    """

    with open(path_to_json, "r") as f:
        data = json.load(f)

    os.makedirs(Path(save_path).parent, exist_ok=True)

    bboxes_per_image = defaultdict(list)
    scores_per_image = defaultdict(list)
    annots_per_image = defaultdict(list)

    for d in data["annotations"]:
        if d["score"] < score_threshold:
            continue
        bboxes_per_image[d["image_id"]].append(d["bbox"])
        scores_per_image[d["image_id"]].append(d["score"])
        annots_per_image[d["image_id"]].append(d)

    output = []

    for image_id in tqdm(bboxes_per_image.keys(), total=len(bboxes_per_image)):

        if device == "cpu":
            bboxes = np.array(bboxes_per_image[image_id])
            scores = np.array(scores_per_image[image_id])
            keep = nms_cpu(bboxes, scores, iou_threshold)
        else:
            bboxes = torch.tensor(bboxes_per_image[image_id])
            scores = torch.tensor(scores_per_image[image_id])
            keep = torch.ops.torchvision.nms(bboxes.to(device), scores.to(device), iou_threshold)

        for k in keep:
            output.append(
                annots_per_image[image_id][k]
            )

    data["annotations"] = output
    data["info"]["nms_iou_threshold"] = iou_threshold
    data["info"]["nms_score_threshold"] = score_threshold
    data["info"]["no_detections_after_nms"] = len(output)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


class GroundingDino:

    def __init__(self, model_id, device, use_amp):
        super().__init__()
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.gdino_processor = AutoProcessor.from_pretrained(model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.use_amp = use_amp
        self.model_id = model_id

        torch.set_float32_matmul_precision("medium" if use_amp else "high")

    def run_coco(
        self,
        path_to_json,
        save_path,
        prompt,
        box_threshold,
        base_path,
        batch_size,
        num_workers,
        save_every_niter,
        category_id,
    ):
        """
        Apply GDINO to a COCO format dataset.

        Setup for A100 80GB: batch_size=40, num_workers=2 or 4. Takes 3:30 minutes for 3000 images.

        Args:
            path_to_json (str): Path to COCO format JSON file.
            save_path (str): Path to save the results.
            prompt (str): Prompt for grounding.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for DataLoader.
            box_threshold (float): Threshold for bounding box confidence.
            save_every_niter (int): Save results every n iterations.
        """

        print(f"Loading dataset from {path_to_json}")

        try:
            with open(save_path + "_processed_images.txt", "r") as f:
                processed_images = [int(line) for line in f.readlines()]
                processed_images = set(processed_images)
        except FileNotFoundError:
            processed_images = set()

        dataset = CocoDataset(
            path_to_json,
            base_path=base_path,
            skip_image_ids=processed_images,
            transform=ResizeTo1080p(),
        )

        if len(dataset) == 0:
            print("No images to process!")
            return

        collate_fn = partial(gdino_preprocess_collate_fn, self.gdino_processor, prompt)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True,
            pin_memory_device=str(self.device),
        )

        os.makedirs(Path(save_path).parent, exist_ok=True)
        print(f"Saving results to {save_path}")

        self._apply_gdino(loader, save_path, box_threshold, save_every_niter, category_id)

        info = {
            "gdino_prompt": prompt,
            "gdino_model": self.model_id,
            "gdino_box_threshold": box_threshold,
            "gdino_use_amp": self.use_amp,
        }

        merge_output_files(path_to_json, save_path, info=info)

    @torch.no_grad()
    def _apply_gdino(self, loader, save_path, box_threshold, save_every_niter, category_id):

        fp = open(save_path + ".nl", "a")

        fp2 = open(save_path + "_processed_images.txt", "a")

        detections = []
        processed_images = []

        det_id = 0

        def save_results():
            nonlocal detections
            nonlocal processed_images
            for output in detections:
                fp.write(json.dumps(output) + "\n")
            for img_id in processed_images:
                fp2.write(f"{img_id}\n")
            fp.flush()
            fp2.flush()
            detections = []
            processed_images = []

        for idx, batch in enumerate(tqdm(loader)):
            with torch.autocast("cuda", enabled=self.use_amp):
                img_infos = batch.pop("_img_infos")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.gdino_model(**batch)

                logits, boxes = outputs.logits, outputs.pred_boxes
                probs = torch.sigmoid(logits)  # (batch_size, num_queries=900, 256)
                scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

                # Convert to [x0, y0, x1, y1] format
                boxes = center_to_corners_format(boxes)  # (batch_size, num_queries, 4)

            boxes = boxes.detach().cpu()
            scores = scores.detach().cpu()

            for i, img_info in enumerate(img_infos):
                mask = scores[i] > box_threshold
                boxes_i = boxes[i][mask]  # (n_bboxes, 4)
                scores_i = scores[i][mask]  # (n_bboxes)

                boxes_i = torch.stack(
                    [
                        boxes_i[:, 0] * img_info["width"],
                        boxes_i[:, 1] * img_info["height"],
                        (boxes_i[:, 2] - boxes_i[:, 0]) * img_info["width"],
                        (boxes_i[:, 3] - boxes_i[:, 1]) * img_info["height"],
                    ],
                    dim=-1,
                )

                for box, score in zip(boxes_i, scores_i):
                    detections.append(
                        {
                            "image_id": img_info["id"],
                            "bbox": box.tolist(),
                            "score": score.item(),
                            "category_id": category_id,
                            "iscrowd": 0,
                            "area": (box[2] * box[3]).item(),
                            "id": det_id,
                        }
                    )
                    det_id += 1

                processed_images.append(img_info["id"])

            if (idx + 1) % save_every_niter == 0:
                save_results()

        save_results()

        fp.close()
        print("Done!")


def run_coco(
    path_to_json,
    save_path,
    prompt,
    box_threshold,
    base_path="",
    batch_size=32,
    num_workers=4,
    save_every_niter=10,
    category_id=1,
    model_id="IDEA-Research/grounding-dino-base",
    device="cuda",
    use_amp=True,
):
    gdino = GroundingDino(model_id=model_id, device=device, use_amp=use_amp)
    gdino.run_coco(
        path_to_json,
        save_path,
        prompt,
        box_threshold,
        base_path=base_path,
        batch_size=batch_size,
        num_workers=num_workers,
        save_every_niter=save_every_niter,
        category_id=category_id,
    )


if __name__ == "__main__":
    Fire({"run_coco": run_coco, "merge_output_files": merge_output_files, "nms": nms})
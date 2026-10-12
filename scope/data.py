from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


class CocoBoxCropDataset(Dataset):
    """
    COCO-format image dataset returning:
        image: [3, crop_size, crop_size], float in [0, 1]
        mask:  [1, crop_size, crop_size], object box mask in {0, 1}

    This does not require pycocotools.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_json: str | Path,
        crop_size: int = 512,
        object_crop_prob: float = 0.90,
        crop_margin: float = 1.8,
        min_box_area: float = 4.0,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_json = Path(annotations_json)
        self.crop_size = int(crop_size)
        self.object_crop_prob = float(object_crop_prob)
        self.crop_margin = float(crop_margin)
        self.min_box_area = float(min_box_area)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images_dir: {self.images_dir}")

        if not self.annotations_json.exists():
            raise FileNotFoundError(f"Missing annotations_json: {self.annotations_json}")

        with open(self.annotations_json, "r") as f:
            coco = json.load(f)

        self.images = []
        self.image_id_to_info = {}

        for im in coco["images"]:
            image_id = int(im["id"])
            file_name = im["file_name"]
            width = int(im.get("width", 0))
            height = int(im.get("height", 0))
            info = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
            self.images.append(info)
            self.image_id_to_info[image_id] = info

        self.anns_by_image: Dict[int, List[List[float]]] = defaultdict(list)

        for ann in coco.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue

            image_id = int(ann["image_id"])
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1 or (w * h) < self.min_box_area:
                continue

            self.anns_by_image[image_id].append([x, y, w, h])

        self.object_indices = [
            idx
            for idx, im in enumerate(self.images)
            if len(self.anns_by_image[int(im["id"])]) > 0
        ]

        if len(self.images) == 0:
            raise RuntimeError("No images found in COCO JSON.")

        print(
            f"Loaded COCO dataset: images={len(self.images)} "
            f"images_with_boxes={len(self.object_indices)} "
            f"annotations={sum(len(v) for v in self.anns_by_image.values())}"
        )

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).contiguous()

    def _choose_index(self) -> int:
        if (
            len(self.object_indices) > 0
            and random.random() < self.object_crop_prob
        ):
            return random.choice(self.object_indices)
        return random.randrange(len(self.images))

    def _compute_crop_box(
        self,
        W: int,
        H: int,
        boxes: Sequence[Sequence[float]],
    ) -> Tuple[int, int, int, int]:
        # Fallback random square-ish crop.
        if len(boxes) == 0 or random.random() > self.object_crop_prob:
            side = min(W, H)
            if side <= 0:
                return 0, 0, W, H
            if W == side:
                x0 = 0
            else:
                x0 = random.randint(0, max(0, W - side))
            if H == side:
                y0 = 0
            else:
                y0 = random.randint(0, max(0, H - side))
            return x0, y0, x0 + side, y0 + side

        x, y, w, h = random.choice(list(boxes))
        cx = x + 0.5 * w
        cy = y + 0.5 * h

        side = max(w, h) * self.crop_margin
        side = max(side, 64.0)
        side = min(side, float(max(W, H)))

        # Jitter around object center.
        jitter = 0.15 * side
        cx += random.uniform(-jitter, jitter)
        cy += random.uniform(-jitter, jitter)

        x0 = int(round(cx - 0.5 * side))
        y0 = int(round(cy - 0.5 * side))
        x1 = int(round(cx + 0.5 * side))
        y1 = int(round(cy + 0.5 * side))

        # Clamp while preserving crop size as much as possible.
        crop_w = x1 - x0
        crop_h = y1 - y0

        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > W:
            shift = x1 - W
            x0 -= shift
            x1 = W
        if y1 > H:
            shift = y1 - H
            y0 -= shift
            y1 = H

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, max(x0 + 1, x1))
        y1 = min(H, max(y0 + 1, y1))

        return x0, y0, x1, y1

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        # Ignore the dataloader index so every access gets a random crop.
        idx = self._choose_index()
        info = self.images[idx]
        image_id = int(info["id"])
        path = self.images_dir / info["file_name"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Robust fallback: pick another image.
            idx = random.randrange(len(self.images))
            info = self.images[idx]
            image_id = int(info["id"])
            path = self.images_dir / info["file_name"]
            img = Image.open(path).convert("RGB")

        W, H = img.size
        boxes = self.anns_by_image.get(image_id, [])

        mask_full = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask_full)
        for x, y, w, h in boxes:
            x0 = max(0, int(math.floor(x)))
            y0 = max(0, int(math.floor(y)))
            x1 = min(W, int(math.ceil(x + w)))
            y1 = min(H, int(math.ceil(y + h)))
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1, y1], fill=255)

        crop_box = self._compute_crop_box(W, H, boxes)
        img = img.crop(crop_box)
        mask = mask_full.crop(crop_box)

        img = img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)

        image_t = self._pil_to_tensor(img)
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        mask_t = torch.from_numpy(mask_arr[None, :, :]).contiguous()

        return {
            "image": image_t,
            "mask": mask_t,
        }

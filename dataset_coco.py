from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from utils import load_json, xywh_to_xyxy


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    path: Path


class CocoDataset:
    def __init__(self, images_dir: str, annotations_json: str):
        self.images_dir = Path(images_dir)
        self.annotations_json = Path(annotations_json)
        self.coco = load_json(self.annotations_json)

        self.images: List[ImageRecord] = []
        for image in self.coco["images"]:
            self.images.append(
                ImageRecord(
                    image_id=int(image["id"]),
                    file_name=str(image["file_name"]),
                    width=int(image["width"]),
                    height=int(image["height"]),
                    path=self.images_dir / image["file_name"],
                )
            )

        self.anns_by_image: Dict[int, List[dict]] = {}
        for ann in self.coco.get("annotations", []):
            image_id = int(ann["image_id"])
            self.anns_by_image.setdefault(image_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def get_gt_boxes_xyxy(self, image_id: int) -> List[List[float]]:
        boxes: List[List[float]] = []
        for ann in self.anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0) == 1:
                continue
            boxes.append(xywh_to_xyxy(ann["bbox"]))
        return boxes
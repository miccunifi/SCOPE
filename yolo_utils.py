from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from utils import iou_xyxy, xyxy_to_xywh


class YoloPredictor:
    def __init__(self, model_path: str, device: int | str | None = None):
        self.model = YOLO(model_path)
        self.device = device

    def predict_boxes_xyxy(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 100,
    ) -> List[List[float]]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=self.device,
        )

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        return boxes.astype(float).tolist()

    def predict_full(
        self,
        image_path: str | Path,
        image_id: int,
        category_id_offset: int = 1,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> List[Dict]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=self.device,
        )

        if not results or results[0].boxes is None:
            return []

        result = results[0]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        predictions: List[Dict] = []

        for box, score, cls_id in zip(boxes_xyxy, scores, classes):
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cls_id + category_id_offset),
                    "bbox": [float(v) for v in xyxy_to_xywh(box.tolist())],
                    "score": float(score),
                    "_xyxy": [float(v) for v in box.tolist()],
                }
            )

        return predictions


def remove_internal_xyxy(predictions: List[Dict]) -> List[Dict]:
    clean = []
    for p in predictions:
        q = dict(p)
        q.pop("_xyxy", None)
        clean.append(q)
    return clean
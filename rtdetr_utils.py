from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import RTDETR


def remove_internal_xyxy(preds: List[Dict]) -> List[Dict]:
    clean = []
    for p in preds:
        q = dict(p)
        q.pop("_xyxy", None)
        return_pred = q
        clean.append(return_pred)
    return clean


class RTDetrPredictor:
    """
    Ultralytics RT-DETR predictor with YoloPredictor-compatible API.

    Exports COCO predictions:
      bbox = [x, y, width, height] in pixel coordinates
      category_id = cls + category_id_offset
    """

    def __init__(
        self,
        weights: str,
        device: str | None = "cuda",
        imgsz: int = 640,
        verbose: bool = False,
    ):
        self.weights = str(weights)
        self.device = device
        self.imgsz = int(imgsz)
        self.verbose = bool(verbose)
        self.model = RTDETR(self.weights)

    def predict_full(
        self,
        image_path: str | Path,
        image_id: int,
        category_id_offset: int = 1,
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> List[Dict]:
        image_path = Path(image_path)

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)

        img_h, img_w = img.shape[:2]

        result = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            device=self.device,
            verbose=self.verbose,
        )[0]

        preds: List[Dict] = []

        if result.boxes is None or len(result.boxes) == 0:
            return preds

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)

        for box, score, cls_id in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = map(float, box)

            x1 = float(np.clip(x1, 0, img_w - 1))
            x2 = float(np.clip(x2, 0, img_w - 1))
            y1 = float(np.clip(y1, 0, img_h - 1))
            y2 = float(np.clip(y2, 0, img_h - 1))

            w = x2 - x1
            h = y2 - y1

            if w <= 1e-3 or h <= 1e-3:
                continue

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cls_id) + int(category_id_offset),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                    "_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return preds
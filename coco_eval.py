from __future__ import annotations

from pathlib import Path
from typing import Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco_map(
    gt_json: str | Path,
    pred_json: str | Path,
    iou_type: str = "bbox",
) -> Dict[str, float]:
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))

    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats

    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }
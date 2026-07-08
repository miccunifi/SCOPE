from __future__ import annotations

from typing import Dict, List

from utils import box_area_xyxy, iou_xyxy


def intersection_over_smaller(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(box_area_xyxy(a), box_area_xyxy(b))

    if smaller <= 0:
        return 0.0
    return inter / smaller


def union_box(a: List[float], b: List[float]) -> List[float]:
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def merge_boxes_by_intersection(
    boxes: List[List[float]],
    threshold: float = 0.10,
    max_iterations: int = 100,
) -> List[List[float]]:
    if not boxes:
        return []

    boxes = [list(map(float, b)) for b in boxes]

    if threshold < 0:
        return boxes

    for _ in range(max_iterations):
        changed = False
        used = [False] * len(boxes)
        new_boxes: List[List[float]] = []

        for i in range(len(boxes)):
            if used[i]:
                continue

            current = boxes[i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                if intersection_over_smaller(current, boxes[j]) >= threshold:
                    current = union_box(current, boxes[j])
                    used[j] = True
                    changed = True

            new_boxes.append(current)

        boxes = new_boxes

        if not changed:
            break

    return boxes


def roi_quality_by_scale(
    box_xyxy: List[float],
    width: int,
    height: int,
    tiny_thr: float = 0.001,
    small_thr: float = 0.005,
    medium_thr: float = 0.02,
    tiny_quality: int = 85,
    small_quality: int = 75,
    medium_quality: int = 65,
    large_quality: int = 55,
) -> int:
    area_ratio = box_area_xyxy(box_xyxy) / max(1.0, float(width * height))

    if area_ratio < tiny_thr:
        return tiny_quality
    if area_ratio < small_thr:
        return small_quality
    if area_ratio < medium_thr:
        return medium_quality
    return large_quality


def proposal_recall(
    gt_boxes: List[List[float]],
    proposal_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> float:
    if not gt_boxes:
        return 1.0

    matched = 0
    for gt in gt_boxes:
        best = 0.0
        for prop in proposal_boxes:
            best = max(best, iou_xyxy(gt, prop))
        if best >= iou_threshold:
            matched += 1

    return matched / len(gt_boxes)


def proposal_recall_by_size(
    gt_boxes: List[List[float]],
    proposal_boxes: List[List[float]],
    width: int,
    height: int,
    iou_threshold: float = 0.5,
    small_thr: float = 0.005,
    medium_thr: float = 0.02,
) -> Dict[str, float]:
    image_area = max(1.0, float(width * height))

    groups = {
        "small": [],
        "medium": [],
        "large": [],
    }

    for gt in gt_boxes:
        ratio = box_area_xyxy(gt) / image_area
        if ratio < small_thr:
            groups["small"].append(gt)
        elif ratio < medium_thr:
            groups["medium"].append(gt)
        else:
            groups["large"].append(gt)

    return {
        f"proposal_recall_{name}": proposal_recall(boxes, proposal_boxes, iou_threshold)
        for name, boxes in groups.items()
    }
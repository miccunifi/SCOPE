from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f)


def read_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def write_image(path, image):
    from pathlib import Path
    import cv2
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image is None:
        raise RuntimeError(f"Could not write image because image is None: {path}")

    if not isinstance(image, np.ndarray):
        raise RuntimeError(f"Could not write image because image is not numpy array: {path}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    ok = cv2.imwrite(str(path), image)

    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def jpeg_encode_decode(img_bgr: np.ndarray, quality: int) -> Tuple[np.ndarray, int]:
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded, int(len(encoded))


def jpeg_encode_bytes(img_bgr: np.ndarray, quality: int) -> bytes:
    quality = int(np.clip(quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def file_size_bytes(path: str | Path) -> int:
    return os.path.getsize(path)


def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def clip_box_xyxy(box: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return [x1, y1, x2, y2]


def expand_box_xyxy(box: List[float], margin: float, width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    return clip_box_xyxy(
        [x1 - bw * margin, y1 - bh * margin, x2 + bw * margin, y2 + bh * margin],
        width,
        height,
    )


def box_area_xyxy(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area_xyxy(a) + box_area_xyxy(b) - inter

    if union <= 0:
        return 0.0
    return inter / union


def resize_down_up(img_bgr: np.ndarray, scale: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored
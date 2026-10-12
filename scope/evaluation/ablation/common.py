from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


CODEC_ORDER = ["jpeg", "bpg", "bmshj", "cheng"]
EXCLUDED_CODECS = {"jpeg2000"}
EXCLUDED_FAMILIES = {"wavelet"}
METRIC_KEYS = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]
X_KEY_LABELS = {
    "dataset_bpp": "bits per pixel (bpp)",
    "avg_bpp_per_image": "average bpp per image",
    "avg_kb_per_image": "average KB per image",
}
DETECTOR_PRETTY = {
    "fastrcnn": "Faster R-CNN",
    "fasterrcnn": "Faster R-CNN",
    "faster-rcnn": "Faster R-CNN",
    "frcnn": "Faster R-CNN",
    "resnext": "Faster R-CNN",
    "rtdetr": "RT-DETR",
    "rt-detr": "RT-DETR",
    "yolo": "YOLO",
}


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def to_float_or_blank(x: Any) -> Any:
    if x in [None, ""]:
        return ""
    try:
        return float(x)
    except Exception:
        return ""


def canonical_detector_label(label: str) -> str:
    k = norm(label)

    if any(x in k for x in ["fastrcnn", "fasterrcnn", "faster-rcnn", "frcnn", "resnext"]):
        return "fastrcnn"
    if any(x in k for x in ["rtdetr", "rt-detr"]):
        return "rtdetr"
    if "yolo" in k:
        return "yolo"

    return str(label).strip()


def canonical_method_key(label: str) -> str:
    k = norm(label)

    if k in {"main", "scope", "full", "default", "base"}:
        return "full"

    if k in {
        "task", "no_task", "notask", "wo_task", "without_task",
        "no_detection", "no_det", "wodet", "wo_det",
        "without_det", "no_detector", "without_detector",
    }:
        return "no_task"

    if k in {
        "bg", "no_bg", "nobg", "wo_bg", "without_bg",
        "no_background", "without_background",
        "no_background_simplification",
        "wo_background_simplification",
        "without_background_simplification",
        "no_bg_rate", "wo_bg_rate", "without_bg_rate",
    }:
        return "no_bg"

    return k or "unknown"


def parse_labeled_path(s: str) -> Tuple[str, str, Path]:
    """
    Accepted formats:
      fastrcnn=path/to/main_folder
      full:fastrcnn=path/to/full/main_folder
      no_task:fastrcnn=path/to/no_task/main_folder
      no_bg:fastrcnn=path/to/no_bg/main_folder
    """
    if "=" in s:
        label, path_str = s.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
    else:
        path = Path(s)
        label = path.name

    if ":" in label:
        method_label, detector_label = label.split(":", 1)
    elif "/" in label:
        method_label, detector_label = label.split("/", 1)
    else:
        method_label, detector_label = "full", label

    method_label = method_label.strip() or "full"
    detector_label = canonical_detector_label(detector_label.strip() or path.name)

    return method_label, detector_label, path


def pretty_detector(label: str) -> str:
    k = norm(label)
    for needle, pretty in DETECTOR_PRETTY.items():
        if needle in k:
            return pretty
    return str(label)


def pretty_method(label: str) -> str:
    key = canonical_method_key(label)
    aliases = {
        "full": "SCOPE",
        "no_task": "w/o task",
        "no_bg": "w/o bg/rate",
        "baseline": "Baseline",
    }
    return aliases.get(key, str(label).strip().replace("_", " "))


def codec_to_family(codec: str) -> str:
    codec = norm(codec)

    if codec == "jpeg":
        return "block_dct"
    if codec == "bpg":
        return "intra_smooth"
    if codec in {"bmshj", "cheng"}:
        return "learned"
    if codec == "jpeg2000":
        return "wavelet"

    return "unknown"


def normalize_family(x: Any) -> str:
    x = norm(x)

    if x in {"jpeg", "diff_jpeg", "block_dct"}:
        return "block_dct"
    if x in {"bpg", "bpg_like", "intra_smooth"}:
        return "intra_smooth"
    if x in {"bmshj", "cheng", "cheng2020", "compressai", "learned", "mbt"}:
        return "learned"
    if x in {"jpeg2000", "jp2", "jp2_like", "wavelet"}:
        return "wavelet"

    return x or "unknown"


def methods_for_codec(codec: str, methods: list[str] | tuple[str, ...]) -> list[str]:
    if codec == "cheng":
        return ["full"] if "full" in methods else []
    return list(methods)

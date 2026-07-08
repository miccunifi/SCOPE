#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


CODEC_PRETTY = {
    "jpeg": "JPEG",
    "bpg": "BPG",
    "bmshj": "BMSHJ",
    "cheng": "Cheng",
    "jpeg2000": "JPEG2000",
    "unknown": "unknown",
}


DETECTOR_PRETTY = {
    "fastrcnn": "Faster R-CNN",
    "fasterrcnn": "Faster R-CNN",
    "faster-rcnn": "Faster R-CNN",
    "frcnn": "Faster R-CNN",
    "resnext": "Faster R-CNN",
    "x101": "Faster R-CNN",
    "detectron2": "Faster R-CNN",
    "rtdetr": "RT-DETR",
    "rt-detr": "RT-DETR",
    "yolo": "YOLO",
}


def read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def safe_float(x: Any) -> Any:
    try:
        if x in [None, ""]:
            return ""
        return float(x)
    except Exception:
        return ""


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def quantize_float(x: float, ndigits: int = 6) -> float:
    return round(float(x), ndigits)


def parse_labeled_path(s: str) -> Tuple[Optional[str], Path]:
    """
    Supports:
      path/to/folder
      detector_label=path/to/folder

    Example:
      fastrcnn=outputs_bpp_coco_detectron2_fasterrcnn_x101_fpn
      rtdetr=outputs_bpp_coco_rtdetr_fpn
      yolo=outputs_conditional
    """
    if "=" in s:
        label, path = s.split("=", 1)
        return label.strip(), Path(path.strip())
    return None, Path(s)


def pretty_detector(label: str) -> str:
    k = norm(label)
    for needle, pretty in DETECTOR_PRETTY.items():
        if needle in k:
            return pretty
    return str(label)


def infer_detector_from_path(path: Path, provided_label: Optional[str]) -> str:
    if provided_label:
        return pretty_detector(provided_label)

    text = norm(str(path))
    for needle, pretty in DETECTOR_PRETTY.items():
        if needle in text:
            return pretty

    return "unknown"


def load_summary_near_predictions(pred_path: Path) -> Dict[str, Any]:
    summary_path = pred_path.parent / "summary.json"
    if summary_path.exists():
        try:
            return read_json(summary_path)
        except Exception:
            return {}
    return {}


def infer_curve_type(pred_path: Path, preedit_prefix: str = "preedit_") -> Tuple[str, str]:
    """
    Example:
      .../jpeg_q10/predictions_coco.json         -> baseline, jpeg_q10
      .../preedit_jpeg_q10/predictions_coco.json -> preedit,  jpeg_q10
    """
    name = pred_path.parent.name
    if name.startswith(preedit_prefix):
        return "preedit", name[len(preedit_prefix):]
    return "baseline", name


def infer_codec_from_text(text: str) -> str:
    text = norm(text)

    # Order matters.
    if "jpeg2000" in text or "jp2" in text or "j2k" in text or "wavelet" in text:
        return "jpeg2000"
    if "cheng" in text or "cheng2020" in text:
        return "cheng"
    if "bmshj" in text or "mbt" in text or "hyperprior" in text:
        return "bmshj"
    if "bpg" in text:
        return "bpg"
    if "diff_jpeg" in text or "diffjpeg" in text or "jpeg" in text or "jpg" in text or "block_dct" in text:
        return "jpeg"

    return "unknown"


def infer_codec(pred_path: Path, summary: Dict[str, Any], base_name: str) -> str:
    text = " ".join(
        str(x)
        for x in [
            base_name,
            pred_path.parent.name,
            pred_path.parent.parent.name,
            pred_path,
            summary.get("experiment", ""),
            summary.get("codec", ""),
            summary.get("codec_family", ""),
            summary.get("actual_codec_family", ""),
            summary.get("condition_codec_family", ""),
            summary.get("compressai_model", ""),
            summary.get("mode", ""),
        ]
    )
    return infer_codec_from_text(text)


def extract_first_number(patterns: List[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def get_first_numeric(summary: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in summary and summary[key] not in [None, ""]:
            try:
                return float(summary[key])
            except Exception:
                pass
    return None


def infer_raw_level_from_filename(pred_path: Path, base_name: str, codec: str) -> Tuple[Optional[float], str]:
    """
    Infer codec-native compression value from folder/file names.

    Expected examples:
      jpeg_q10              -> raw_level=10, jpeg_quality
      preedit_jpeg_q10      -> raw_level=10, jpeg_quality
      bpg_qp44              -> raw_level=44, bpg_qp
      preedit_bpg_qp44      -> raw_level=44, bpg_qp
      bmshj_q3              -> raw_level=3, compressai_quality
      preedit_bmshj_q3      -> raw_level=3, compressai_quality
    """
    text = norm(
        " ".join(
            str(x)
            for x in [
                base_name,
                pred_path.parent.name,
                pred_path.parent.parent.name,
                pred_path,
            ]
        )
    )

    if codec == "jpeg":
        patterns = [
            r"(?:^|[_/\-])jpeg[_\-]?q(?:uality)?[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])jpg[_\-]?q(?:uality)?[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])q([0-9]+(?:\.[0-9]+)?)",
        ]
        return extract_first_number(patterns, text), "jpeg_quality"

    if codec == "bpg":
        patterns = [
            r"(?:^|[_/\-])bpg[_\-]?qp[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])qp[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])bpg[_\-]?q[_\-]?([0-9]+(?:\.[0-9]+)?)",
        ]
        return extract_first_number(patterns, text), "bpg_qp"

    if codec == "bmshj":
        patterns = [
            r"(?:^|[_/\-])bmshj[_\-]?q[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])q([0-9]+(?:\.[0-9]+)?)",
        ]
        return extract_first_number(patterns, text), "compressai_quality"

    if codec == "cheng":
        patterns = [
            r"(?:^|[_/\-])cheng[_\-]?q[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])q([0-9]+(?:\.[0-9]+)?)",
        ]
        return extract_first_number(patterns, text), "compressai_quality"

    if codec == "jpeg2000":
        patterns = [
            r"(?:^|[_/\-])jpeg2000[_\-]?r(?:ate)?[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])jp2[_\-]?r(?:ate)?[_\-]?([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[_/\-])r([0-9]+(?:\.[0-9]+)?)",
        ]
        return extract_first_number(patterns, text), "jp2_rate"

    return None, "unknown_level"


def infer_raw_level(
    pred_path: Path,
    summary: Dict[str, Any],
    base_name: str,
    codec: str,
) -> Tuple[Optional[float], str, str]:
    """
    Prefer filename/folder compression value. Fall back to summary.json only if
    the filename does not contain a usable value.
    """
    v, name = infer_raw_level_from_filename(pred_path, base_name, codec)
    if v is not None:
        return v, name, "filename"

    if codec == "jpeg":
        v = get_first_numeric(summary, ["jpeg_quality", "quality_or_rate", "chosen_quality", "quality"])
        return v, "jpeg_quality", "summary"

    if codec == "bpg":
        v = get_first_numeric(summary, ["bpg_qp", "bpg_quality", "quality_or_rate", "chosen_quality", "qp", "quality"])
        return v, "bpg_qp", "summary"

    if codec in {"bmshj", "cheng"}:
        v = get_first_numeric(summary, ["compressai_quality", "quality_or_rate", "chosen_quality", "quality"])
        return v, "compressai_quality", "summary"

    if codec == "jpeg2000":
        v = get_first_numeric(summary, ["jp2_rate", "jp2_compression_x1000", "quality_or_rate", "chosen_quality", "rate"])
        return v, "jp2_rate", "summary"

    v = get_first_numeric(summary, ["quality_or_rate", "chosen_quality", "quality", "qp"])
    return v, "unknown_level", "summary"


def compression_direction(codec: str) -> str:
    """
    How raw filename value maps to compression strength.

    decreasing:
      smaller value = stronger compression
      examples: JPEG q10 is stronger than q50, BMSHJ q1 is stronger than q5

    increasing:
      larger value = stronger compression
      example: BPG qp44 is stronger than qp28
    """
    if codec in {"jpeg", "bmshj", "cheng"}:
        return "decreasing"
    if codec in {"bpg", "jpeg2000"}:
        return "increasing"
    return "increasing"


def normalized_severity_from_raw(codec: str, raw_level: Optional[float]) -> Tuple[Any, str]:
    """
    This is metadata only. Selection is done using raw filename compression values.
    Convention:
      0 = lowest compression severity
      1 = highest compression severity
    """
    if raw_level is None:
        return "", "missing_raw_level"

    raw = float(raw_level)

    if codec == "jpeg":
        q_min, q_max = 10.0, 60.0
        sev = 1.0 - (raw - q_min) / max(q_max - q_min, 1e-6)
        return float(np.clip(sev, 0.0, 1.0)), "filename_jpeg_quality"

    if codec == "bpg":
        qp_min, qp_max = 28.0, 48.0
        sev = (raw - qp_min) / max(qp_max - qp_min, 1e-6)
        return float(np.clip(sev, 0.0, 1.0)), "filename_bpg_qp"

    if codec == "bmshj":
        q_min, q_max = 1.0, 8.0
        sev = 1.0 - (raw - q_min) / max(q_max - q_min, 1e-6)
        return float(np.clip(sev, 0.0, 1.0)), "filename_bmshj_quality"

    if codec == "cheng":
        q_min, q_max = 1.0, 6.0
        sev = 1.0 - (raw - q_min) / max(q_max - q_min, 1e-6)
        return float(np.clip(sev, 0.0, 1.0)), "filename_cheng_quality"

    if codec == "jpeg2000":
        r_min, r_max = 1.0, 192.0
        sev = (raw - r_min) / max(r_max - r_min, 1e-6)
        return float(np.clip(sev, 0.0, 1.0)), "filename_jp2_rate"

    return "", "unknown"


def coco_stats_to_dict(stats: np.ndarray) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in zip(METRIC_NAMES, stats.tolist())
    }


def evaluate_cat_ids(
    coco_gt: COCO,
    coco_dt,
    cat_ids: List[int],
    img_ids: Optional[List[int]] = None,
    area_rng: Optional[List[List[float]]] = None,
    max_dets: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.catIds = list(cat_ids)

    if img_ids is not None:
        evaluator.params.imgIds = list(img_ids)

    if area_rng is not None:
        evaluator.params.areaRng = area_rng

    if max_dets is not None:
        evaluator.params.maxDets = max_dets

    if verbose:
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    else:
        with redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

    return coco_stats_to_dict(evaluator.stats)


def build_supercategory_groups(coco_gt: COCO) -> Dict[str, List[int]]:
    cats = coco_gt.loadCats(coco_gt.getCatIds())

    groups: Dict[str, List[int]] = {}
    for cat in cats:
        supercat = str(cat.get("supercategory", "unknown"))
        groups.setdefault(supercat, []).append(int(cat["id"]))

    return dict(sorted(groups.items(), key=lambda x: x[0]))


def build_category_groups(coco_gt: COCO) -> Dict[str, List[int]]:
    cats = coco_gt.loadCats(coco_gt.getCatIds())

    groups = {
        str(cat["name"]): [int(cat["id"])]
        for cat in cats
    }

    return dict(sorted(groups.items(), key=lambda x: x[0]))


def build_custom_groups(coco_gt: COCO) -> Dict[str, List[int]]:
    name_to_id = {
        cat["name"]: int(cat["id"])
        for cat in coco_gt.loadCats(coco_gt.getCatIds())
    }

    raw_groups = {
        "person": ["person"],
        "vehicles": [
            "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat",
        ],
        "animals": [
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe",
        ],
        "traffic_and_outdoor": [
            "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench",
        ],
        "sports": [
            "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket",
        ],
        "accessories": [
            "backpack", "umbrella", "handbag", "tie", "suitcase",
        ],
        "kitchen_and_food": [
            "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake",
        ],
        "furniture_and_indoor": [
            "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ],
        "electronics_and_appliances": [
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster",
        ],
    }

    groups: Dict[str, List[int]] = {}
    for group_name, cat_names in raw_groups.items():
        ids = [name_to_id[n] for n in cat_names if n in name_to_id]
        if ids:
            groups[group_name] = ids

    return groups


def find_prediction_files(inputs: List[Tuple[Optional[str], Path]], filename: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for provided_label, p in inputs:
        if p.is_file():
            out.append(
                {
                    "predictions_path": p,
                    "input_label": provided_label,
                    "input_root": str(p),
                }
            )
            continue

        if p.is_dir():
            for pred_path in sorted(p.rglob(filename)):
                out.append(
                    {
                        "predictions_path": pred_path,
                        "input_label": provided_label,
                        "input_root": str(p),
                    }
                )
            continue

        print(f"[WARN] path does not exist: {p}")

    seen = set()
    unique: List[Dict[str, Any]] = []

    for item in out:
        rp = str(Path(item["predictions_path"]).resolve())
        label = item.get("input_label") or ""
        key = f"{label}::{rp}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def build_prediction_metadata(item: Dict[str, Any], preedit_prefix: str) -> Dict[str, Any]:
    pred_path = Path(item["predictions_path"])
    summary = load_summary_near_predictions(pred_path)
    curve_type, base_name = infer_curve_type(pred_path, preedit_prefix=preedit_prefix)
    codec = infer_codec(pred_path, summary, base_name)

    raw_level, raw_level_name, raw_level_source = infer_raw_level(
        pred_path=pred_path,
        summary=summary,
        base_name=base_name,
        codec=codec,
    )

    severity_score, severity_source = normalized_severity_from_raw(codec, raw_level)
    detector = infer_detector_from_path(pred_path, item.get("input_label"))

    return {
        **item,
        "predictions_path": pred_path,
        "summary": summary,
        "detector": detector,
        "curve_type": curve_type,
        "base_name": base_name,
        "experiment_folder": pred_path.parent.name,
        "codec": codec,
        "codec_pretty": CODEC_PRETTY.get(codec, codec),
        "raw_level": raw_level,
        "raw_level_name": raw_level_name,
        "raw_level_source": raw_level_source,
        "severity_score": severity_score,
        "severity_source": severity_source,
        "dataset_bpp": safe_float(summary.get("dataset_bpp", "")),
        "avg_bpp_per_image": safe_float(summary.get("avg_bpp_per_image", "")),
        "avg_kb_per_image": safe_float(summary.get("avg_kb_per_image", "")),
        "summary_experiment": summary.get("experiment", ""),
        "summary_codec": summary.get("codec", ""),
        "summary_codec_family": summary.get("codec_family", ""),
        "summary_actual_codec_family": summary.get("actual_codec_family", ""),
    }


def choose_low_mid_high_from_filename_levels(codec: str, levels: List[float]) -> Dict[float, str]:
    """
    Choose lowest, middle, and highest compression from filename values.

    The middle value is chosen as the available non-extreme level closest to the
    numerical midpoint between min(levels) and max(levels). This is more stable
    than taking the median index when levels are unevenly spaced.
    """
    levels = sorted({quantize_float(v) for v in levels})

    if not levels:
        return {}

    if len(levels) == 1:
        return {levels[0]: "mid"}

    direction = compression_direction(codec)

    if direction == "decreasing":
        low_level = levels[-1]
        high_level = levels[0]
    else:
        low_level = levels[0]
        high_level = levels[-1]

    if len(levels) == 2:
        return {
            low_level: "low",
            high_level: "high",
        }

    raw_midpoint = (levels[0] + levels[-1]) / 2.0
    middle_candidates = [x for x in levels if x not in {low_level, high_level}]

    if middle_candidates:
        mid_level = min(
            middle_candidates,
            key=lambda x: (abs(x - raw_midpoint), x),
        )
    else:
        mid_level = levels[len(levels) // 2]

    return {
        low_level: "low",
        mid_level: "mid",
        high_level: "high",
    }


def select_low_mid_high_by_codec(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select representative compression values from filenames only.

    For each codec:
      1. collect filename compression values, such as q10, qp44, q3
      2. choose low, mid, and high compression settings
      3. keep all detectors and both baseline/SCOPE files at those values

    Expected output with 3 codecs, 3 detectors, and 2 curve types:
      3 codecs x 3 compression values x 3 detectors x 2 curve types = 54 files
    """
    selected: List[Dict[str, Any]] = []

    codecs = sorted({str(x.get("codec", "unknown")) for x in items})

    for codec in codecs:
        codec_items = [
            x for x in items
            if x.get("codec") == codec and x.get("raw_level") not in [None, ""]
        ]

        if not codec_items:
            print(f"[WARN] codec={codec}: no filename compression values found; skipping.")
            continue

        levels = sorted({quantize_float(float(x["raw_level"])) for x in codec_items})
        selected_levels = choose_low_mid_high_from_filename_levels(codec, levels)

        print(f"\nAvailable filename compression levels for codec={codec}: {levels}")
        print(f"Selected filename compression levels for codec={codec}:")
        for level, group_name in sorted(selected_levels.items(), key=lambda kv: kv[0]):
            print(f"  {group_name}: raw_level={level:g}")

        for x in codec_items:
            level = quantize_float(float(x["raw_level"]))
            if level not in selected_levels:
                continue

            y = dict(x)
            y["severity_group"] = selected_levels[level]
            y["selected_level"] = level
            y["selected_level_key"] = str(y.get("raw_level_name", "raw_level"))
            y["selected_level_source"] = str(y.get("raw_level_source", ""))
            y["selected_severity_score"] = (
                quantize_float(float(y["severity_score"]))
                if y.get("severity_score") not in [None, ""]
                else ""
            )
            selected.append(y)

    return selected


def select_all_severities(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for x in items:
        y = dict(x)
        if y.get("raw_level") not in [None, ""]:
            y["severity_group"] = f"{y.get('raw_level_name', 'level')}={float(y['raw_level']):g}"
            y["selected_level"] = quantize_float(float(y["raw_level"]))
            y["selected_level_key"] = str(y.get("raw_level_name", "raw_level"))
            y["selected_level_source"] = str(y.get("raw_level_source", ""))
        else:
            y["severity_group"] = "unknown"
            y["selected_level"] = ""
            y["selected_level_key"] = ""
            y["selected_level_source"] = ""

        y["selected_severity_score"] = (
            quantize_float(float(y["severity_score"]))
            if y.get("severity_score") not in [None, ""]
            else ""
        )
        out.append(y)

    return out


def filter_codecs(
    items: List[Dict[str, Any]],
    include_codecs: Optional[List[str]],
    exclude_codecs: Optional[List[str]],
) -> List[Dict[str, Any]]:
    include_set = {norm(x) for x in include_codecs} if include_codecs else None
    exclude_set = {norm(x) for x in exclude_codecs} if exclude_codecs else set()

    out = []
    for x in items:
        codec = norm(x.get("codec", "unknown"))

        if include_set is not None and codec not in include_set:
            continue

        if codec in exclude_set:
            continue

        out.append(x)

    return out


def build_groups(coco_gt: COCO, grouping: str) -> Dict[str, List[int]]:
    if grouping == "supercategory":
        return build_supercategory_groups(coco_gt)
    if grouping == "custom":
        return build_custom_groups(coco_gt)
    raise ValueError(f"Unknown grouping: {grouping}")


def evaluate_predictions_file(
    coco_gt: COCO,
    item: Dict[str, Any],
    grouping: str,
    include_overall: bool,
    include_per_category: bool,
    verbose_coco: bool,
) -> List[Dict[str, Any]]:
    pred_path = Path(item["predictions_path"])

    print(
        f"\n=== Evaluating {pred_path} | "
        f"detector={item.get('detector')} | "
        f"codec={item.get('codec')} | "
        f"group={item.get('severity_group')} | "
        f"level={item.get('selected_level')} | "
        f"curve={item.get('curve_type')} ==="
    )

    preds = read_json(pred_path)
    if not preds:
        print(f"[WARN] Empty predictions file: {pred_path}")
        return []

    coco_dt = coco_gt.loadRes(str(pred_path))

    common = {
        "predictions_path": str(pred_path),
        "input_root": item.get("input_root", ""),
        "input_label": item.get("input_label", ""),
        "detector": item.get("detector", ""),
        "experiment_folder": item.get("experiment_folder", ""),
        "curve_type": item.get("curve_type", ""),
        "base_name": item.get("base_name", ""),
        "codec": item.get("codec", ""),
        "codec_pretty": item.get("codec_pretty", ""),
        "severity_group": item.get("severity_group", ""),
        "selected_level": item.get("selected_level", ""),
        "selected_level_key": item.get("selected_level_key", ""),
        "selected_level_source": item.get("selected_level_source", ""),
        "severity_score": item.get("severity_score", ""),
        "selected_severity_score": item.get("selected_severity_score", ""),
        "severity_source": item.get("severity_source", ""),
        "raw_level": item.get("raw_level", ""),
        "raw_level_name": item.get("raw_level_name", ""),
        "raw_level_source": item.get("raw_level_source", ""),
        "dataset_bpp": item.get("dataset_bpp", ""),
        "avg_bpp_per_image": item.get("avg_bpp_per_image", ""),
        "avg_kb_per_image": item.get("avg_kb_per_image", ""),
        "summary_experiment": item.get("summary_experiment", ""),
        "summary_codec": item.get("summary_codec", ""),
        "summary_codec_family": item.get("summary_codec_family", ""),
        "summary_actual_codec_family": item.get("summary_actual_codec_family", ""),
    }

    rows: List[Dict[str, Any]] = []

    all_cat_ids = coco_gt.getCatIds()
    groups = build_groups(coco_gt, grouping)

    if include_overall:
        metrics = evaluate_cat_ids(
            coco_gt,
            coco_dt,
            all_cat_ids,
            verbose=verbose_coco,
        )

        row = {
            **common,
            "grouping": "overall",
            "group": "overall",
            "num_categories": len(all_cat_ids),
            "category_ids": ",".join(map(str, all_cat_ids)),
            "category_names": "all",
            **metrics,
        }
        rows.append(row)

    cats_by_id = {
        int(cat["id"]): str(cat["name"])
        for cat in coco_gt.loadCats(coco_gt.getCatIds())
    }

    for group_name, cat_ids in groups.items():
        cat_names = [cats_by_id[cid] for cid in cat_ids if cid in cats_by_id]

        metrics = evaluate_cat_ids(
            coco_gt,
            coco_dt,
            cat_ids,
            verbose=verbose_coco,
        )

        row = {
            **common,
            "grouping": grouping,
            "group": group_name,
            "num_categories": len(cat_ids),
            "category_ids": ",".join(map(str, cat_ids)),
            "category_names": ",".join(cat_names),
            **metrics,
        }
        rows.append(row)

    if include_per_category:
        category_groups = build_category_groups(coco_gt)

        for cat_name, cat_ids in category_groups.items():
            metrics = evaluate_cat_ids(
                coco_gt,
                coco_dt,
                cat_ids,
                verbose=verbose_coco,
            )

            row = {
                **common,
                "grouping": "category",
                "group": cat_name,
                "num_categories": 1,
                "category_ids": ",".join(map(str, cat_ids)),
                "category_names": cat_name,
                **metrics,
            }
            rows.append(row)

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] No rows to write: {path}")
        return

    fieldnames = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r.keys() if k not in fieldnames})
    fieldnames.extend(extra)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {path}")


def write_metadata_csv(path: Path, items: List[Dict[str, Any]]) -> None:
    rows = []
    for x in items:
        rows.append(
            {
                "predictions_path": str(x.get("predictions_path", "")),
                "input_root": x.get("input_root", ""),
                "input_label": x.get("input_label", ""),
                "detector": x.get("detector", ""),
                "curve_type": x.get("curve_type", ""),
                "base_name": x.get("base_name", ""),
                "experiment_folder": x.get("experiment_folder", ""),
                "codec": x.get("codec", ""),
                "codec_pretty": x.get("codec_pretty", ""),
                "severity_group": x.get("severity_group", ""),
                "selected_level": x.get("selected_level", ""),
                "selected_level_key": x.get("selected_level_key", ""),
                "selected_level_source": x.get("selected_level_source", ""),
                "severity_score": x.get("severity_score", ""),
                "selected_severity_score": x.get("selected_severity_score", ""),
                "severity_source": x.get("severity_source", ""),
                "raw_level": x.get("raw_level", ""),
                "raw_level_name": x.get("raw_level_name", ""),
                "raw_level_source": x.get("raw_level_source", ""),
                "dataset_bpp": x.get("dataset_bpp", ""),
                "avg_bpp_per_image": x.get("avg_bpp_per_image", ""),
                "avg_kb_per_image": x.get("avg_kb_per_image", ""),
                "summary_experiment": x.get("summary_experiment", ""),
            }
        )

    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations-json",
        required=True,
        help="COCO ground-truth annotations JSON.",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help=(
            "One or more predictions_coco.json files or folders. "
            "Supports detector_label=folder. "
            "If a folder is passed, the script searches recursively."
        ),
    )
    parser.add_argument(
        "--prediction-filename",
        default="predictions_coco.json",
        help="Filename to search for inside folders.",
    )
    parser.add_argument(
        "--out-csv",
        default="grouped_coco_map_low_mid_high.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional CSV with selected prediction files and inferred metadata.",
    )
    parser.add_argument(
        "--grouping",
        choices=["supercategory", "custom"],
        default="supercategory",
        help="Use official COCO supercategories or a coarser custom grouping.",
    )
    parser.add_argument(
        "--severity-selection",
        choices=["low_mid_high", "all"],
        default="low_mid_high",
        help="Select low/mid/high compression values per codec, or evaluate all compression values.",
    )
    parser.add_argument(
        "--include-codecs",
        nargs="*",
        default=None,
        help="Optional codecs to include, e.g. jpeg bpg bmshj.",
    )
    parser.add_argument(
        "--exclude-codecs",
        nargs="*",
        default=None,
        help="Optional codecs to exclude, e.g. cheng jpeg2000 unknown.",
    )
    parser.add_argument(
        "--preedit-prefix",
        default="preedit_",
        help="Prefix used to identify SCOPE/pre-edited folders.",
    )
    parser.add_argument(
        "--no-overall",
        action="store_true",
        help="Do not compute overall COCO mAP.",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Also compute AP for each individual COCO category.",
    )
    parser.add_argument(
        "--verbose-coco",
        action="store_true",
        help="Print full COCOeval output for every group.",
    )

    args = parser.parse_args()

    coco_gt = COCO(args.annotations_json)

    inputs = [parse_labeled_path(x) for x in args.predictions]
    pred_items = find_prediction_files(inputs, args.prediction_filename)

    if not pred_items:
        raise SystemExit("No prediction files found.")

    print(f"Found {len(pred_items)} prediction files before filtering.")

    metadata_items = [
        build_prediction_metadata(x, preedit_prefix=args.preedit_prefix)
        for x in pred_items
    ]

    metadata_items = filter_codecs(
        metadata_items,
        include_codecs=args.include_codecs,
        exclude_codecs=args.exclude_codecs,
    )

    print(f"Prediction files after codec filtering: {len(metadata_items)}")

    if args.severity_selection == "low_mid_high":
        selected_items = select_low_mid_high_by_codec(metadata_items)
    else:
        selected_items = select_all_severities(metadata_items)

    if not selected_items:
        raise SystemExit("No prediction files selected.")

    print(f"\nSelected prediction files: {len(selected_items)}")

    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else Path(args.out_csv).with_name(
        Path(args.out_csv).stem + "_selected_files.csv"
    )
    write_metadata_csv(metadata_csv, selected_items)

    all_rows: List[Dict[str, Any]] = []

    for item in selected_items:
        rows = evaluate_predictions_file(
            coco_gt=coco_gt,
            item=item,
            grouping=args.grouping,
            include_overall=not args.no_overall,
            include_per_category=args.per_category,
            verbose_coco=args.verbose_coco,
        )
        all_rows.extend(rows)

    write_csv(Path(args.out_csv), all_rows)

    print("\nDone.")
    print(f"Main results: {args.out_csv}")
    print(f"Selected files: {metadata_csv}")


if __name__ == "__main__":
    main()
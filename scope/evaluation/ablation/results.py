from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .common import (
    METRIC_KEYS, codec_to_family, canonical_method_key, normalize_family,
    norm, pretty_detector, pretty_method, read_json, to_float_or_blank,
)


def load_coco_image_size_map(
    annotations_json: Optional[str],
) -> Dict[int, Tuple[int, int]]:
    if not annotations_json:
        return {}

    p = Path(annotations_json)
    if not p.exists():
        print(f"[WARN] annotations JSON does not exist: {p}")
        return {}

    data = read_json(p)
    out: Dict[int, Tuple[int, int]] = {}

    for img in data.get("images", []):
        try:
            out[int(img["id"])] = (int(img["width"]), int(img["height"]))
        except Exception:
            pass

    return out


def read_first_bandwidth_row(summary_path: Path) -> Dict[str, Any]:
    p = summary_path.parent / "bandwidth.csv"
    if not p.exists():
        return {}

    with open(p, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)

    return {}


def bandwidth_stats_from_csv(
    summary_path: Path,
    image_size_map: Dict[int, Tuple[int, int]],
) -> Dict[str, Any]:
    p = summary_path.parent / "bandwidth.csv"
    empty = {
        "dataset_bpp": "",
        "avg_bpp_per_image": "",
        "avg_kb_per_image": "",
    }

    if not p.exists():
        return empty

    rows: List[Dict[str, str]] = []
    with open(p, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    if not rows:
        return empty

    total_bytes = 0.0
    total_pixels = 0.0
    bpps: List[float] = []
    kbs: List[float] = []

    for row in rows:
        estimated_bytes = to_float_or_blank(row.get("estimated_bytes", ""))
        if estimated_bytes == "":
            estimated_bytes = to_float_or_blank(row.get("bytes", ""))
        if estimated_bytes == "":
            estimated_bytes = to_float_or_blank(row.get("nbytes", ""))
        if estimated_bytes == "":
            estimated_bytes = to_float_or_blank(row.get("encoded_bytes", ""))

        estimated_kb = to_float_or_blank(row.get("estimated_kb", ""))
        if estimated_kb == "":
            estimated_kb = to_float_or_blank(row.get("kb", ""))
        if estimated_kb == "":
            estimated_kb = to_float_or_blank(row.get("encoded_kb", ""))

        if estimated_bytes == "" and estimated_kb != "":
            estimated_bytes = float(estimated_kb) * 1024.0
        if estimated_kb == "" and estimated_bytes != "":
            estimated_kb = float(estimated_bytes) / 1024.0

        if estimated_kb != "":
            kbs.append(float(estimated_kb))

        num_pixels = to_float_or_blank(row.get("num_pixels", ""))
        if num_pixels == "":
            try:
                image_id = int(float(row.get("image_id", "")))
                if image_id in image_size_map:
                    w, h = image_size_map[image_id]
                    num_pixels = float(w * h)
            except Exception:
                num_pixels = ""

        estimated_bpp = to_float_or_blank(row.get("estimated_bpp", ""))
        if estimated_bpp != "":
            bpps.append(float(estimated_bpp))
        elif estimated_bytes != "" and num_pixels != "" and float(num_pixels) > 0:
            bpps.append(float(estimated_bytes) * 8.0 / float(num_pixels))

        if estimated_bytes != "" and num_pixels != "" and float(num_pixels) > 0:
            total_bytes += float(estimated_bytes)
            total_pixels += float(num_pixels)

    return {
        "dataset_bpp": total_bytes * 8.0 / total_pixels if total_pixels > 0 else "",
        "avg_bpp_per_image": sum(bpps) / len(bpps) if bpps else "",
        "avg_kb_per_image": sum(kbs) / len(kbs) if kbs else "",
    }


def strip_scope_prefix(name: str, scope_prefix: str) -> Tuple[str, str]:
    for prefix in (part.strip() for part in scope_prefix.split(",")):
        if prefix and name.startswith(prefix):
            return "scope", name[len(prefix):]
    return "baseline", name


def infer_codec_from_base_name(
    base_name: str,
    summary: Dict[str, Any],
    summary_path: Path,
    bandwidth_meta: Dict[str, Any],
) -> str:
    text = norm(
        " ".join(
            str(x)
            for x in [
                base_name,
                summary.get("experiment", ""),
                summary.get("codec", ""),
                summary.get("codec_family", ""),
                summary.get("actual_codec_family", ""),
                summary.get("compressai_model", ""),
                summary.get("mode", ""),
                bandwidth_meta.get("codec", ""),
                bandwidth_meta.get("codec_family", ""),
                bandwidth_meta.get("compressai_model", ""),
                bandwidth_meta.get("mode", ""),
                bandwidth_meta.get("chosen_quality", ""),
                summary_path.parent.name,
            ]
        )
    )

    if "jpeg2000" in text or "jp2" in text or "j2k" in text or "wavelet" in text:
        return "jpeg2000"
    if "cheng" in text or "cheng2020" in text:
        return "cheng"
    if "bmshj" in text or "mbt" in text:
        return "bmshj"
    if "bpg" in text:
        return "bpg"
    if "jpeg" in text or "jpg" in text or "diff_jpeg" in text or "block_dct" in text:
        return "jpeg"

    return "unknown"


def infer_level(
    base_name: str,
    summary: Dict[str, Any],
    bandwidth_meta: Dict[str, Any],
) -> float:
    for src in [summary, bandwidth_meta]:
        for key in [
            "quality_or_rate",
            "jpeg_quality",
            "compressai_quality",
            "bpg_qp",
            "bpg_quality",
            "jp2_rate",
            "jp2_compression_x1000",
            "chosen_quality",
            "quality",
            "qp",
        ]:
            if key in src and src[key] not in [None, ""]:
                try:
                    return float(src[key])
                except Exception:
                    pass

    text = norm(base_name)

    patterns = [
        r"quality[_-]?([0-9]+(?:\.[0-9]+)?)",
        r"cheng[_-]?q([0-9]+(?:\.[0-9]+)?)",
        r"bmshj[_-]?q([0-9]+(?:\.[0-9]+)?)",
        r"qp[_-]?([0-9]+(?:\.[0-9]+)?)",
        r"_q([0-9]+(?:\.[0-9]+)?)",
        r"-q([0-9]+(?:\.[0-9]+)?)",
        r"_r([0-9]+(?:\.[0-9]+)?)",
        r"-r([0-9]+(?:\.[0-9]+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))

    return 0.0


def normalize_row(
    method_label: str,
    method_index: int,
    source_label: str,
    source_dir: Path,
    summary_path: Path,
    image_size_map: Dict[int, Tuple[int, int]],
    scope_prefix: str,
) -> Dict[str, Any]:
    summary = read_json(summary_path)

    exp_folder_name = summary_path.parent.name
    summary_exp_name = str(summary.get("experiment", exp_folder_name))

    curve_type, base_name = strip_scope_prefix(exp_folder_name, scope_prefix)
    if curve_type == "baseline":
        curve_type, base_name = strip_scope_prefix(summary_exp_name, scope_prefix)
        if curve_type == "baseline":
            base_name = exp_folder_name

    bandwidth_meta = read_first_bandwidth_row(summary_path)

    codec = infer_codec_from_base_name(
        base_name=base_name,
        summary=summary,
        summary_path=summary_path,
        bandwidth_meta=bandwidth_meta,
    )

    family = codec_to_family(codec)
    if family == "unknown":
        family = normalize_family(
            summary.get("actual_codec_family", summary.get("codec_family", "unknown"))
        )

    fallback = bandwidth_stats_from_csv(summary_path, image_size_map)

    avg_kb = to_float_or_blank(summary.get("avg_kb_per_image", ""))
    if avg_kb == "":
        avg_bytes = to_float_or_blank(summary.get("avg_bytes_per_image", ""))
        if avg_bytes != "":
            avg_kb = float(avg_bytes) / 1024.0
    if avg_kb == "":
        avg_kb = fallback["avg_kb_per_image"]

    avg_bpp = to_float_or_blank(summary.get("avg_bpp_per_image", ""))
    if avg_bpp == "":
        avg_bpp = fallback["avg_bpp_per_image"]

    dataset_bpp = to_float_or_blank(summary.get("dataset_bpp", ""))
    if dataset_bpp == "":
        dataset_bpp = fallback["dataset_bpp"]

    method_key = canonical_method_key(method_label)

    row: Dict[str, Any] = {
        "ablation_label": method_label,
        "ablation_key": method_key,
        "ablation_display": pretty_method(method_key),
        "ablation_index": method_index,
        "source_label": source_label,
        "detector": pretty_detector(source_label),
        "source_dir": str(source_dir),
        "experiment_folder": exp_folder_name,
        "experiment": summary_exp_name,
        "base_name": base_name,
        "summary_path": str(summary_path),
        "codec": codec,
        "family": family,
        "curve_type": curve_type,
        "level": infer_level(base_name, summary, bandwidth_meta),
        "dataset_bpp": dataset_bpp,
        "avg_bpp_per_image": avg_bpp,
        "avg_kb_per_image": avg_kb,
    }

    for k in METRIC_KEYS:
        row[k] = to_float_or_blank(summary.get(k, ""))

    return row


def collect_rows(
    labeled_dirs: Iterable[Tuple[str, str, Path]],
    image_size_map: Dict[int, Tuple[int, int]],
    scope_prefix: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()

    for method_index, (method_label, source_label, source_dir) in enumerate(labeled_dirs):
        source_dir = Path(source_dir)
        if not source_dir.exists():
            print(f"[WARN] missing folder: {source_dir}")
            continue

        for summary_path in sorted(source_dir.rglob("summary.json")):
            key = f"{method_label}::{source_label}::{summary_path.resolve()}"
            if key in seen:
                continue
            seen.add(key)

            try:
                rows.append(
                    normalize_row(
                        method_label=method_label,
                        method_index=method_index,
                        source_label=source_label,
                        source_dir=source_dir,
                        summary_path=summary_path,
                        image_size_map=image_size_map,
                        scope_prefix=scope_prefix,
                    )
                )
            except Exception as e:
                print(f"[WARN] could not parse {summary_path}: {e}")

    return rows


def write_dict_rows(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_collected_results(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "ablation_label",
        "ablation_key",
        "ablation_display",
        "ablation_index",
        "source_label",
        "detector",
        "source_dir",
        "experiment_folder",
        "experiment",
        "base_name",
        "codec",
        "family",
        "curve_type",
        "level",
        "dataset_bpp",
        "avg_bpp_per_image",
        "avg_kb_per_image",
        *METRIC_KEYS,
        "summary_path",
    ]
    write_dict_rows(rows, path, fieldnames)


def count_values(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda item: item[0]))

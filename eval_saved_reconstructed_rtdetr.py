#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Make the script work both from the repo root and from PreprocessingICM/.
_THIS_DIR = Path(__file__).resolve().parent
for _p in [
    _THIS_DIR,
    _THIS_DIR.parent,
    Path.cwd(),
    Path.cwd().parent,
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coco_eval import evaluate_coco_map
from dataset_coco import CocoDataset
from rtdetr_utils import RTDetrPredictor as YoloPredictor, remove_internal_xyxy
from utils import ensure_dir, save_json


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# Ultralytics COCO models output contiguous COCO80 class ids: 0..79.
# Official COCO annotations use category ids 1..90 with gaps.
YOLO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def _remap_predictions_for_eval(preds: List[Dict], category_map: str) -> List[Dict]:
    """
    category_map:
      - offset / none: keep predictor category ids as produced.
      - coco91: interpret predictor category ids as YOLO/Ultralytics COCO80 ids
                and remap them to official COCO category ids.
    """
    category_map = str(category_map or "offset").lower()

    if category_map in {"offset", "none"}:
        return preds

    if category_map != "coco91":
        raise ValueError(
            f"Unknown category_map={category_map!r}. Use 'offset', 'none', or 'coco91'."
        )

    out = []
    for p in preds:
        q = dict(p)
        cls_id = int(q["category_id"])

        if cls_id < 0 or cls_id >= len(YOLO80_TO_COCO91):
            raise ValueError(
                f"Invalid YOLO COCO80 class id {cls_id}; expected 0..79. "
                "This usually means category-id-offset was not set to 0 before remapping."
            )

        q["category_id"] = YOLO80_TO_COCO91[cls_id]
        out.append(q)

    return out


def predict_server(
    server_predictor: YoloPredictor,
    image_path: Path,
    image_id: int,
    category_id_offset: int,
    conf: float,
    iou: float,
    max_det: int,
    category_map: str,
) -> List[Dict]:
    category_map = str(category_map or "offset").lower()

    # Important:
    # For coco91 remapping, the predictor must output raw COCO80 ids: 0..79.
    # So we force the offset to 0 before applying YOLO80_TO_COCO91.
    pred_category_id_offset = 0 if category_map == "coco91" else category_id_offset

    preds = server_predictor.predict_full(
        image_path=image_path,
        image_id=image_id,
        category_id_offset=pred_category_id_offset,
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
    )

    preds = _remap_predictions_for_eval(preds, category_map=category_map)
    return remove_internal_xyxy(preds)


def discover_experiment_dirs(
    source_eval_dir: Path,
    experiments: Optional[List[str]] = None,
) -> List[Path]:
    if experiments:
        dirs = [source_eval_dir / name for name in experiments]
    else:
        dirs = sorted(
            p for p in source_eval_dir.iterdir()
            if p.is_dir() and (p / "reconstructed").is_dir()
        )

    missing = [p for p in dirs if not (p / "reconstructed").is_dir()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "These experiment folders do not contain a reconstructed/ directory:\n"
            f"{msg}"
        )

    if not dirs:
        raise RuntimeError(f"No experiment folders with reconstructed/ found in {source_eval_dir}")

    return dirs


def find_reconstructed_image(recon_dir: Path, original_file_name: str) -> Path:
    stem = Path(original_file_name).stem

    candidates = [recon_dir / f"{stem}.png"]
    candidates += [recon_dir / f"{stem}{ext}" for ext in IMAGE_EXTS if ext != ".png"]
    candidates += [recon_dir / original_file_name]

    for p in candidates:
        if p.exists():
            return p

    matches = []
    for ext in IMAGE_EXTS:
        matches.extend(recon_dir.glob(f"{stem}{ext}"))
        matches.extend(recon_dir.glob(f"{stem}{ext.upper()}"))

    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(
        f"Missing reconstructed image for {original_file_name}. "
        f"Tried stem={stem} in {recon_dir}"
    )


def _bytes_from_bandwidth_row(row: Dict[str, str]) -> int:
    for key in ("estimated_bytes", "bytes", "nbytes", "encoded_bytes"):
        if key in row and str(row[key]).strip() != "":
            return int(float(row[key]))

    for key in ("estimated_kb", "kb", "encoded_kb"):
        if key in row and str(row[key]).strip() != "":
            return int(round(float(row[key]) * 1024.0))

    raise ValueError(f"Could not find byte/KB field in bandwidth row: {row}")


def load_bandwidth_csv(exp_dir: Path) -> Tuple[Dict[int, Dict[str, str]], Dict[str, Dict[str, str]]]:
    csv_path = exp_dir / "bandwidth.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing bandwidth.csv in {exp_dir}")

    by_id: Dict[int, Dict[str, str]] = {}
    by_stem: Dict[str, Dict[str, str]] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("image_id", "") != "":
                try:
                    by_id[int(float(row["image_id"]))] = row
                except ValueError:
                    pass

            fn = row.get("file_name", "")
            if fn:
                by_stem[Path(fn).stem] = row

    return by_id, by_stem


def get_bandwidth_row_for_record(
    record,
    by_id: Dict[int, Dict[str, str]],
    by_stem: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    if int(record.image_id) in by_id:
        return by_id[int(record.image_id)]

    stem = Path(record.file_name).stem
    if stem in by_stem:
        return by_stem[stem]

    raise KeyError(
        f"No bandwidth row found for image_id={record.image_id}, file_name={record.file_name}"
    )


def run_single_saved_experiment(
    exp_dir: Path,
    dataset: CocoDataset,
    output_dir: Path,
    server_predictor: YoloPredictor,
    category_id_offset: int,
    conf: float,
    iou: float,
    max_det: int,
    category_map: str,
    limit: Optional[int] = None,
    skip_missing: bool = False,
) -> Dict:
    name = exp_dir.name
    recon_dir = exp_dir / "reconstructed"

    out_exp_dir = ensure_dir(output_dir / name)

    by_id, by_stem = load_bandwidth_csv(exp_dir)

    predictions: List[Dict] = []
    bandwidth_rows: List[Dict] = []
    missing_rows: List[Dict] = []

    records = dataset.images[:limit] if limit is not None else dataset.images

    for record in tqdm(records, desc=name):
        try:
            recon_path = find_reconstructed_image(recon_dir, record.file_name)
            old_bw_row = get_bandwidth_row_for_record(record, by_id, by_stem)
            estimated_bytes = _bytes_from_bandwidth_row(old_bw_row)
        except Exception as e:
            if not skip_missing:
                raise
            missing_rows.append(
                {
                    "image_id": record.image_id,
                    "file_name": record.file_name,
                    "reason": str(e),
                }
            )
            continue

        preds = predict_server(
            server_predictor=server_predictor,
            image_path=recon_path,
            image_id=record.image_id,
            category_id_offset=category_id_offset,
            conf=conf,
            iou=iou,
            max_det=max_det,
            category_map=category_map,
        )
        predictions.extend(preds)

        bandwidth_rows.append(
            {
                "image_id": record.image_id,
                "file_name": record.file_name,
                "reconstructed_path": str(recon_path),
                "estimated_bytes": int(estimated_bytes),
                "estimated_kb": int(estimated_bytes) / 1024.0,
                "chosen_quality": old_bw_row.get("chosen_quality", ""),
            }
        )

    pred_json = out_exp_dir / "predictions_coco.json"
    save_json(predictions, pred_json)

    bandwidth_csv = out_exp_dir / "bandwidth.csv"
    with open(bandwidth_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "reconstructed_path",
                "estimated_bytes",
                "estimated_kb",
                "chosen_quality",
            ],
        )
        writer.writeheader()
        writer.writerows(bandwidth_rows)

    if missing_rows:
        save_json(missing_rows, out_exp_dir / "missing_images.json")

    metrics = evaluate_coco_map(dataset.annotations_json, pred_json)

    avg_bytes = sum(r["estimated_bytes"] for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    avg_kb = avg_bytes / 1024.0

    summary = {
        "experiment": name,
        "source_experiment_dir": str(exp_dir),
        "num_images_evaluated": len(bandwidth_rows),
        "num_images_missing": len(missing_rows),
        "avg_bytes_per_image": avg_bytes,
        "avg_kb_per_image": avg_kb,
        "avg_rois_per_image": 0.0,
        "category_map": category_map,
        **metrics,
    }

    save_json(summary, out_exp_dir / "summary.json")
    return summary


def write_all_results(output_dir: Path, summaries: List[Dict]) -> Path:
    summary_csv = output_dir / "all_results.csv"

    fieldnames = [
        "experiment",
        "source_experiment_dir",
        "num_images_evaluated",
        "num_images_missing",
        "avg_bytes_per_image",
        "avg_kb_per_image",
        "avg_rois_per_image",
        "category_map",
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

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RT-DETR on already-saved reconstructed image folders. "
            "Each experiment folder must contain reconstructed/ and bandwidth.csv."
        )
    )

    parser.add_argument("--source-eval-dir", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--annotations-json", type=str, required=True)
    parser.add_argument("--detector-weights", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--category-id-offset", type=int, default=1)
    parser.add_argument(
        "--category-map",
        type=str,
        default="offset",
        choices=["offset", "none", "coco91"],
        help=(
            "Use 'coco91' when evaluating Ultralytics COCO80 predictions "
            "against standard COCO annotations with category ids 1..90 with gaps."
        ),
    )

    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help=(
            "Optional experiment folder names inside --source-eval-dir. "
            "If omitted, all folders with reconstructed/ are evaluated."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_eval_dir = Path(args.source_eval_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    dataset = CocoDataset(
        images_dir=args.images_dir,
        annotations_json=args.annotations_json,
    )

    server_predictor = YoloPredictor(
        args.detector_weights,
        device=args.device,
    )

    exp_dirs = discover_experiment_dirs(source_eval_dir, args.experiments)

    print(f"Found {len(exp_dirs)} experiment folders.")
    print(f"category_map={args.category_map}")
    if args.category_map == "coco91":
        print("Using YOLO/Ultralytics COCO80 -> official COCO91 category-id remapping.")
        print("category_id_offset will be forced to 0 internally before remapping.")
    else:
        print(f"Using category_id_offset={args.category_id_offset} without COCO91 remapping.")

    summaries: List[Dict] = []

    for exp_dir in exp_dirs:
        summary = run_single_saved_experiment(
            exp_dir=exp_dir,
            dataset=dataset,
            output_dir=output_dir,
            server_predictor=server_predictor,
            category_id_offset=int(args.category_id_offset),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            category_map=str(args.category_map),
            limit=args.limit,
            skip_missing=bool(args.skip_missing),
        )
        summaries.append(summary)

    summary_csv = write_all_results(output_dir, summaries)
    print(f"\nSaved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
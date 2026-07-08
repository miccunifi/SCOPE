#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


DETECTRON2_PRESETS = {
    "fasterrcnn_x101_fpn": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "faster_rcnn_x101_fpn": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "x101_fpn": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
}


COCO80_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_").replace("|", "_")


def load_coco_images_by_id(annotations_json: Path) -> Dict[int, Dict]:
    with open(annotations_json, "r") as f:
        coco = json.load(f)
    return {int(im["id"]): im for im in coco["images"]}


def find_reconstructed_image(recon_dir: Path, file_name: str) -> Path:
    stem = Path(file_name).stem

    candidates = [recon_dir / f"{stem}.png"]
    candidates += [recon_dir / f"{stem}{ext}" for ext in IMAGE_EXTS if ext != ".png"]
    candidates += [recon_dir / file_name]

    for p in candidates:
        if p.exists():
            return p

    matches = []
    for ext in IMAGE_EXTS:
        matches.extend(recon_dir.glob(f"{stem}{ext}"))
        matches.extend(recon_dir.glob(f"{stem}{ext.upper()}"))

    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(f"Could not find reconstructed image for {file_name} in {recon_dir}")


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


def get_bpp_for_image(image_id: int, file_name: str, by_id, by_stem) -> Optional[float]:
    row = by_id.get(int(image_id), None)
    if row is None:
        row = by_stem.get(Path(file_name).stem, None)

    if row is None:
        return None

    if row.get("estimated_bpp", "") != "":
        return float(row["estimated_bpp"])

    byte_keys = ["estimated_bytes", "compressed_bytes", "bytes", "nbytes", "encoded_bytes"]
    bytes_val = None
    for k in byte_keys:
        if row.get(k, "") != "":
            bytes_val = float(row[k])
            break

    width = float(row.get("width", 0) or 0)
    height = float(row.get("height", 0) or 0)
    num_pixels = float(row.get("num_pixels", 0) or 0)

    if num_pixels <= 0 and width > 0 and height > 0:
        num_pixels = width * height

    if bytes_val is None or num_pixels <= 0:
        return None

    return float(bytes_val * 8.0 / num_pixels)


def build_predictor(
    detector: str,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    detectron2_config: Optional[str],
    detectron2_weights: Optional[str],
):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    detector = detector.lower()
    config_name = detectron2_config or DETECTRON2_PRESETS.get(detector)
    if config_name is None:
        raise ValueError(
            f"Unknown detector={detector!r}. Use fasterrcnn_x101_fpn or pass --detectron2-config."
        )

    cfg = get_cfg()

    config_path = Path(config_name)
    if config_path.exists():
        cfg.merge_from_file(str(config_path))
    else:
        cfg.merge_from_file(model_zoo.get_config_file(config_name))

    if detectron2_weights:
        cfg.MODEL.WEIGHTS = detectron2_weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)

    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = iou
    cfg.TEST.DETECTIONS_PER_IMAGE = max_det

    return DefaultPredictor(cfg)


def parse_bgr_color(s: str) -> Tuple[int, int, int]:
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) != 3:
        raise ValueError("--box-color-bgr must have format B,G,R, for example 0,255,255")
    vals = [max(0, min(255, v)) for v in vals]
    return int(vals[0]), int(vals[1]), int(vals[2])


def draw_predictions_same_color(
    image_bgr,
    predictor,
    box_color: Tuple[int, int, int],
    box_thickness: int,
    font_scale: float,
    text_thickness: int,
    label_x_shift: int,
    show_labels: bool = True,
    show_scores: bool = True,
):
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return image_bgr.copy(), 0

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy().astype(int)

    drawn = image_bgr.copy()
    h, w = drawn.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.tolist()

        x1 = int(round(max(0, min(w - 1, x1))))
        y1 = int(round(max(0, min(h - 1, y1))))
        x2 = int(round(max(0, min(w - 1, x2))))
        y2 = int(round(max(0, min(h - 1, y2))))

        cv2.rectangle(
            drawn,
            (x1, y1),
            (x2, y2),
            box_color,
            thickness=box_thickness,
            lineType=cv2.LINE_AA,
        )

        label_parts = []

        if show_labels:
            if 0 <= int(cls_id) < len(COCO80_CLASS_NAMES):
                label_parts.append(COCO80_CLASS_NAMES[int(cls_id)])
            else:
                label_parts.append(f"class {int(cls_id)}")

        if show_scores:
            label_parts.append(f"{float(score):.2f}")

        if not label_parts:
            continue

        text = " ".join(label_parts)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = max(0, x1 - label_x_shift)
        text_y = y1 - 7

        if text_y - th - baseline < 0:
            text_y = y1 + th + baseline + 7

        bg_x1 = max(0, text_x)
        bg_y1 = max(0, text_y - th - baseline - 4)
        bg_x2 = min(w - 1, text_x + tw + 8)
        bg_y2 = min(h - 1, text_y + baseline + 4)

        cv2.rectangle(
            drawn,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            box_color,
            thickness=-1,
        )

        cv2.putText(
            drawn,
            text,
            (text_x + 4, text_y),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    return drawn, len(instances)


def add_bottom_padding_caption(
    image_bgr,
    text: str = "",
    pad_h: int = 70,
    font_scale: float = 0.85,
    thickness: int = 1,
):
    h, w = image_bgr.shape[:2]

    out = cv2.copyMakeBorder(
        image_bgr,
        top=0,
        bottom=pad_h,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    if text.strip() == "":
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = max(10, (w - tw) // 2)
    y = h + (pad_h + th) // 2 - 4

    cv2.putText(
        out,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return out


def resize_to_match(image_bgr, target_hw: Tuple[int, int]):
    target_h, target_w = target_hw
    h, w = image_bgr.shape[:2]
    if h == target_h and w == target_w:
        return image_bgr
    return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_horizontal_row(panels: List[np.ndarray], gap_px: int = 8) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to concatenate")

    target_h = panels[0].shape[0]
    resized = []

    for p in panels:
        h, w = p.shape[:2]
        if h != target_h:
            new_w = int(round(w * (target_h / h)))
            p = cv2.resize(p, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(p)

    if gap_px <= 0:
        return np.concatenate(resized, axis=1)

    gap = np.full((target_h, gap_px, 3), 255, dtype=np.uint8)

    row_parts = []
    for i, p in enumerate(resized):
        if i > 0:
            row_parts.append(gap)
        row_parts.append(p)

    return np.concatenate(row_parts, axis=1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Create one horizontal comparison row per selected image."
    )

    p.add_argument("--source-eval-dir", type=str, required=True)
    p.add_argument("--annotations-json", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--baseline-experiment", type=str, required=True)
    p.add_argument("--preedit-experiment", type=str, required=True)

    p.add_argument("--baseline-name", type=str, default="Baseline")
    p.add_argument("--preedit-name", type=str, default="SCOPE")

    p.add_argument("--image-ids", type=str, required=True)

    p.add_argument("--detector", type=str, default="fasterrcnn_x101_fpn")
    p.add_argument(
        "--detectron2-config",
        type=str,
        default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    )
    p.add_argument("--detectron2-weights", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=50)

    p.add_argument("--box-color-bgr", type=str, default="0,255,255")
    p.add_argument("--box-thickness", type=int, default=5)
    p.add_argument("--font-scale", type=float, default=0.70)
    p.add_argument("--text-thickness", type=int, default=2)
    p.add_argument("--label-x-shift", type=int, default=35)

    p.add_argument("--hide-class-labels", action="store_true")
    p.add_argument("--hide-scores", action="store_true")

    p.add_argument("--caption-pad-px", type=int, default=70)
    p.add_argument("--caption-font-scale", type=float, default=1.15)
    p.add_argument("--caption-text-thickness", type=int, default=1)

    p.add_argument("--panel-gap-px", type=int, default=8)
    p.add_argument("--png-compression", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()

    if args.conf < 0.0 or args.conf > 1.0:
        raise ValueError("--conf must be in the 0-1 range")

    if args.png_compression < 0 or args.png_compression > 9:
        raise ValueError("--png-compression must be between 0 and 9")

    if args.caption_pad_px < 0:
        raise ValueError("--caption-pad-px must be >= 0")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    source_eval_dir = Path(args.source_eval_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    baseline_dir = source_eval_dir / args.baseline_experiment
    preedit_dir = source_eval_dir / args.preedit_experiment

    baseline_recon_dir = baseline_dir / "reconstructed"
    preedit_recon_dir = preedit_dir / "reconstructed"

    if not baseline_recon_dir.is_dir():
        raise FileNotFoundError(f"Missing reconstructed/ directory: {baseline_recon_dir}")
    if not preedit_recon_dir.is_dir():
        raise FileNotFoundError(f"Missing reconstructed/ directory: {preedit_recon_dir}")

    baseline_by_id, baseline_by_stem = load_bandwidth_csv(baseline_dir)
    preedit_by_id, preedit_by_stem = load_bandwidth_csv(preedit_dir)

    images_by_id = load_coco_images_by_id(Path(args.annotations_json))
    selected_ids = [int(x.strip()) for x in args.image_ids.split(",") if x.strip()]

    box_color = parse_bgr_color(args.box_color_bgr)

    predictor = build_predictor(
        detector=args.detector,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        detectron2_config=args.detectron2_config,
        detectron2_weights=args.detectron2_weights or None,
    )

    summary_rows = []

    for image_id in tqdm(selected_ids, desc="comparison rows"):
        if image_id not in images_by_id:
            print(f"[WARN] image_id={image_id} not found in annotations. Skipping.")
            continue

        im = images_by_id[image_id]
        file_name = im["file_name"]
        stem = Path(file_name).stem

        baseline_path = find_reconstructed_image(baseline_recon_dir, file_name)
        preedit_path = find_reconstructed_image(preedit_recon_dir, file_name)

        baseline_bpp = get_bpp_for_image(
            image_id=image_id,
            file_name=file_name,
            by_id=baseline_by_id,
            by_stem=baseline_by_stem,
        )
        preedit_bpp = get_bpp_for_image(
            image_id=image_id,
            file_name=file_name,
            by_id=preedit_by_id,
            by_stem=preedit_by_stem,
        )

        baseline_img = cv2.imread(str(baseline_path), cv2.IMREAD_COLOR)
        preedit_img = cv2.imread(str(preedit_path), cv2.IMREAD_COLOR)

        if baseline_img is None:
            raise RuntimeError(f"Could not read baseline image: {baseline_path}")
        if preedit_img is None:
            raise RuntimeError(f"Could not read preedit image: {preedit_path}")

        target_hw = baseline_img.shape[:2]
        preedit_img = resize_to_match(preedit_img, target_hw)

        baseline_bpp_str = "NA" if baseline_bpp is None else f"{baseline_bpp:.6f}"
        preedit_bpp_str = "NA" if preedit_bpp is None else f"{preedit_bpp:.6f}"

        baseline_caption = (
            f"{args.baseline_name} | bpp: N/A"
            if baseline_bpp is None
            else f"{args.baseline_name} | bpp: {baseline_bpp:.4f}"
        )
        preedit_caption = (
            f"{args.preedit_name} | bpp: N/A"
            if preedit_bpp is None
            else f"{args.preedit_name} | bpp: {preedit_bpp:.4f}"
        )

        baseline_no_boxes = add_bottom_padding_caption(
            baseline_img,
            text=baseline_caption,
            pad_h=args.caption_pad_px,
            font_scale=args.caption_font_scale,
            thickness=args.caption_text_thickness,
        )

        baseline_boxes_img, baseline_n_pred = draw_predictions_same_color(
            image_bgr=baseline_img,
            predictor=predictor,
            box_color=box_color,
            box_thickness=args.box_thickness,
            font_scale=args.font_scale,
            text_thickness=args.text_thickness,
            label_x_shift=args.label_x_shift,
            show_labels=not args.hide_class_labels,
            show_scores=not args.hide_scores,
        )

        baseline_with_boxes = add_bottom_padding_caption(
            baseline_boxes_img,
            text="",
            pad_h=args.caption_pad_px,
            font_scale=args.caption_font_scale,
            thickness=args.caption_text_thickness,
        )

        preedit_no_boxes = add_bottom_padding_caption(
            preedit_img,
            text=preedit_caption,
            pad_h=args.caption_pad_px,
            font_scale=args.caption_font_scale,
            thickness=args.caption_text_thickness,
        )

        preedit_boxes_img, preedit_n_pred = draw_predictions_same_color(
            image_bgr=preedit_img,
            predictor=predictor,
            box_color=box_color,
            box_thickness=args.box_thickness,
            font_scale=args.font_scale,
            text_thickness=args.text_thickness,
            label_x_shift=args.label_x_shift,
            show_labels=not args.hide_class_labels,
            show_scores=not args.hide_scores,
        )

        preedit_with_boxes = add_bottom_padding_caption(
            preedit_boxes_img,
            text="",
            pad_h=args.caption_pad_px,
            font_scale=args.caption_font_scale,
            thickness=args.caption_text_thickness,
        )

        row = make_horizontal_row(
            [
                baseline_no_boxes,
                baseline_with_boxes,
                preedit_no_boxes,
                preedit_with_boxes,
            ],
            gap_px=args.panel_gap_px,
        )

        out_name = (
            f"{image_id:012d}_{stem}_comparison_row_"
            f"{safe_name(args.baseline_name)}_bpp_{baseline_bpp_str}_"
            f"vs_{safe_name(args.preedit_name)}_bpp_{preedit_bpp_str}.png"
        )
        out_name = safe_name(out_name)
        out_path = output_dir / out_name

        cv2.imwrite(
            str(out_path),
            row,
            [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression],
        )

        print(
            f"image_id={image_id} | "
            f"{args.baseline_name} bpp={baseline_bpp_str}, preds={baseline_n_pred} | "
            f"{args.preedit_name} bpp={preedit_bpp_str}, preds={preedit_n_pred} | "
            f"{out_path}"
        )

        summary_rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "baseline_experiment": args.baseline_experiment,
                "preedit_experiment": args.preedit_experiment,
                "baseline_bpp": baseline_bpp_str,
                "preedit_bpp": preedit_bpp_str,
                "baseline_num_predictions": baseline_n_pred,
                "preedit_num_predictions": preedit_n_pred,
                "baseline_reconstructed_path": str(baseline_path),
                "preedit_reconstructed_path": str(preedit_path),
                "output_path": str(out_path),
            }
        )

    summary_csv = output_dir / "comparison_rows_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "image_id",
            "file_name",
            "baseline_experiment",
            "preedit_experiment",
            "baseline_bpp",
            "preedit_bpp",
            "baseline_num_predictions",
            "preedit_num_predictions",
            "baseline_reconstructed_path",
            "preedit_reconstructed_path",
            "output_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved comparison-row summary to: {summary_csv}")


if __name__ == "__main__":
    main()
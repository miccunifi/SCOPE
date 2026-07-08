#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

# Make the script work both from repo root and from PreprocessingICM/.
_THIS_DIR = Path(__file__).resolve().parent
for _p in [_THIS_DIR, _THIS_DIR.parent, Path.cwd(), Path.cwd().parent]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coco_eval import evaluate_coco_map
from dataset_coco import CocoDataset
from utils import ensure_dir, save_json


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# COCO80 contiguous class ids -> official COCO category ids used by COCO JSON.
# Use this for detectors that output contiguous ids 0..79: YOLO, Detectron2, MMDet.
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
VALID_COCO91_IDS = set(COCO80_TO_COCO91)


class DetectorProtocol(Protocol):
    backend: str
    detector: str
    weights_name: str
    label_space: str

    def predict_full(self, image_path: str | Path, image_id: int) -> List[Dict[str, Any]]:
        ...


def xyxy_to_xywh(box: Iterable[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def remap_category_id(raw_label: int, label_space: str, category_id_offset: int) -> Optional[int]:
    """
    Convert detector output labels to official COCO category ids.

    label_space:
      auto    -> handled by the backend before calling this function.
      coco91  -> labels are already official COCO category ids 1..90 with holes.
                 Correct for TorchVision COCO detectors.
      coco80  -> labels are contiguous COCO80 ids 0..79; map to COCO91.
                 Correct for Detectron2 / MMDet / YOLO COCO detectors.
      offset  -> custom contiguous ids; category_id = raw_label + category_id_offset.
      none    -> keep raw label unchanged.
    """
    label_space = str(label_space).lower()
    raw_label = int(raw_label)

    if label_space == "coco91":
        return raw_label if raw_label in VALID_COCO91_IDS else None

    if label_space == "coco80":
        if raw_label < 0 or raw_label >= len(COCO80_TO_COCO91):
            return None
        return int(COCO80_TO_COCO91[raw_label])

    if label_space == "offset":
        return int(raw_label + category_id_offset)

    if label_space in {"none", "raw"}:
        return raw_label

    raise ValueError(f"Unknown label_space={label_space!r}")


# -----------------------------------------------------------------------------
# TorchVision detector wrapper
# -----------------------------------------------------------------------------

class TorchvisionCocoDetector:
    """
    TorchVision COCO detector wrapper.

    Important for official COCO evaluation:
    TorchVision pretrained detection models output labels aligned with official
    COCO category ids, i.e. COCO91 ids with holes. Therefore use label_space=coco91.

    This version does NOT silently choose v2 models for short aliases:
      fasterrcnn -> fasterrcnn_resnet50_fpn
      retinanet  -> retinanet_resnet50_fpn
      fcos       -> fcos_resnet50_fpn
    """

    backend = "torchvision"

    def __init__(
        self,
        detector: str,
        device: Optional[str] = None,
        weights: str = "DEFAULT",
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
        label_space: str = "coco91",
        category_id_offset: int = 1,
    ):
        self.detector = str(detector).lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_name = str(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.label_space = str(label_space).lower()
        self.category_id_offset = int(category_id_offset)

        self.model = self._build_model().to(self.device).eval()

    def _select_weights(self, enum_cls):
        if self.weights_name.lower() in {"none", "null", "false"}:
            return None
        if self.weights_name.upper() == "DEFAULT":
            return enum_cls.DEFAULT
        return enum_cls[self.weights_name]

    def _build_model(self):
        d = self.detector

        if d in {"fasterrcnn", "faster_rcnn", "frcnn", "fasterrcnn_resnet50_fpn"}:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )
            weights = self._select_weights(FasterRCNN_ResNet50_FPN_Weights)
            self.detector = "fasterrcnn_resnet50_fpn"
            return fasterrcnn_resnet50_fpn(
                weights=weights,
                box_score_thresh=self.conf,
                box_nms_thresh=self.iou,
                box_detections_per_img=self.max_det,
            )

        if d in {"fasterrcnn_resnet50_fpn_v2", "faster_rcnn_resnet50_fpn_v2"}:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
            weights = self._select_weights(FasterRCNN_ResNet50_FPN_V2_Weights)
            self.detector = "fasterrcnn_resnet50_fpn_v2"
            return fasterrcnn_resnet50_fpn_v2(
                weights=weights,
                box_score_thresh=self.conf,
                box_nms_thresh=self.iou,
                box_detections_per_img=self.max_det,
            )

        if d in {"retinanet", "retina", "retinanet_resnet50_fpn"}:
            from torchvision.models.detection import (
                RetinaNet_ResNet50_FPN_Weights,
                retinanet_resnet50_fpn,
            )
            weights = self._select_weights(RetinaNet_ResNet50_FPN_Weights)
            self.detector = "retinanet_resnet50_fpn"
            return retinanet_resnet50_fpn(
                weights=weights,
                score_thresh=self.conf,
                nms_thresh=self.iou,
                detections_per_img=self.max_det,
            )

        if d in {"retinanet_resnet50_fpn_v2", "retina_resnet50_fpn_v2"}:
            from torchvision.models.detection import (
                RetinaNet_ResNet50_FPN_V2_Weights,
                retinanet_resnet50_fpn_v2,
            )
            weights = self._select_weights(RetinaNet_ResNet50_FPN_V2_Weights)
            self.detector = "retinanet_resnet50_fpn_v2"
            return retinanet_resnet50_fpn_v2(
                weights=weights,
                score_thresh=self.conf,
                nms_thresh=self.iou,
                detections_per_img=self.max_det,
            )

        if d in {"fcos", "fcos_resnet50_fpn"}:
            from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
            weights = self._select_weights(FCOS_ResNet50_FPN_Weights)
            self.detector = "fcos_resnet50_fpn"
            return fcos_resnet50_fpn(
                weights=weights,
                score_thresh=self.conf,
                nms_thresh=self.iou,
                detections_per_img=self.max_det,
            )

        raise ValueError(
            f"Unknown TorchVision --detector {self.detector!r}. Use one of: "
            "fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, "
            "retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn."
        )

    @torch.no_grad()
    def predict_full(self, image_path: str | Path, image_id: int) -> List[Dict[str, Any]]:
        img = Image.open(image_path).convert("RGB")
        x = TF.to_tensor(img).to(self.device)

        out = self.model([x])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy().astype(int)

        order = scores.argsort()[::-1]
        preds: List[Dict[str, Any]] = []

        for idx in order:
            score = float(scores[idx])
            if score < self.conf:
                continue

            category_id = remap_category_id(
                raw_label=int(labels[idx]),
                label_space=self.label_space,
                category_id_offset=self.category_id_offset,
            )
            if category_id is None:
                continue

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in xyxy_to_xywh(boxes[idx].tolist())],
                    "score": score,
                }
            )

            if len(preds) >= self.max_det:
                break

        return preds


# -----------------------------------------------------------------------------
# Detectron2 detector wrapper, including Faster R-CNN ResNeXt-101-FPN
# -----------------------------------------------------------------------------

DETECTRON2_PRESETS = {
    # Faster R-CNN with ResNeXt-101-32x8d + FPN, COCO pretrained.
    "fasterrcnn_x101_fpn": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "faster_rcnn_x101_fpn": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "fasterrcnn_x_101_32x8d_fpn_3x": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "faster_rcnn_x_101_32x8d_fpn_3x": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "x101_fpn": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    # Convenience Detectron2 R50/R101 baselines.
    "fasterrcnn_r50_fpn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "fasterrcnn_r101_fpn": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
}


class Detectron2CocoDetector:
    """
    Detectron2 COCO detector wrapper.

    Detectron2 COCO models return contiguous dataset labels 0..79. For official
    COCO JSON evaluation, use label_space=coco80 so predictions are remapped to
    COCO91 ids with holes.
    """

    backend = "detectron2"

    def __init__(
        self,
        detector: str,
        device: Optional[str] = None,
        weights: str = "model_zoo",
        conf: float = 0.001,
        iou: float = 0.7,
        max_det: int = 300,
        label_space: str = "coco80",
        category_id_offset: int = 1,
        detectron2_config: Optional[str] = None,
        detectron2_weights: Optional[str] = None,
    ):
        self.detector = str(detector).lower()
        self.device_str = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_name = str(detectron2_weights or weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.label_space = str(label_space).lower()
        self.category_id_offset = int(category_id_offset)
        self.config_name = detectron2_config or DETECTRON2_PRESETS.get(self.detector, None)
        if self.config_name is None:
            raise ValueError(
                f"Unknown Detectron2 --detector {detector!r}. Either use one of "
                f"{sorted(DETECTRON2_PRESETS.keys())}, or pass --detectron2-config."
            )

        self.predictor = self._build_predictor()

    def _build_predictor(self):
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo
        except Exception as e:
            raise ImportError(
                "Detectron2 is required for --backend detectron2. Install a Detectron2 "
                "build matching your PyTorch/CUDA version. Original import error: "
                f"{repr(e)}"
            ) from e

        cfg = get_cfg()

        config_path = Path(str(self.config_name))
        if config_path.exists():
            cfg.merge_from_file(str(config_path))
            resolved_config = str(config_path)
        else:
            resolved_config = model_zoo.get_config_file(str(self.config_name))
            cfg.merge_from_file(resolved_config)

        w = self.weights_name
        if w.lower() in {"model_zoo", "default", "auto"}:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(str(self.config_name))
        elif w.lower() in {"none", "null", "false"}:
            cfg.MODEL.WEIGHTS = ""
        else:
            cfg.MODEL.WEIGHTS = w

        cfg.MODEL.DEVICE = self.device_str
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.iou
        cfg.TEST.DETECTIONS_PER_IMAGE = self.max_det

        # Preserve for summary/debug.
        self.resolved_config = resolved_config
        self.resolved_weights = cfg.MODEL.WEIGHTS
        self.detector = self.detector if self.detector not in DETECTRON2_PRESETS else self.detector

        return DefaultPredictor(cfg)

    @torch.no_grad()
    def predict_full(self, image_path: str | Path, image_id: int) -> List[Dict[str, Any]]:
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Could not read image for Detectron2: {image_path}")

        out = self.predictor(img_bgr)
        inst = out["instances"].to("cpu")
        if len(inst) == 0:
            return []

        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        labels = inst.pred_classes.numpy().astype(int)

        order = scores.argsort()[::-1]
        preds: List[Dict[str, Any]] = []
        for idx in order:
            score = float(scores[idx])
            if score < self.conf:
                continue

            category_id = remap_category_id(
                raw_label=int(labels[idx]),
                label_space=self.label_space,
                category_id_offset=self.category_id_offset,
            )
            if category_id is None:
                continue

            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in xyxy_to_xywh(boxes[idx].tolist())],
                    "score": score,
                }
            )
            if len(preds) >= self.max_det:
                break

        return preds


# -----------------------------------------------------------------------------
# Dataset/reconstructed-folder utilities
# -----------------------------------------------------------------------------

def discover_experiment_dirs(source_eval_dir: Path, experiments: Optional[List[str]] = None) -> List[Path]:
    if experiments:
        dirs = [source_eval_dir / name for name in experiments]
    else:
        dirs = sorted(p for p in source_eval_dir.iterdir() if p.is_dir() and (p / "reconstructed").is_dir())

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

    matches: List[Path] = []
    for ext in IMAGE_EXTS:
        matches.extend(recon_dir.glob(f"{stem}{ext}"))
        matches.extend(recon_dir.glob(f"{stem}{ext.upper()}"))

    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(
        f"Missing reconstructed image for {original_file_name}. Tried stem={stem} in {recon_dir}"
    )


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _bytes_from_bandwidth_row(row: Dict[str, str]) -> int:
    for key in ("estimated_bytes", "compressed_bytes", "bytes", "nbytes", "encoded_bytes"):
        val = _float_or_none(row.get(key, None))
        if val is not None:
            return int(round(val))

    for key in ("estimated_kb", "kb", "encoded_kb"):
        val = _float_or_none(row.get(key, None))
        if val is not None:
            return int(round(val * 1024.0))

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


def get_bandwidth_row_for_record(record, by_id: Dict[int, Dict[str, str]], by_stem: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if int(record.image_id) in by_id:
        return by_id[int(record.image_id)]

    stem = Path(record.file_name).stem
    if stem in by_stem:
        return by_stem[stem]

    raise KeyError(f"No bandwidth row found for image_id={record.image_id}, file_name={record.file_name}")


def get_num_pixels(record, fallback_image_path: Path, bandwidth_row: Optional[Dict[str, str]] = None) -> Tuple[int, int, int]:
    """Return width, height, pixels for bpp. Prefer stored metadata; otherwise read image header."""
    if bandwidth_row is not None:
        w = _float_or_none(bandwidth_row.get("width", None))
        h = _float_or_none(bandwidth_row.get("height", None))
        npix = _float_or_none(bandwidth_row.get("num_pixels", None))
        if w is not None and h is not None and npix is not None and npix > 0:
            return int(w), int(h), int(npix)

    for w_attr, h_attr in [("width", "height"), ("w", "h")]:
        if hasattr(record, w_attr) and hasattr(record, h_attr):
            w = int(getattr(record, w_attr))
            h = int(getattr(record, h_attr))
            if w > 0 and h > 0:
                return w, h, w * h

    with Image.open(fallback_image_path) as im:
        w, h = im.size
    return int(w), int(h), int(w) * int(h)


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def run_single_saved_experiment(
    exp_dir: Path,
    dataset: CocoDataset,
    output_dir: Path,
    detector: DetectorProtocol,
    limit: Optional[int] = None,
    skip_missing: bool = False,
) -> Dict[str, Any]:
    name = exp_dir.name
    recon_dir = exp_dir / "reconstructed"
    out_exp_dir = ensure_dir(output_dir / name)

    by_id, by_stem = load_bandwidth_csv(exp_dir)

    predictions: List[Dict[str, Any]] = []
    bandwidth_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    records = dataset.images[:limit] if limit is not None else dataset.images

    for record in tqdm(records, desc=f"{detector.backend}:{detector.detector}:{name}"):
        try:
            recon_path = find_reconstructed_image(recon_dir, record.file_name)
            old_bw_row = get_bandwidth_row_for_record(record, by_id, by_stem)
            compressed_bytes = _bytes_from_bandwidth_row(old_bw_row)
            width, height, num_pixels = get_num_pixels(record, record.path, old_bw_row)
        except Exception as e:
            if not skip_missing:
                raise
            missing_rows.append({"image_id": record.image_id, "file_name": record.file_name, "reason": str(e)})
            continue

        predictions.extend(detector.predict_full(recon_path, image_id=record.image_id))

        bpp = (float(compressed_bytes) * 8.0) / float(max(1, num_pixels))
        bandwidth_rows.append(
            {
                "image_id": int(record.image_id),
                "file_name": record.file_name,
                "reconstructed_path": str(recon_path),
                "width": int(width),
                "height": int(height),
                "num_pixels": int(num_pixels),
                "estimated_bytes": int(compressed_bytes),
                "estimated_bpp": bpp,
                "chosen_quality": old_bw_row.get("chosen_quality", ""),
            }
        )

    pred_json = out_exp_dir / "predictions_coco.json"
    save_json(predictions, pred_json)

    bandwidth_csv = out_exp_dir / "bandwidth.csv"
    with open(bandwidth_csv, "w", newline="") as f:
        fieldnames = [
            "image_id", "file_name", "reconstructed_path", "width", "height", "num_pixels",
            "estimated_bytes", "estimated_bpp", "chosen_quality",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bandwidth_rows)

    if missing_rows:
        save_json(missing_rows, out_exp_dir / "missing_images.json")

    metrics = evaluate_coco_map(dataset.annotations_json, pred_json)

    avg_bytes = sum(float(r["estimated_bytes"]) for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    avg_bpp_per_image = sum(float(r["estimated_bpp"]) for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    total_bytes = sum(float(r["estimated_bytes"]) for r in bandwidth_rows)
    total_pixels = sum(float(r["num_pixels"]) for r in bandwidth_rows)
    dataset_bpp = (total_bytes * 8.0) / float(max(1.0, total_pixels))

    summary = {
        "experiment": name,
        "source_experiment_dir": str(exp_dir),
        "detector_backend": detector.backend,
        "detector": detector.detector,
        "weights": detector.weights_name,
        "label_space": detector.label_space,
        "target_annotation_space": "coco91",
        "num_images_evaluated": len(bandwidth_rows),
        "num_images_missing": len(missing_rows),
        "avg_bytes_per_image": avg_bytes,
        "avg_bpp_per_image": avg_bpp_per_image,
        "dataset_bpp": dataset_bpp,
        "avg_rois_per_image": 0.0,
        **metrics,
    }

    if detector.backend == "detectron2":
        summary["detectron2_config"] = getattr(detector, "resolved_config", "")
        summary["detectron2_weights"] = getattr(detector, "resolved_weights", "")

    save_json(summary, out_exp_dir / "summary.json")
    return summary


def write_all_results(output_dir: Path, summaries: List[Dict[str, Any]]) -> Path:
    summary_csv = output_dir / "all_results.csv"
    metric_keys = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
    ]
    fieldnames = [
        "experiment", "source_experiment_dir", "detector_backend", "detector", "weights",
        "label_space", "target_annotation_space", "num_images_evaluated", "num_images_missing",
        "avg_bytes_per_image", "avg_bpp_per_image", "dataset_bpp", "avg_rois_per_image",
        "detectron2_config", "detectron2_weights",
        *metric_keys,
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return summary_csv


def resolve_label_space(backend: str, label_space: str) -> str:
    if label_space.lower() != "auto":
        return label_space.lower()
    if backend == "torchvision":
        return "coco91"
    if backend == "detectron2":
        return "coco80"
    raise ValueError(f"Cannot infer label_space for backend={backend}")


def build_detector_from_args(args: argparse.Namespace) -> DetectorProtocol:
    backend = str(args.backend).lower()
    label_space = resolve_label_space(backend, args.label_space)

    if backend == "torchvision":
        return TorchvisionCocoDetector(
            detector=args.detector,
            device=args.device,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            label_space=label_space,
            category_id_offset=args.category_id_offset,
        )

    if backend == "detectron2":
        return Detectron2CocoDetector(
            detector=args.detector,
            device=args.device,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            label_space=label_space,
            category_id_offset=args.category_id_offset,
            detectron2_config=args.detectron2_config,
            detectron2_weights=args.detectron2_weights,
        )

    raise ValueError(f"Unknown backend={backend!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate COCO detectors on already-saved reconstructed images. "
            "The GT annotations are official COCO category ids (COCO91 / 1..90 with holes). "
            "Supports TorchVision detectors and Detectron2 Faster R-CNN X101-FPN."
        )
    )
    parser.add_argument("--source-eval-dir", type=str, required=True,
                        help="Directory containing experiment subfolders with reconstructed/ and bandwidth.csv.")
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--annotations-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--backend", type=str, default="torchvision", choices=["torchvision", "detectron2"])
    parser.add_argument("--detector", type=str, required=True,
                        help=(
                            "TorchVision: fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, "
                            "retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, fcos_resnet50_fpn. "
                            "Detectron2: fasterrcnn_x101_fpn, fasterrcnn_r50_fpn, fasterrcnn_r101_fpn, "
                            "or any name if --detectron2-config is passed."
                        ))
    parser.add_argument("--weights", type=str, default="DEFAULT",
                        help=(
                            "TorchVision weights enum member, usually DEFAULT. For Detectron2, "
                            "use model_zoo/default/auto or a checkpoint path/URL."
                        ))

    parser.add_argument("--detectron2-config", type=str, default=None,
                        help=(
                            "Detectron2 config path or model-zoo config name. If omitted and "
                            "--detector=fasterrcnn_x101_fpn, uses "
                            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml."
                        ))
    parser.add_argument("--detectron2-weights", type=str, default=None,
                        help="Optional Detectron2 checkpoint path/URL. Overrides --weights for Detectron2.")

    parser.add_argument("--label-space", type=str, default="auto",
                        choices=["auto", "coco91", "coco80", "offset", "none", "raw"],
                        help=(
                            "Detector output label space. auto => torchvision:coco91, detectron2:coco80. "
                            "For official COCO annotations this is usually what you want."
                        ))
    parser.add_argument("--category-id-offset", type=int, default=1,
                        help="Only used when --label-space offset.")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Optional experiment folder names inside --source-eval-dir. If omitted, all folders with reconstructed/ are evaluated.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug only. COCO AP on a limited subset is not comparable to full val AP.")
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit is not None:
        print("[WARN] --limit is for debugging only. Full COCO AP needs the full evaluation set.")

    source_eval_dir = Path(args.source_eval_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    dataset = CocoDataset(images_dir=args.images_dir, annotations_json=args.annotations_json)
    detector = build_detector_from_args(args)

    print(
        f"Using detector backend={detector.backend}, detector={detector.detector}, "
        f"label_space={detector.label_space}, weights={detector.weights_name}"
    )

    if detector.backend == "torchvision" and detector.label_space != "coco91":
        print("[WARN] TorchVision pretrained COCO detectors usually need --label-space coco91.")
    if detector.backend == "detectron2" and detector.label_space != "coco80":
        print("[WARN] Detectron2 COCO detectors usually need --label-space coco80.")

    exp_dirs = discover_experiment_dirs(source_eval_dir, args.experiments)

    summaries: List[Dict[str, Any]] = []
    for exp_dir in exp_dirs:
        summaries.append(
            run_single_saved_experiment(
                exp_dir=exp_dir,
                dataset=dataset,
                output_dir=output_dir,
                detector=detector,
                limit=args.limit,
                skip_missing=args.skip_missing,
            )
        )

    summary_csv = write_all_results(output_dir, summaries)
    print(f"\nSaved summary to: {summary_csv}")


if __name__ == "__main__":
    main()

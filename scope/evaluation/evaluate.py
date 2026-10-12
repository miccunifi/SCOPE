from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from .categories import YOLO80_TO_COCO91
from .coco_metrics import evaluate_coco_map
from .conditioning import infer_codec_family_from_experiment, get_experiment_severity
from .dataset import CocoDataset
from .transmission import compressai_transmission, uniform_jpeg_transmission
from .utils import ensure_dir, jpeg_encode_decode, read_image_bgr, save_json, write_image
from .yolo_utils import YoloPredictor, remove_internal_xyxy
from .codec_transmissions import (
    real_bpg_transmission,
    real_jpeg2000_transmission,
    scope_bpg_transmission,
    scope_compressai_transmission,
    scope_jpeg2000_transmission,
    scope_jpeg_transmission,
    _chosen_quality,
)


def _remap_predictions_for_eval(preds: List[Dict], category_map: str) -> List[Dict]:
    category_map = str(category_map or "offset").lower()
    if category_map in {"offset", "none"}:
        return preds

    if category_map != "coco91":
        raise ValueError(f"Unknown category_map={category_map!r}. Use 'offset' or 'coco91'.")

    out = []
    for p in preds:
        q = dict(p)
        cls_id = int(q["category_id"])
        if cls_id < 0 or cls_id >= len(YOLO80_TO_COCO91):
            raise ValueError(f"Invalid YOLO COCO80 class id {cls_id}; expected 0..79.")
        q["category_id"] = YOLO80_TO_COCO91[cls_id]
        out.append(q)

    return out


def predict_server(
    server_predictor: YoloPredictor,
    image_path: Path,
    image_id: int,
    category_id_offset: int,
    server_predict_cfg: Dict,
) -> List[Dict]:
    category_map = str(server_predict_cfg.get("category_map", "offset")).lower()
    pred_category_id_offset = 0 if category_map == "coco91" else category_id_offset

    preds = server_predictor.predict_full(
        image_path=image_path,
        image_id=image_id,
        category_id_offset=pred_category_id_offset,
        conf=float(server_predict_cfg.get("conf", 0.001)),
        iou=float(server_predict_cfg.get("iou", 0.7)),
        max_det=int(server_predict_cfg.get("max_det", 300)),
    )

    preds = _remap_predictions_for_eval(preds, category_map=category_map)
    return remove_internal_xyxy(preds)


def run_single_experiment(
    exp_cfg: Dict,
    dataset: CocoDataset,
    output_dir: Path,
    server_predictor: YoloPredictor,
    category_id_offset: int,
    server_predict_cfg: Dict,
    device,
) -> Dict:
    name = exp_cfg["name"]
    mode = exp_cfg["mode"]

    exp_dir = ensure_dir(output_dir / name)
    recon_dir = ensure_dir(exp_dir / "reconstructed")
    debug_dir = ensure_dir(exp_dir / "attention_debug")

    save_debug = bool(exp_cfg.get("save_debug", False))
    max_debug_images = int(exp_cfg.get("max_debug_images", 25))
    saved_debug_count = 0

    predictions = []
    bandwidth_rows = []

    for record in tqdm(dataset.images, desc=name):
        img = read_image_bgr(record.path)
        debug = None

        if mode in {"uniform", "jpeg"}:
            result_obj = uniform_jpeg_transmission(
                img_bgr=img,
                quality=int(exp_cfg["jpeg_quality"]),
            )
            reconstructed = result_obj.reconstructed_bgr
            estimated_bytes = int(result_obj.estimated_bytes)

        elif mode in {"jpeg2000", "jp2"}:
            result = real_jpeg2000_transmission(
                img_bgr=img,
                jp2_rate=exp_cfg.get("jp2_rate", None),
                encoder_bin=str(exp_cfg.get("jp2_encoder_bin", "opj_compress")),
                decoder_bin=str(exp_cfg.get("jp2_decoder_bin", "opj_decompress")),
                encoder_args=exp_cfg.get("jp2_encoder_args", []),
                backend=str(exp_cfg.get("jp2_backend", "auto")),
                cv2_compression_x1000=exp_cfg.get("jp2_compression_x1000", None),
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])

        elif mode == "bpg":
            result = real_bpg_transmission(
                img_bgr=img,
                bpg_qp=int(exp_cfg.get("bpg_qp", exp_cfg.get("bpg_quality", 40))),
                encoder_bin=str(exp_cfg.get("bpg_encoder_bin", "bpgenc")),
                decoder_bin=str(exp_cfg.get("bpg_decoder_bin", "bpgdec")),
                encoder_args=exp_cfg.get("bpg_encoder_args", []),
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])

        elif mode == "compressai":
            result_obj = compressai_transmission(
                img_bgr=img,
                model_name=str(exp_cfg["compressai_model"]),
                quality=int(exp_cfg["compressai_quality"]),
                device=device,
            )
            reconstructed = result_obj.reconstructed_bgr
            estimated_bytes = int(result_obj.estimated_bytes)

        elif mode == "scope_jpeg":
            result = scope_jpeg_transmission(
                img,
                exp_cfg,
                int(exp_cfg["jpeg_quality"]),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        elif mode == "scope_bpg":
            result = scope_bpg_transmission(
                img,
                exp_cfg,
                int(exp_cfg.get("bpg_qp", exp_cfg.get("bpg_quality", 40))),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        elif mode in {"scope_jpeg2000", "scope_jp2"}:
            result = scope_jpeg2000_transmission(
                img,
                exp_cfg,
                exp_cfg.get("jp2_rate", None),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        elif mode == "scope_compressai":
            result = scope_compressai_transmission(
                img,
                exp_cfg,
                str(exp_cfg["compressai_model"]),
                int(exp_cfg["compressai_quality"]),
                device=device,
                return_debug=save_debug and saved_debug_count < max_debug_images,
            )
            reconstructed = result["reconstructed_bgr"]
            estimated_bytes = int(result["estimated_bytes"])
            debug = result["debug"]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        recon_path = recon_dir / (Path(record.file_name).stem + ".png")
        write_image(recon_path, reconstructed)

        if save_debug and debug is not None and saved_debug_count < max_debug_images:
            stem = Path(record.file_name).stem

            write_image(debug_dir / f"{stem}_edited.png", debug["edited_bgr"])

            attn = debug["attention"][0, 0].numpy()
            cv2.imwrite(
                str(debug_dir / f"{stem}_attention.png"),
                np.round(np.clip(attn, 0, 1) * 255).astype(np.uint8),
            )

            delta = debug["effective_delta"][0].abs().mean(dim=0).numpy()
            delta = delta / max(float(delta.max()), 1e-8)
            cv2.imwrite(
                str(debug_dir / f"{stem}_delta.png"),
                np.round(delta * 255).astype(np.uint8),
            )

            with open(debug_dir / f"{stem}_meta.json", "w") as f:
                json.dump(
                    {
                        "codec_family": debug.get("codec_family"),
                        "severity": debug.get("severity"),
                    },
                    f,
                    indent=2,
                )

            saved_debug_count += 1

        preds = predict_server(
            server_predictor,
            recon_path,
            record.image_id,
            category_id_offset,
            server_predict_cfg,
        )
        predictions.extend(preds)

        h, w = img.shape[:2]
        num_pixels = max(1, int(h) * int(w))
        estimated_bpp = (float(estimated_bytes) * 8.0) / float(num_pixels)

        bandwidth_rows.append(
            {
                "image_id": record.image_id,
                "file_name": record.file_name,
                "width": int(w),
                "height": int(h),
                "num_pixels": num_pixels,
                "estimated_bytes": estimated_bytes,
                "estimated_bpp": estimated_bpp,
                "num_rois": 0,
                "chosen_quality": _chosen_quality(exp_cfg),
                "codec_family": infer_codec_family_from_experiment(exp_cfg),
                "severity": get_experiment_severity(exp_cfg),
            }
        )

    pred_json = exp_dir / "predictions_coco.json"
    save_json(predictions, pred_json)

    bandwidth_csv = exp_dir / "bandwidth.csv"
    with open(bandwidth_csv, "w", newline="") as f:
        fieldnames = [
            "image_id",
            "file_name",
            "width",
            "height",
            "num_pixels",
            "estimated_bytes",
            "estimated_bpp",
            "num_rois",
            "chosen_quality",
            "codec_family",
            "severity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bandwidth_rows)

    metrics = evaluate_coco_map(dataset.annotations_json, pred_json)

    avg_bytes = sum(r["estimated_bytes"] for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    avg_bpp_per_image = sum(r["estimated_bpp"] for r in bandwidth_rows) / max(1, len(bandwidth_rows))
    total_bytes = sum(r["estimated_bytes"] for r in bandwidth_rows)
    total_pixels = sum(r["num_pixels"] for r in bandwidth_rows)
    dataset_bpp = (float(total_bytes) * 8.0) / float(max(1, total_pixels))

    summary = {
        "experiment": name,
        "mode": mode,
        "codec_family": infer_codec_family_from_experiment(exp_cfg),
        "severity": get_experiment_severity(exp_cfg),
        "avg_bytes_per_image": avg_bytes,
        "avg_bpp_per_image": avg_bpp_per_image,
        "dataset_bpp": dataset_bpp,
        "avg_rois_per_image": 0.0,
        **metrics,
    }

    save_json(summary, exp_dir / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = CocoDataset(
        images_dir=cfg["dataset"]["images_dir"],
        annotations_json=cfg["dataset"]["annotations_json"],
    )

    output_dir = ensure_dir(cfg["output_dir"])
    device = cfg.get("device", None)

    server_predictor = YoloPredictor(cfg["models"]["server_model"], device=device)

    summaries = []

    root_passthrough_keys = [
        "bpg_encoder_bin",
        "bpg_decoder_bin",
        "bpg_encoder_args",
        "jp2_encoder_bin",
        "jp2_decoder_bin",
        "jp2_encoder_args",

        "scope_ckpt",
        "architecture",
        "condition_mode",
        "base_ch",
        "family_emb_dim",
        "cond_dim",
        "max_delta_obj",
        "max_delta_bg",
        "attention_bias",
        "bg_mode",
        "bg_blur_kernel",
        "bg_lowres_scale",
    ]

    for exp_cfg in cfg["experiments"]:
        exp_cfg = dict(exp_cfg)

        for k in root_passthrough_keys:
            if k in cfg and k not in exp_cfg:
                exp_cfg[k] = cfg[k]

        summaries.append(
            run_single_experiment(
                exp_cfg,
                dataset,
                output_dir,
                server_predictor,
                int(cfg.get("category_id_offset", 1)),
                cfg.get("server_predict", {}),
                device,
            )
        )

    summary_csv = output_dir / "all_results.csv"
    fieldnames = [
        "experiment",
        "mode",
        "codec_family",
        "severity",
        "avg_bytes_per_image",
        "avg_bpp_per_image",
        "dataset_bpp",
        "avg_rois_per_image",
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

    print(f"\nSaved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
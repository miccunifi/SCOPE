#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import importlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset_coco import CocoDataset
from utils import ensure_dir, read_image_bgr, write_image


# ---------------------------------------------------------------------
# Make training file importable when this script is run from repo root.
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
for _p in [
    _THIS_DIR,
    _THIS_DIR / "standalone_conditional_preeditor",
    Path.cwd(),
    Path.cwd() / "standalone_conditional_preeditor",
]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------
# Import old FiLM / blur_replace model class.
# ---------------------------------------------------------------------
_IMPORT_ERRORS = []
ConditionalAttentionPreEditor = None

for _module_name in [
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg_sevbasis",
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg",
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8",
    "train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_fix_coco_d2weights",
    "train_preedit_attention_conditional_yolo_hooks_objectmask_boxattention_bgreplace_objstrong_checked",
    "train_preedit_attention_conditional_yolo_hooks_objectmask_boxattention_bgreplace",
]:
    try:
        _m = importlib.import_module(_module_name)
        ConditionalAttentionPreEditor = _m.ConditionalAttentionPreEditor
        print(f"Using ConditionalAttentionPreEditor from: {_module_name}")
        break
    except Exception as _e:
        _IMPORT_ERRORS.append((_module_name, repr(_e)))

if ConditionalAttentionPreEditor is None:
    msg = "Could not import ConditionalAttentionPreEditor.\n"
    msg += "Import errors:\n"
    for _name, _err in _IMPORT_ERRORS:
        msg += f"  {_name}: {_err}\n"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------
def now() -> float:
    return time.perf_counter()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def to_float_or_blank(x: Any) -> Any:
    if x in [None, ""]:
        return ""
    try:
        return float(x)
    except Exception:
        return ""


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def _ckpt_arg(ckpt: Dict, key: str, default):
    args = ckpt.get("args", {}) or {}
    if not isinstance(args, dict):
        args = vars(args)

    if key in ckpt:
        return ckpt[key]
    if key in args:
        return args[key]
    return default


# ---------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------
def normalize_codec_family_name(name: str) -> str:
    aliases = {
        "diffjpeg": "diff_jpeg",
        "diff_jpeg": "diff_jpeg",
        "jpeg": "diff_jpeg",
        "block_dct": "diff_jpeg",

        "bpg": "bpg_like",
        "bpg_like": "bpg_like",
        "bpglike": "bpg_like",
        "intra_smooth": "bpg_like",

        "bmshj": "bmshj",
        "learned": "bmshj",
        "compressai": "bmshj",
        "cheng": "bmshj",

        # Allowed as actual codec alias, but not valid if checkpoint did not train it.
        "jp2_like": "jp2_like",
        "jp2": "jp2_like",
        "jpeg2000": "jp2_like",
        "wavelet": "jp2_like",
    }

    name = str(name).strip()
    if name not in aliases:
        raise ValueError(
            f"Unknown codec_family={name!r}. "
            "Use diff_jpeg, bpg_like, bmshj, or aliases."
        )

    return aliases[name]


def infer_codec_family_from_experiment(exp_cfg: Dict) -> str:
    explicit = exp_cfg.get("codec_family")
    if explicit is not None:
        return normalize_codec_family_name(str(explicit))

    mode = str(exp_cfg.get("mode", ""))
    codec = str(exp_cfg.get("compressai_model", "")).lower()

    if mode in {"uniform", "jpeg", "attention_preedit_jpeg"}:
        return "diff_jpeg"
    if mode in {"bpg", "attention_preedit_bpg"}:
        return "bpg_like"
    if mode in {"jpeg2000", "jp2", "attention_preedit_jpeg2000", "attention_preedit_jp2"}:
        return "jp2_like"

    if "cheng" in codec:
        return "bmshj"
    if "bmshj" in codec:
        return "bmshj"

    return "bmshj"


def infer_severity_from_experiment(exp_cfg: Dict) -> float:
    mode = str(exp_cfg.get("mode", ""))
    codec = str(exp_cfg.get("compressai_model", "")).lower()

    if mode in {"uniform", "jpeg", "attention_preedit_jpeg"}:
        q = float(exp_cfg.get("jpeg_quality", 10))
        return float(np.clip(1.0 - (q - 10.0) / max(60.0 - 10.0, 1e-6), 0.0, 1.0))

    if mode in {"jpeg2000", "jp2", "attention_preedit_jpeg2000", "attention_preedit_jp2"}:
        r = float(exp_cfg.get("jp2_rate", 32))
        r_min = float(exp_cfg.get("jp2_rate_min", 1))
        r_max = float(exp_cfg.get("jp2_rate_max", 192))
        return float(np.clip((r - r_min) / max(r_max - r_min, 1e-6), 0.0, 1.0))

    if mode in {"bpg", "attention_preedit_bpg"}:
        q = float(exp_cfg.get("bpg_qp", exp_cfg.get("bpg_quality", 40)))
        q_min = float(exp_cfg.get("bpg_qp_min", 28))
        q_max = float(exp_cfg.get("bpg_qp_max", 48))
        return float(np.clip((q - q_min) / max(q_max - q_min, 1e-6), 0.0, 1.0))

    if "bmshj" in codec:
        q = float(exp_cfg.get("compressai_quality", 1))
        return float(np.clip(1.0 - (q - 1.0) / 7.0, 0.0, 1.0))

    if "cheng" in codec or "mbt" in codec:
        q = float(exp_cfg.get("compressai_quality", 1))
        return float(np.clip(1.0 - (q - 1.0) / 5.0, 0.0, 1.0))

    return 0.5


def get_experiment_severity(exp_cfg: Dict) -> float:
    cond_cfg = exp_cfg.get("condition", {}) or {}

    if "severity" in cond_cfg:
        return float(np.clip(float(cond_cfg["severity"]), 0.0, 1.0))

    if "severity" in exp_cfg:
        return float(np.clip(float(exp_cfg["severity"]), 0.0, 1.0))

    return infer_severity_from_experiment(exp_cfg)


def severity_group(severity: float, bin_size: Optional[float]) -> str:
    if bin_size is None or bin_size <= 0:
        return f"{severity:.6f}"

    b = round(float(severity) / float(bin_size)) * float(bin_size)
    b = float(np.clip(b, 0.0, 1.0))
    return f"{b:.6f}"


def is_preedit_mode(mode: str) -> bool:
    return str(mode) in {
        "attention_preedit_jpeg",
        "attention_preedit_bpg",
        "attention_preedit_jpeg2000",
        "attention_preedit_jp2",
        "attention_preedit_compressai",
    }


# ---------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------
def bgr_to_model_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )


def tensor_to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    rgb = np.round(x * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
_PREEDIT_CACHE: Dict[Tuple[str, str], Tuple[torch.nn.Module, Dict[str, int]]] = {}


def load_conditional_preedit_model(
    ckpt_path: str,
    device: torch.device,
    exp_cfg: Dict,
):
    key = (str(Path(ckpt_path).resolve()), str(device))
    if key in _PREEDIT_CACHE:
        return _PREEDIT_CACHE[key]

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"Checkpoint {ckpt_path} must be the conditional training checkpoint with key 'model'."
        )

    state = _strip_module_prefix(ckpt["model"])

    codec_family_to_id = ckpt.get("codec_family_to_id")
    if codec_family_to_id is None:
        raise ValueError(f"Checkpoint {ckpt_path} has no codec_family_to_id.")

    codec_family_to_id = {str(k): int(v) for k, v in codec_family_to_id.items()}

    bg_mode = str(exp_cfg.get("bg_mode", _ckpt_arg(ckpt, "bg_mode", "blur_replace")))
    bg_blur_kernel = int(exp_cfg.get("bg_blur_kernel", _ckpt_arg(ckpt, "bg_blur_kernel", 51)))
    bg_lowres_scale = int(exp_cfg.get("bg_lowres_scale", _ckpt_arg(ckpt, "bg_lowres_scale", 16)))

    model = ConditionalAttentionPreEditor(
        num_codec_families=len(codec_family_to_id),
        base_ch=int(exp_cfg.get("base_ch", _ckpt_arg(ckpt, "base_ch", 64))),
        family_emb_dim=int(exp_cfg.get("family_emb_dim", _ckpt_arg(ckpt, "family_emb_dim", 12))),
        cond_dim=int(exp_cfg.get("cond_dim", _ckpt_arg(ckpt, "cond_dim", 96))),
        max_delta_obj=float(exp_cfg.get("max_delta_obj", _ckpt_arg(ckpt, "max_delta_obj", 0.09))),
        max_delta_bg=float(exp_cfg.get("max_delta_bg", _ckpt_arg(ckpt, "max_delta_bg", 0.03))),
        attention_bias=float(exp_cfg.get("attention_bias", _ckpt_arg(ckpt, "attention_bias", -4.0))),
        bg_mode=bg_mode,
        bg_blur_kernel=bg_blur_kernel,
        bg_lowres_scale=bg_lowres_scale,
    )

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    print(f"Loaded conditional pre-editor: {ckpt_path}")
    print(f"  codec_family_to_id: {codec_family_to_id}")
    print(f"  model class: {ConditionalAttentionPreEditor.__module__}.{ConditionalAttentionPreEditor.__name__}")
    print(f"  base_ch: {int(exp_cfg.get('base_ch', _ckpt_arg(ckpt, 'base_ch', 64)))}")
    print(f"  family_emb_dim: {int(exp_cfg.get('family_emb_dim', _ckpt_arg(ckpt, 'family_emb_dim', 12)))}")
    print(f"  cond_dim: {int(exp_cfg.get('cond_dim', _ckpt_arg(ckpt, 'cond_dim', 96)))}")
    print(f"  bg_mode: {bg_mode}")

    _PREEDIT_CACHE[key] = (model, codec_family_to_id)
    return model, codec_family_to_id


# ---------------------------------------------------------------------
# Timed pre-edit only
# ---------------------------------------------------------------------
@torch.no_grad()
def apply_preedit_timed(
    img_bgr: np.ndarray,
    exp_cfg: Dict,
    device,
    *,
    save_output: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, codec_family_to_id = load_conditional_preedit_model(
        str(exp_cfg["preedit_ckpt"]),
        torch_device,
        exp_cfg,
    )

    family_name = infer_codec_family_from_experiment(exp_cfg)
    if family_name not in codec_family_to_id:
        raise ValueError(
            f"Family {family_name!r} not in checkpoint mapping {codec_family_to_id}. "
            f"For unseen JP2 ablation, condition with diff_jpeg, bmshj, or bpg_like."
        )

    severity = get_experiment_severity(exp_cfg)

    t0 = now()

    x = bgr_to_model_tensor(img_bgr, torch_device)
    family_id = torch.tensor(
        [codec_family_to_id[family_name]],
        device=torch_device,
        dtype=torch.long,
    )
    severity_t = torch.tensor([severity], device=torch_device, dtype=x.dtype)

    sync_if_cuda(torch_device)
    t1 = now()

    edited, delta_obj, delta_bg, attention, effective_delta = model(
        x,
        severity=severity_t,
        codec_family=family_id,
    )

    sync_if_cuda(torch_device)
    t2 = now()

    edited_bgr = None
    if save_output:
        edited_bgr = tensor_to_bgr_uint8(edited)
        sync_if_cuda(torch_device)

    t3 = now()

    timings = {
        "condition_codec_family": family_name,
        "condition_severity": severity,
        "preedit_tensor_ms": (t1 - t0) * 1000.0,
        "preedit_forward_ms": (t2 - t1) * 1000.0,
        "preedit_to_bgr_ms": (t3 - t2) * 1000.0 if save_output else 0.0,
        "preedit_total_ms": (t3 - t0) * 1000.0 if save_output else (t2 - t0) * 1000.0,
    }

    return edited_bgr, timings


# ---------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------
def merge_experiment_config(cfg: Dict[str, Any], exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp_cfg = dict(exp_cfg)

    common = cfg.get("preedit_common", {}) or {}
    for k, v in common.items():
        if k not in exp_cfg:
            exp_cfg[k] = v

    root_passthrough_keys = [
        "preedit_ckpt",
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

    for k in root_passthrough_keys:
        if k in cfg and k not in exp_cfg:
            exp_cfg[k] = cfg[k]

    return exp_cfg


# ---------------------------------------------------------------------
# Summary utilities
# ---------------------------------------------------------------------
def values_from_rows(rows: List[Dict[str, Any]], key: str) -> List[float]:
    vals = []
    for r in rows:
        v = to_float_or_blank(r.get(key, ""))
        if v != "":
            vals.append(float(v))
    return vals


def summarize_group(rows: List[Dict[str, Any]], group_info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(group_info)
    out["num_measurements"] = len(rows)

    timing_keys = [
        "read_ms",
        "preedit_tensor_ms",
        "preedit_forward_ms",
        "preedit_to_bgr_ms",
        "preedit_total_ms",
        "image_megapixels",
    ]

    for key in timing_keys:
        vals = values_from_rows(rows, key)
        if not vals:
            continue

        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_p95"] = percentile(vals, 95)
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))

    fwd_ms = out.get("preedit_forward_ms_mean", None)
    total_ms = out.get("preedit_total_ms_mean", None)
    mpix = out.get("image_megapixels_mean", None)

    if fwd_ms is not None and fwd_ms > 0:
        out["preedit_forward_fps"] = float(1000.0 / fwd_ms)
        if mpix is not None:
            out["preedit_forward_megapixels_per_second"] = float(mpix / (fwd_ms / 1000.0))

    if total_ms is not None and total_ms > 0:
        out["preedit_total_fps"] = float(1000.0 / total_ms)
        if mpix is not None:
            out["preedit_total_megapixels_per_second"] = float(mpix / (total_ms / 1000.0))

    return out


def group_and_summarize(
    rows: List[Dict[str, Any]],
    group_keys: List[str],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(r)

    summaries = []
    for key, group_rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        info = {k: v for k, v in zip(group_keys, key)}
        summaries.append(summarize_group(group_rows, info))

    return summaries


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r.keys() if k not in fieldnames})
    fieldnames.extend(extra)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------
def benchmark_experiment(
    exp_cfg: Dict[str, Any],
    dataset: CocoDataset,
    output_dir: Path,
    device,
    *,
    limit: Optional[int],
    warmup: int,
    repeat: int,
    save_outputs: bool,
    max_saved_outputs: int,
    severity_bin_size: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    exp_name = str(exp_cfg["name"])
    mode = str(exp_cfg["mode"])

    if not is_preedit_mode(mode):
        print(f"[SKIP] {exp_name}: mode={mode!r} is not a preedit mode.")
        return [], {
            "experiment": exp_name,
            "skipped": True,
            "reason": f"mode={mode!r} is not a preedit mode",
            "num_measurements": 0,
        }

    exp_dir = ensure_dir(output_dir / exp_name)
    edited_dir = ensure_dir(exp_dir / "edited") if save_outputs else None

    records = list(dataset.images)
    if limit is not None and limit > 0:
        records = records[:limit]

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, codec_family_to_id = load_conditional_preedit_model(
        str(exp_cfg["preedit_ckpt"]),
        torch_device,
        exp_cfg,
    )

    condition_family = infer_codec_family_from_experiment(exp_cfg)
    condition_severity = get_experiment_severity(exp_cfg)
    severity_bin = severity_group(condition_severity, severity_bin_size)

    if condition_family not in codec_family_to_id:
        raise ValueError(
            f"{exp_name}: condition family {condition_family!r} is not in checkpoint mapping "
            f"{codec_family_to_id}. Use diff_jpeg, bmshj, or bpg_like."
        )

    warmup_records = records[: min(warmup, len(records))]
    if warmup_records:
        print(f"[{exp_name}] warmup images: {len(warmup_records)}")

    for record in warmup_records:
        img = read_image_bgr(record.path)
        _edited, _timing = apply_preedit_timed(
            img,
            exp_cfg,
            device,
            save_output=False,
        )

    sync_if_cuda(torch_device)

    print(
        f"[{exp_name}] family={condition_family}, "
        f"severity={condition_severity:.6f}, "
        f"timing images={len(records)}, repeat={repeat}"
    )

    per_image_rows: List[Dict[str, Any]] = []
    saved_count = 0

    for record in tqdm(records, desc=f"timing {exp_name}"):
        t_read0 = now()
        img = read_image_bgr(record.path)
        t_read1 = now()

        read_ms = (t_read1 - t_read0) * 1000.0
        h, w = img.shape[:2]
        num_pixels = max(1, int(h) * int(w))
        image_megapixels = float(num_pixels) / 1_000_000.0

        for rep in range(repeat):
            should_save = bool(
                save_outputs
                and rep == 0
                and saved_count < max_saved_outputs
            )

            edited_bgr, preedit_timing = apply_preedit_timed(
                img,
                exp_cfg,
                device,
                save_output=should_save,
            )

            if should_save and edited_bgr is not None and edited_dir is not None:
                out_path = edited_dir / (Path(record.file_name).stem + ".png")
                write_image(out_path, edited_bgr)
                saved_count += 1

            row: Dict[str, Any] = {
                "experiment": exp_name,
                "mode": mode,
                "image_id": record.image_id,
                "file_name": record.file_name,
                "repeat": rep,
                "width": int(w),
                "height": int(h),
                "num_pixels": int(num_pixels),
                "image_megapixels": image_megapixels,
                "read_ms": read_ms,
                "condition_codec_family": condition_family,
                "condition_severity": condition_severity,
                "condition_severity_bin": severity_bin,
                **preedit_timing,
            }

            per_image_rows.append(row)

    exp_summary = summarize_group(
        per_image_rows,
        {
            "experiment": exp_name,
            "mode": mode,
            "condition_codec_family": condition_family,
            "condition_severity": condition_severity,
            "condition_severity_bin": severity_bin,
        },
    )

    write_csv(exp_dir / "preedit_timing_per_image.csv", per_image_rows)
    with open(exp_dir / "preedit_timing_summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    return per_image_rows, exp_summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only first N validation images. Default: all images.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup images before timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Repeated timing passes per image.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional list of experiment names to benchmark.",
    )
    parser.add_argument(
        "--severity-bin-size",
        type=float,
        default=None,
        help="Optional severity bin size. Example: 0.2 groups 0.162/0.246/etc. into nearest bins.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Optionally save a few edited images. Disabled by default because it adds CPU copy/write time.",
    )
    parser.add_argument(
        "--max-saved-outputs",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = CocoDataset(
        images_dir=cfg["dataset"]["images_dir"],
        annotations_json=cfg["dataset"]["annotations_json"],
    )

    device = cfg.get("device", None)

    if args.out_dir is not None:
        output_dir = ensure_dir(Path(args.out_dir))
    else:
        output_dir = ensure_dir(
            Path(cfg.get("output_dir", "outputs_preedit_timing")) / "preedit_timing_by_family_severity"
        )

    wanted = set(args.experiments) if args.experiments else None

    all_rows: List[Dict[str, Any]] = []
    experiment_summaries: List[Dict[str, Any]] = []

    for raw_exp_cfg in cfg["experiments"]:
        exp_cfg = merge_experiment_config(cfg, raw_exp_cfg)
        exp_name = str(exp_cfg["name"])

        if wanted is not None and exp_name not in wanted:
            continue

        rows, summary = benchmark_experiment(
            exp_cfg=exp_cfg,
            dataset=dataset,
            output_dir=output_dir,
            device=device,
            limit=args.limit,
            warmup=args.warmup,
            repeat=args.repeat,
            save_outputs=args.save_outputs,
            max_saved_outputs=args.max_saved_outputs,
            severity_bin_size=args.severity_bin_size,
        )

        if rows:
            all_rows.extend(rows)
        experiment_summaries.append(summary)

    # Main outputs
    write_csv(output_dir / "preedit_timing_all_per_image.csv", all_rows)
    write_csv(output_dir / "preedit_timing_by_experiment.csv", experiment_summaries)

    # Aggregated analyses
    by_family = group_and_summarize(
        all_rows,
        ["condition_codec_family"],
    )
    by_severity = group_and_summarize(
        all_rows,
        ["condition_severity_bin"],
    )
    by_family_severity = group_and_summarize(
        all_rows,
        ["condition_codec_family", "condition_severity_bin"],
    )

    write_csv(output_dir / "preedit_timing_by_family.csv", by_family)
    write_csv(output_dir / "preedit_timing_by_severity.csv", by_severity)
    write_csv(output_dir / "preedit_timing_by_family_severity.csv", by_family_severity)

    print("\nPreedit timing by family:")
    for s in by_family:
        fam = s.get("condition_codec_family", "")
        fwd = s.get("preedit_forward_ms_mean", float("nan"))
        total = s.get("preedit_total_ms_mean", float("nan"))
        fps = s.get("preedit_forward_fps", float("nan"))
        print(
            f"  {fam}: "
            f"forward={fwd:.3f} ms/img, "
            f"total={total:.3f} ms/img, "
            f"forward_fps={fps:.2f}"
        )

    print("\nPreedit timing by family + severity:")
    for s in by_family_severity:
        fam = s.get("condition_codec_family", "")
        sev = s.get("condition_severity_bin", "")
        fwd = s.get("preedit_forward_ms_mean", float("nan"))
        total = s.get("preedit_total_ms_mean", float("nan"))
        print(
            f"  {fam}, severity={sev}: "
            f"forward={fwd:.3f} ms/img, "
            f"total={total:.3f} ms/img"
        )

    print(f"\nWrote: {output_dir / 'preedit_timing_all_per_image.csv'}")
    print(f"Wrote: {output_dir / 'preedit_timing_by_experiment.csv'}")
    print(f"Wrote: {output_dir / 'preedit_timing_by_family.csv'}")
    print(f"Wrote: {output_dir / 'preedit_timing_by_severity.csv'}")
    print(f"Wrote: {output_dir / 'preedit_timing_by_family_severity.csv'}")


if __name__ == "__main__":
    main()
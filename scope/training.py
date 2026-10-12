from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CocoBoxCropDataset
from .teachers import FrozenDetectron2FPNConsistency, FrozenYOLOFeatureConsistency
from .codecs import RealCodecProxyManager, codec_proxy_batch
from .model import SCOPE
from .losses import attention_loss, background_simplified_target, sobel_edges, soften_mask, tv_loss, weighted_l1, weighted_mean
from .conditioning import (
    build_codec_family_mapping,
    normalize_severity_tensor,
    parse_csv_floats,
    parse_csv_strings,
    parse_family_float_lists,
    parse_family_prob_lists,
    parse_sampling_probs,
    sample_conditions,
)


def make_dataloader(
    images_dir: str,
    annotations_json: str,
    crop_size: int,
    object_crop_prob: float,
    crop_margin: float,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = CocoBoxCropDataset(
        images_dir=images_dir,
        annotations_json=annotations_json,
        crop_size=crop_size,
        object_crop_prob=object_crop_prob,
        crop_margin=crop_margin,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )


def next_batch(loader: DataLoader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def format_live_metrics(
    accum: Dict[str, float],
    steps_done: int,
    lr: float,
    device: torch.device,
) -> str:
    keys = [
        "loss",
        "obj_l1",
        "obj_edge",
        "bg_l1",
        "bg_flat",
        "rate",
        "attn_mean",
        "attn_obj",
        "attn_bg",
        "delta",
        "sev",
    ]

    parts = []
    denom = max(1, steps_done)
    for k in keys:
        if k in accum:
            parts.append(f"{k}={accum[k] / denom:.5f}")

    parts.append(f"lr={lr:.2e}")

    if device.type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        parts.append(f"gpu_mem={mem_gb:.2f}GB")

    return ", ".join(parts)


def compute_losses(
    *,
    args,
    model: nn.Module,
    codec_proxy_manager: Optional[nn.Module],
    detector_loss_module: Optional[nn.Module],
    images: torch.Tensor,
    masks: torch.Tensor,
    family_ids: torch.Tensor,
    severity_norm: torch.Tensor,
    id_to_codec_family: Dict[int, str],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Annotation-box masks
    # masks is the hard box mask produced from GT annotation boxes.
    #
    # We intentionally use two different masks:
    #
    #   attn_target:
    #       tight target for the model attention map.
    #       This teaches attention to activate on annotated object boxes,
    #       not broadly on background.
    #
    #   det_mask:
    #       wider object/context mask for YOLO feature consistency.
    #       Detector features need some context around objects, so this can be
    #       larger than the attention target.
    #
    #   bg_mask:
    #       background region used for background smoothness/rate losses.
    #       It excludes the wider detector/context region.
    mask_box = masks.float().clamp(0.0, 1.0)
    attn_target = soften_mask(mask_box, blur_kernel=args.attn_blur)
    det_mask = soften_mask(mask_box, blur_kernel=args.det_mask_blur)
    bg_mask = (1.0 - det_mask).clamp(0.0, 1.0)

    edited, delta_obj, delta_bg, attention, effective_delta = model(
        images,
        severity=severity_norm,
        codec_family=family_ids,
    )

    if codec_proxy_manager is not None:
        # Original compression is diagnostic only; no gradient needed.
        with torch.no_grad():
            compressed_original, _ = codec_proxy_manager(
                images,
                family_ids=family_ids,
                severity_norm=severity_norm,
                id_to_codec_family=id_to_codec_family,
            )

        # Edited compression must keep gradients to the SCOPE.
        compressed_edited, codec_bpp = codec_proxy_manager(
            edited,
            family_ids=family_ids,
            severity_norm=severity_norm,
            id_to_codec_family=id_to_codec_family,
        )
    else:
        with torch.no_grad():
            compressed_original = codec_proxy_batch(
                images,
                family_ids=family_ids,
                severity_norm=severity_norm,
                id_to_codec_family=id_to_codec_family,
                proxy_args=args,
            )

        compressed_edited = codec_proxy_batch(
            edited,
            family_ids=family_ids,
            severity_norm=severity_norm,
            id_to_codec_family=id_to_codec_family,
            proxy_args=args,
        )
        codec_bpp = images.new_tensor(0.0)

    # Object preservation after compression.
    obj_l1 = weighted_l1(compressed_edited, images, attn_target)

    # Edge preservation on objects, useful for detection-like features.
    edge_clean = sobel_edges(images).detach()
    edge_comp = sobel_edges(compressed_edited)
    obj_edge = weighted_l1(edge_comp, edge_clean, attn_target)

    # Diagnostic only if lambda_bg_l1=0.
    # Use this only if you want visual similarity of the background to clean.
    bg_l1 = weighted_l1(edited, images, bg_mask)

    # Optional low-frequency/flat background target.
    # This is the key non-conservative background objective: it does NOT force
    # the background to match the clean image; it pushes it toward a cheaper
    # representation such as blur/lowres/mean.
    if args.lambda_bg_flat > 0:
        bg_target = background_simplified_target(
            images,
            mode=args.bg_flat_mode,
            blur_kernel=args.bg_blur_kernel,
            lowres_scale=args.bg_lowres_scale,
        ).detach()
        bg_flat = weighted_l1(edited, bg_target, bg_mask)
    else:
        bg_flat = images.new_tensor(0.0)

    bg_smooth = tv_loss(edited, bg_mask)

    # Rate/compressibility proxy.
    # For BMSHJ, codec_bpp comes from CompressAI likelihoods.
    # For all other families, codec_bpp is zero and TV remains the rate proxy.
    rate_tv = tv_loss(compressed_edited, bg_mask)
    rate_proxy = float(args.rate_tv_scale) * rate_tv + float(args.bmshj_bpp_scale) * codec_bpp

    attn_loss = attention_loss(attention, attn_target)
    attn_sparse = attention.mean()

    target_area = attn_target.mean().detach()
    attn_area_penalty = torch.relu(attention.mean() - args.attn_area_mult * target_area)

    delta_abs = effective_delta.abs().mean()
    delta_obj_abs = delta_obj.abs().mean()
    delta_bg_abs = delta_bg.abs().mean()
    delta_tv = tv_loss(effective_delta)

    if detector_loss_module is not None and args.lambda_det > 0:
        # Detector feature consistency is applied only on object regions.
        # This is critical: the background should be free to become more compressible.
        det_cons = detector_loss_module(
            clean=images,
            degraded=compressed_edited,
            object_mask=det_mask,
        )
    else:
        det_cons = images.new_tensor(0.0)

    # Severity-aware weighting:
    # severe compression should preserve detector features more aggressively.
    # This is the main change for improving q1/q2 AP.
    sev_mean = severity_norm.float().mean().detach()
    task_mult = 1.0 + float(args.task_severity_gain) * sev_mean
    rate_mult = torch.clamp(
        1.0 - float(args.rate_severity_drop) * sev_mean,
        min=float(args.min_rate_mult),
    )
    bg_mult = torch.clamp(
        1.0 - float(args.bg_severity_drop) * sev_mean,
        min=float(args.min_bg_mult),
    )

    loss = (
        (args.lambda_obj_l1 * task_mult) * obj_l1
        + (args.lambda_obj_edge * task_mult) * obj_edge
        + args.lambda_bg_l1 * bg_l1
        + (args.lambda_bg_flat * bg_mult) * bg_flat
        + (args.lambda_bg_smooth * bg_mult) * bg_smooth
        + (args.lambda_rate * rate_mult) * rate_proxy
        + args.lambda_attn * attn_loss
        + args.lambda_attn_sparse * attn_sparse
        + args.lambda_attn_area * attn_area_penalty
        + args.lambda_delta * delta_abs
        + args.lambda_delta_tv * delta_tv
        + (args.lambda_det * task_mult) * det_cons
    )

    with torch.no_grad():
        attn_obj = weighted_mean(attention, attn_target)
        attn_bg = weighted_mean(attention, bg_mask)
        comp_l1_original = (compressed_original - images).abs().mean()
        comp_l1_scope = (compressed_edited - images).abs().mean()

    logs = {
        "loss": loss.detach(),
        "obj_l1": obj_l1.detach(),
        "obj_edge": obj_edge.detach(),
        "bg_l1": bg_l1.detach(),
        "bg_flat": bg_flat.detach(),
        "bg_smooth": bg_smooth.detach(),
        "rate": rate_proxy.detach(),
        "rate_tv": rate_tv.detach(),
        "codec_bpp": codec_bpp.detach(),
        "attn": attn_loss.detach(),
        "attn_mean": attention.mean().detach(),
        "attn_obj": attn_obj.detach(),
        "attn_bg": attn_bg.detach(),
        "attn_area": attn_area_penalty.detach(),
        "delta": delta_abs.detach(),
        "delta_obj": delta_obj_abs.detach(),
        "delta_bg": delta_bg_abs.detach(),
        "delta_tv": delta_tv.detach(),
        "det": det_cons.detach(),
        "comp_orig_l1": comp_l1_original.detach(),
        "comp_edit_l1": comp_l1_scope.detach(),
        "sev": severity_norm.mean().detach(),
        "w_task": task_mult.detach() if torch.is_tensor(task_mult) else torch.tensor(float(task_mult), device=images.device),
        "w_rate": rate_mult.detach() if torch.is_tensor(rate_mult) else torch.tensor(float(rate_mult), device=images.device),
        "w_bg": bg_mult.detach() if torch.is_tensor(bg_mult) else torch.tensor(float(bg_mult), device=images.device),
    }

    return loss, logs


@torch.no_grad()
def validate(
    *,
    args,
    model: nn.Module,
    codec_proxy_manager: Optional[nn.Module],
    detector_loss_module: Optional[nn.Module],
    val_loader: Optional[DataLoader],
    device: torch.device,
    codec_family_to_id: Dict[str, int],
    id_to_codec_family: Dict[int, str],
) -> Dict[str, float]:
    if val_loader is None:
        return {}

    model.eval()

    iterator = iter(val_loader)
    totals: Dict[str, float] = defaultdict(float)
    n = 0

    family_names = list(codec_family_to_id.keys())

    # Deterministic validation grid. Prefer explicit validation-by-family grid,
    # then training-by-family grid, then global validation/global training grid.
    val_grid: List[Tuple[str, float]] = []
    val_levels_by_family = getattr(args, "val_severity_levels_by_family_dict", {}) or {}
    train_levels_by_family = getattr(args, "severity_levels_by_family_dict", {}) or {}

    if val_levels_by_family:
        for fam in family_names:
            vals = val_levels_by_family.get(fam, None)
            if vals:
                for sev in vals:
                    val_grid.append((fam, float(sev)))
    elif train_levels_by_family:
        for fam in family_names:
            vals = train_levels_by_family.get(fam, None)
            if vals:
                for sev in vals:
                    val_grid.append((fam, float(sev)))
    else:
        severity_levels = args.val_severity_levels or args.severity_levels
        if severity_levels is None:
            severity_levels = [args.severity_min, args.severity_max]
        for sev in severity_levels:
            for fam in family_names:
                val_grid.append((fam, float(sev)))

    if not val_grid:
        raise RuntimeError("Empty validation severity/family grid.")

    for step in range(args.val_steps):
        batch, iterator = next_batch(val_loader, iterator)
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B = images.shape[0]

        family_name, raw_sev = val_grid[step % len(val_grid)]

        family_ids = torch.full(
            (B,),
            codec_family_to_id[family_name],
            dtype=torch.long,
            device=device,
        )
        severity_raw = torch.full((B,), float(raw_sev), dtype=torch.float32, device=device)
        severity_norm = normalize_severity_tensor(
            severity_raw,
            severity_min=args.severity_min,
            severity_max=args.severity_max,
            invert=args.invert_severity,
        )

        with torch.no_grad():
            loss, logs = compute_losses(
                args=args,
                model=model,
                codec_proxy_manager=codec_proxy_manager,
                detector_loss_module=detector_loss_module,
                images=images,
                masks=masks,
                family_ids=family_ids,
                severity_norm=severity_norm,
                id_to_codec_family=id_to_codec_family,
            )

        for k, v in logs.items():
            totals[k] += float(v.detach().cpu().item())
        n += 1

    model.train()

    if n == 0:
        return {}

    return {f"val_{k}": v / n for k, v in totals.items()}


def load_model_init_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt_path = str(ckpt_path or "").strip()
    if not ckpt_path:
        return

    print(f"Loading initial SCOPE checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading init checkpoint: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading init checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")

    if isinstance(ckpt, dict) and "codec_family_to_id" in ckpt:
        print(f"Init checkpoint codec_family_to_id: {ckpt['codec_family_to_id']}")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    best_score: float,
    args: argparse.Namespace,
    codec_family_to_id: Dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "codec_family_to_id": codec_family_to_id,
        "severity_min": float(args.severity_min),
        "severity_max": float(args.severity_max),
        "invert_severity": bool(args.invert_severity),
        "base_ch": int(args.base_ch),
        "family_emb_dim": int(args.family_emb_dim),
        "cond_dim": int(args.cond_dim),
        "max_delta_obj": float(args.max_delta_obj),
        "max_delta_bg": float(args.max_delta_bg),
        "attention_bias": float(args.attention_bias),
        "bg_mode": str(args.bg_mode),
        "bg_blur_kernel": int(args.bg_blur_kernel),
        "bg_lowres_scale": int(args.bg_lowres_scale),
        "bg_flat_mode": str(args.bg_flat_mode),
        "args": vars(args),
    }

    torch.save(payload, path)


def print_and_assert_critical_config(args: argparse.Namespace) -> None:
    keys = [
        "bg_mode",
        "bg_blur_kernel",
        "bg_lowres_scale",
        "object_crop_prob",
        "crop_margin",
        "attn_blur",
        "det_mask_blur",
        "attn_area_mult",
        "lambda_obj_l1",
        "lambda_obj_edge",
        "lambda_bg_l1",
        "lambda_bg_flat",
        "bg_flat_mode",
        "lambda_bg_smooth",
        "lambda_rate",
        "lambda_attn",
        "lambda_attn_sparse",
        "lambda_attn_area",
        "lambda_delta",
        "lambda_delta_tv",
        "lambda_det",
        "detector_backend",
        "lambda_obj_l1",
        "lambda_obj_edge",
        "lambda_bg_flat",
        "lambda_bg_smooth",
        "lambda_rate",
        "task_severity_gain",
        "rate_severity_drop",
        "bg_severity_drop",
        "codec_family_probs",
        "severity_probs",
        "severity_levels_by_family",
        "severity_probs_by_family",
        "val_severity_levels_by_family",
        "init_checkpoint",
        "use_real_diffjpeg",
        "diffjpeg_quality_min",
        "diffjpeg_quality_max",
        "jp2_rate_min",
        "jp2_rate_max",
        "bpg_qp_min",
        "bpg_qp_max",
        "bpg_proxy_mode",
        "use_real_bmshj",
        "bmshj_model",
        "bmshj_quality_min",
        "bmshj_quality_max",
        "bmshj_bpp_scale",
        "rate_tv_scale",
    ]

    print("=" * 80, flush=True)
    print("OBJECT_STRONG_BG_REPLACE_CONFIG_SENTINEL", flush=True)
    for k in keys:
        print(f"  {k}: {getattr(args, k, None)}", flush=True)
    print("  severity_conditioning: scalar_input_internal_basis[s,s^2,s^3,hi50,hi75,hi90]", flush=True)
    print("=" * 80, flush=True)

    if not getattr(args, "assert_objstrong", False):
        return

    expected_float = {
        "object_crop_prob": 0.60,
        "crop_margin": 4.0,
        "attn_blur": 3,
        "det_mask_blur": 31,
        "attn_area_mult": 1.0,
        "lambda_obj_l1": 4.0,
        "lambda_obj_edge": 3.0,
        "lambda_bg_l1": 0.0,
        "lambda_bg_flat": 0.75,
        "lambda_bg_smooth": 1.0,
        "lambda_rate": 2.0,
        "lambda_attn": 1.0,
        "lambda_attn_sparse": 0.75,
        "lambda_attn_area": 2.0,
        "lambda_delta": 0.02,
        "lambda_delta_tv": 0.05,
        "lambda_det": 2.0,
    }
    expected_str = {
        "bg_mode": "blur_replace",
        "bg_flat_mode": "blur",
    }

    for k, exp in expected_float.items():
        got = getattr(args, k)
        if abs(float(got) - float(exp)) > 1e-8:
            raise RuntimeError(
                f"Object-strong sanity check failed for {k}: got {got}, expected {exp}. "
                "You are not running the intended SH/PY combination."
            )

    for k, exp in expected_str.items():
        got = getattr(args, k)
        if str(got) != exp:
            raise RuntimeError(
                f"Object-strong sanity check failed for {k}: got {got!r}, expected {exp!r}. "
                "You are not running the intended SH/PY combination."
            )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    if isinstance(args.detector_hook_layers, str):
        if args.detector_hook_layers.strip():
            args.detector_hook_layers = [
                int(x.strip())
                for x in args.detector_hook_layers.split(",")
                if x.strip()
            ]
        else:
            args.detector_hook_layers = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_and_assert_critical_config(args)

    codec_families = parse_csv_strings(args.codec_families)
    codec_family_to_id = build_codec_family_mapping(codec_families)
    id_to_codec_family = {v: k for k, v in codec_family_to_id.items()}

    severity_levels = parse_csv_floats(args.severity_levels) if args.severity_levels else None
    args.severity_levels = severity_levels

    val_severity_levels = (
        parse_csv_floats(args.val_severity_levels)
        if args.val_severity_levels
        else None
    )
    args.val_severity_levels = val_severity_levels

    print(f"Codec family mapping: {codec_family_to_id}")
    print(
        f"Severity normalization: raw [{args.severity_min}, {args.severity_max}] "
        f"-> [0, 1], invert={args.invert_severity}"
    )
    if severity_levels is not None:
        print(f"Training raw severity levels: {severity_levels}")

    family_sampling_probs = parse_sampling_probs(
        args.codec_family_probs,
        names=codec_families,
        expected_len=len(codec_families),
    )
    severity_sampling_probs = None
    if severity_levels is not None:
        severity_sampling_probs = parse_sampling_probs(
            args.severity_probs,
            names=None,
            expected_len=len(severity_levels),
        )

    severity_levels_by_family = parse_family_float_lists(args.severity_levels_by_family)
    severity_probs_by_family = parse_family_prob_lists(
        args.severity_probs_by_family,
        severity_levels_by_family,
    )
    val_severity_levels_by_family = parse_family_float_lists(args.val_severity_levels_by_family)

    unknown_train_fams = [k for k in severity_levels_by_family if k not in codec_family_to_id]
    unknown_val_fams = [k for k in val_severity_levels_by_family if k not in codec_family_to_id]
    if unknown_train_fams:
        raise ValueError(f"severity_levels_by_family has unknown families: {unknown_train_fams}")
    if unknown_val_fams:
        raise ValueError(f"val_severity_levels_by_family has unknown families: {unknown_val_fams}")

    args.family_sampling_probs = family_sampling_probs
    args.severity_sampling_probs = severity_sampling_probs
    args.severity_levels_by_family_dict = severity_levels_by_family
    args.severity_probs_by_family_dict = severity_probs_by_family
    args.val_severity_levels_by_family_dict = val_severity_levels_by_family

    if family_sampling_probs is not None:
        print(f"Family sampling probabilities: {dict(zip(codec_families, family_sampling_probs))}")
    if severity_sampling_probs is not None:
        print(f"Global severity sampling probabilities: {list(zip(severity_levels, severity_sampling_probs))}")
    if severity_levels_by_family:
        print("Family-specific severity levels:")
        for fam, vals in severity_levels_by_family.items():
            print(f"  {fam}: {vals}")
    if severity_probs_by_family:
        print("Family-specific severity probabilities:")
        for fam, vals in severity_probs_by_family.items():
            print(f"  {fam}: {vals}")
    if val_severity_levels_by_family:
        print("Family-specific validation severity levels:")
        for fam, vals in val_severity_levels_by_family.items():
            print(f"  {fam}: {vals}")

    train_loader = make_dataloader(
        images_dir=args.images_dir,
        annotations_json=args.annotations_json,
        crop_size=args.crop_size,
        object_crop_prob=args.object_crop_prob,
        crop_margin=args.crop_margin,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    val_loader = None
    if args.val_images_dir and args.val_annotations_json:
        val_loader = make_dataloader(
            images_dir=args.val_images_dir,
            annotations_json=args.val_annotations_json,
            crop_size=args.crop_size,
            object_crop_prob=args.object_crop_prob,
            crop_margin=args.crop_margin,
            batch_size=args.batch,
            num_workers=args.workers,
        )

    model = SCOPE(
        num_codec_families=len(codec_family_to_id),
        base_ch=args.base_ch,
        family_emb_dim=args.family_emb_dim,
        cond_dim=args.cond_dim,
        max_delta_obj=args.max_delta_obj,
        max_delta_bg=args.max_delta_bg,
        attention_bias=args.attention_bias,
        bg_mode=args.bg_mode,
        bg_blur_kernel=args.bg_blur_kernel,
        bg_lowres_scale=args.bg_lowres_scale,
    ).to(device)

    load_model_init_checkpoint(model, args.init_checkpoint, device)

    codec_proxy_manager = None
    if args.use_real_diffjpeg or args.use_real_bmshj:
        codec_proxy_manager = RealCodecProxyManager(args, device=device).to(device)

    detector_loss_module = None
    if args.detector_backend == "detectron2" and args.lambda_det > 0:
        feature_names = parse_csv_strings(args.detectron2_feature_names)
        print(f"Loading Detectron2 FPN feature consistency loss: {args.detectron2_config}")
        detector_loss_module = FrozenDetectron2FPNConsistency(
            config_file=args.detectron2_config,
            weights=args.detectron2_weights,
            device=device,
            feature_names=feature_names,
            loss_type=args.detector_feature_loss,
            normalize=True,
            mask_weight=args.d2_fpn_mask_weight,
            full_weight=args.d2_fpn_full_weight,
            input_format=args.detectron2_input_format,
        )
        detector_loss_module.eval()
    elif args.detector_backend == "yolo" and args.detector_weights and args.lambda_det > 0:
        print(f"Loading Ultralytics YOLO feature consistency loss: {args.detector_weights}")
        detector_loss_module = FrozenYOLOFeatureConsistency(
            weights=args.detector_weights,
            device=device,
            hook_layers=args.detector_hook_layers,
            loss_type=args.detector_feature_loss,
            normalize=True,
            print_layers=args.print_detector_layers,
        )
        detector_loss_module.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, args.samples_per_epoch // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        if args.scheduler == "cosine":
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_iter = iter(train_loader)
    best_score = float("inf")
    global_step = 0

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "codec_family_to_id": codec_family_to_id,
            },
            f,
            indent=2,
        )

    print("Start training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        accum: Dict[str, float] = defaultdict(float)
        fam_counts = defaultdict(int)

        for step in range(steps_per_epoch):
            batch, train_iter = next_batch(train_loader, train_iter)

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            B = images.shape[0]

            family_ids, severity_raw, severity_norm = sample_conditions(
                batch_size=B,
                device=device,
                codec_family_to_id=codec_family_to_id,
                severity_levels=severity_levels,
                severity_min=args.severity_min,
                severity_max=args.severity_max,
                invert_severity=args.invert_severity,
                family_probs=family_sampling_probs,
                severity_probs=severity_sampling_probs,
                severity_levels_by_family=args.severity_levels_by_family_dict,
                severity_probs_by_family=args.severity_probs_by_family_dict,
                id_to_codec_family=id_to_codec_family,
            )

            for fid in family_ids.detach().cpu().tolist():
                fam_counts[id_to_codec_family[int(fid)]] += 1

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                loss, logs = compute_losses(
                    args=args,
                    model=model,
                    codec_proxy_manager=codec_proxy_manager,
                    detector_loss_module=detector_loss_module,
                    images=images,
                    masks=masks,
                    family_ids=family_ids,
                    severity_norm=severity_norm,
                    id_to_codec_family=id_to_codec_family,
                )

            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at epoch={epoch} step={step}: {loss}")
                continue

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k, v in logs.items():
                accum[k] += float(v.detach().cpu().item())

            global_step += 1

            if args.log_every > 0 and (
                (step + 1) % args.log_every == 0 or (step + 1) == steps_per_epoch
            ):
                elapsed = time.time() - t0
                steps_done = step + 1
                steps_per_sec = steps_done / max(elapsed, 1e-6)
                eta_sec = (steps_per_epoch - steps_done) / max(steps_per_sec, 1e-6)

                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"[{steps_done}/{steps_per_epoch}] "
                    f"{100.0 * steps_done / steps_per_epoch:.1f}% "
                    f"step/s={steps_per_sec:.2f} "
                    f"eta={eta_sec / 60.0:.1f}m | "
                    f"{format_live_metrics(accum, steps_done, optimizer.param_groups[0]['lr'], device)}",
                    flush=True,
                )

        train_metrics = {k: v / steps_per_epoch for k, v in accum.items()}
        train_metrics["lr"] = optimizer.param_groups[0]["lr"]
        train_metrics["time_sec"] = time.time() - t0

        val_metrics = {}
        if val_loader is not None and (epoch % args.val_every == 0):
            val_metrics = validate(
                args=args,
                model=model,
                codec_proxy_manager=codec_proxy_manager,
                detector_loss_module=detector_loss_module,
                val_loader=val_loader,
                device=device,
                codec_family_to_id=codec_family_to_id,
                id_to_codec_family=id_to_codec_family,
            )

        # Use validation loss for checkpoint selection when validation is available.
        # IMPORTANT: do not let non-validation epochs overwrite the best validation
        # checkpoint just because training loss is lower.
        has_val = "val_loss" in val_metrics
        score = val_metrics["val_loss"] if has_val else train_metrics["loss"]

        is_best = bool(has_val and score < best_score)
        if is_best:
            best_score = score

        metrics = {}
        metrics.update(train_metrics)
        metrics.update(val_metrics)
        metrics["score"] = score
        metrics["best"] = 1 if is_best else 0

        fam_str = ", ".join(f"{k}:{v}" for k, v in sorted(fam_counts.items()))

        print(
            f"Epoch {epoch}: "
            f"{format_metrics(metrics)}, "
            f"families=({fam_str})"
        )

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            args=args,
            codec_family_to_id=codec_family_to_id,
        )

        if is_best:
            # Keep the old filename for compatibility with existing eval scripts,
            # and also save an explicit validation-best name.
            for best_name in ["best_loss.pt", "best_val.pt"]:
                save_checkpoint(
                    output_dir / best_name,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_score=best_score,
                    args=args,
                    codec_family_to_id=codec_family_to_id,
                )

    print(f"Done. Best validation score: {best_score:.6f}")
    print(f"Checkpoints saved in: {output_dir}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

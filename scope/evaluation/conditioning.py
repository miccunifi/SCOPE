from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch


def normalize_codec_family_name(name: str) -> str:
    aliases = {
        "diffjpeg": "diff_jpeg",
        "diff_jpeg": "diff_jpeg",
        "jpeg": "diff_jpeg",
        "block_dct": "diff_jpeg",

        "jp2_like": "jp2_like",
        "jp2": "jp2_like",
        "jpeg2000": "jp2_like",
        "wavelet": "jp2_like",

        "bpglike": "bpg_like",
        "bpg_like": "bpg_like",
        "bpg": "bpg_like",
        "intra_smooth": "bpg_like",

        "bmshj": "bmshj",
        "learned": "bmshj",
        "compressai": "bmshj",
        "cheng": "bmshj",
    }

    name = str(name).strip()
    if name not in aliases:
        raise ValueError(
            f"Unknown codec_family={name!r}. "
            "Use diff_jpeg, jp2_like, bpg_like, bmshj, or cheng."
        )

    return aliases[name]


def infer_codec_family_from_experiment(exp_cfg: Dict) -> str:
    explicit = exp_cfg.get("codec_family")
    if explicit is not None:
        return normalize_codec_family_name(str(explicit))

    mode = str(exp_cfg.get("mode", ""))
    codec = str(exp_cfg.get("compressai_model", "")).lower()

    if mode in {"uniform", "jpeg", "scope_jpeg"}:
        return "diff_jpeg"
    if mode in {"jpeg2000", "jp2", "scope_jpeg2000", "scope_jp2"}:
        return "jp2_like"
    if mode in {"bpg", "scope_bpg"}:
        return "bpg_like"

    if any(k in codec for k in ["jpeg2000", "jp2", "j2k"]):
        return "jp2_like"
    if any(k in codec for k in ["bpg", "heic", "heif", "avif", "jxl"]):
        return "bpg_like"

    return "bmshj"


def infer_severity_from_experiment(exp_cfg: Dict) -> float:
    mode = str(exp_cfg.get("mode", ""))
    codec = str(exp_cfg.get("compressai_model", "")).lower()

    if mode in {"uniform", "jpeg", "scope_jpeg"}:
        q = float(exp_cfg.get("jpeg_quality", 10))
        return float(np.clip(1.0 - (q - 10.0) / max(60.0 - 10.0, 1e-6), 0.0, 1.0))

    if mode in {"jpeg2000", "jp2", "scope_jpeg2000", "scope_jp2"}:
        r = float(exp_cfg.get("jp2_rate", 32))
        r_min = float(exp_cfg.get("jp2_rate_min", 1))
        r_max = float(exp_cfg.get("jp2_rate_max", 192))
        return float(np.clip((r - r_min) / max(r_max - r_min, 1e-6), 0.0, 1.0))

    if mode in {"bpg", "scope_bpg"}:
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
    # 1) Preferred: explicit conditioning severity from YAML.
    cond_cfg = exp_cfg.get("condition", {}) or {}
    if "severity" in cond_cfg:
        return float(np.clip(float(cond_cfg["severity"]), 0.0, 1.0))

    # 2) Also allow severity directly inside the experiment.
    if "severity" in exp_cfg:
        return float(np.clip(float(exp_cfg["severity"]), 0.0, 1.0))

    # 3) Fallback: infer from codec settings.
    return infer_severity_from_experiment(exp_cfg)


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

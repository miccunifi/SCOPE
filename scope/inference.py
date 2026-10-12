from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .model import SCOPE


_MODEL_CACHE = {}
_ARCH_FIELDS = (
    "base_ch", "family_emb_dim", "cond_dim", "max_delta_obj", "max_delta_bg",
    "attention_bias", "bg_mode", "bg_blur_kernel", "bg_lowres_scale",
)
_DEFAULTS = {
    "base_ch": 48, "family_emb_dim": 8, "cond_dim": 64,
    "max_delta_obj": 0.09, "max_delta_bg": 0.03,
    "attention_bias": -4.0, "bg_mode": "blur_replace",
    "bg_blur_kernel": 51, "bg_lowres_scale": 16,
}


def load_scope_model(ckpt_path: str, device: torch.device, exp_cfg: dict | None = None):
    exp_cfg = exp_cfg or {}
    arch_override = {k: exp_cfg[k] for k in _ARCH_FIELDS if k in exp_cfg}
    key = (str(Path(ckpt_path).resolve()), str(device), json.dumps(arch_override, sort_keys=True))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid SCOPE checkpoint: {ckpt_path}")
    mapping = checkpoint.get("codec_family_to_id")
    if not mapping:
        raise ValueError(f"Missing codec_family_to_id in checkpoint: {ckpt_path}")
    mapping = {str(name): int(index) for name, index in mapping.items()}

    args = checkpoint.get("args") or {}
    if not isinstance(args, dict):
        args = vars(args)

    config = {name: exp_cfg.get(name, checkpoint.get(name, args.get(name, default)))
              for name, default in _DEFAULTS.items()}
    model = SCOPE(num_codec_families=len(mapping), **config)
    state = checkpoint["model"]
    if any(name.startswith("module.") for name in state):
        state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    _MODEL_CACHE[key] = (model, mapping)
    return model, mapping


@torch.no_grad()
def apply_scope_bgr(
    img_bgr: np.ndarray,
    scope_ckpt: str,
    codec_family: str,
    severity: float,
    device: str | torch.device | None = None,
    exp_cfg: dict | None = None,
) -> np.ndarray:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, codec_family_to_id = load_scope_model(scope_ckpt, torch_device, exp_cfg)
    if codec_family not in codec_family_to_id:
        raise ValueError(f"Codec family {codec_family!r} not in {list(codec_family_to_id)}")
    if not 0 <= severity <= 1:
        raise ValueError("severity must be between 0 and 1")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(
        torch_device, dtype=torch.float32
    ) / 255.0
    family = torch.tensor([codec_family_to_id[codec_family]], device=torch_device)
    sev = torch.tensor([severity], device=torch_device, dtype=x.dtype)
    edited, *_ = model(x, severity=sev, codec_family=family)
    result = (edited[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

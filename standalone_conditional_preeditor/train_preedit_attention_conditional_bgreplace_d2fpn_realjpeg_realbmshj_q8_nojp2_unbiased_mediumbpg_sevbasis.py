#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Model: codec-family + normalized-severity conditional pre-editor
# =============================================================================


class FiLM(nn.Module):
    """
    Feature-wise linear modulation.

    x:    [B, C, H, W]
    cond: [B, cond_dim]

    y = x * (1 + gamma) + beta

    Initialized as identity modulation.
    """

    def __init__(self, num_channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * num_channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class ConditionalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film1 = FiLM(out_ch, cond_dim)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.film2 = FiLM(out_ch, cond_dim)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.film1(x, cond)

        x = self.conv2(x)
        x = self.act(x)
        x = self.film2(x, cond)

        return x


class ConditionalAttentionPreEditor(nn.Module):
    """
    Conditional image-only pre-editor.

    Inputs:
        x:
            image in [0, 1], [B, 3, H, W]

        severity:
            normalized compression severity in [0, 1], [B] or [B, 1]
            0 = mild / high quality
            1 = severe / low quality

        codec_family:
            integer codec family id, [B]

    Outputs:
        edited, delta_obj, delta_bg, attention, effective_delta
    """

    def __init__(
        self,
        num_codec_families: int,
        base_ch: int = 48,
        family_emb_dim: int = 8,
        cond_dim: int = 64,
        max_delta_obj: float = 0.09,
        max_delta_bg: float = 0.03,
        attention_bias: float = -3.0,
        bg_mode: str = "residual",
        bg_blur_kernel: int = 41,
        bg_lowres_scale: int = 16,
    ):
        super().__init__()

        self.num_codec_families = int(num_codec_families)
        self.max_delta_obj = float(max_delta_obj)
        self.max_delta_bg = float(max_delta_bg)

        self.bg_mode = str(bg_mode)
        self.bg_blur_kernel = int(bg_blur_kernel)
        self.bg_lowres_scale = int(bg_lowres_scale)

        valid_bg_modes = {"residual", "blur_replace", "lowres_replace", "mean_replace"}
        if self.bg_mode not in valid_bg_modes:
            raise ValueError(
                f"Invalid bg_mode={self.bg_mode!r}. "
                f"Expected one of {sorted(valid_bg_modes)}."
            )

        self.family_emb = nn.Embedding(
            num_embeddings=self.num_codec_families,
            embedding_dim=family_emb_dim,
        )

        # Keep the public condition as a single scalar severity s in [0, 1].
        # Internally expand it to a small fixed basis so the condition MLP can
        # learn nonlinear responses at high compression without changing the
        # meaning of severity, the sampler, or the evaluation interface.
        # Features: s, s^2, s^3, and three high-severity hinge/gate features.
        self.severity_basis_dim = 6

        self.cond_mlp = nn.Sequential(
            nn.Linear(family_emb_dim + self.severity_basis_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )

        self.enc1 = ConditionalConvBlock(3, base_ch, cond_dim)

        self.down1 = nn.Conv2d(
            base_ch,
            base_ch * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc2 = ConditionalConvBlock(base_ch * 2, base_ch * 2, cond_dim)

        self.down2 = nn.Conv2d(
            base_ch * 2,
            base_ch * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.mid = ConditionalConvBlock(base_ch * 4, base_ch * 4, cond_dim)

        # Bilinear upsample + conv is less artifact-prone than ConvTranspose2d.
        self.up2_conv = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1)
        self.dec2 = ConditionalConvBlock(base_ch * 4, base_ch * 2, cond_dim)

        self.up1_conv = nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1)
        self.dec1 = ConditionalConvBlock(base_ch * 2, base_ch, cond_dim)

        self.final_film = FiLM(base_ch, cond_dim)

        self.delta_obj_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.delta_bg_head = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
        self.attn_head = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)

        self._init_heads(attention_bias=attention_bias)

    def _init_heads(self, attention_bias: float) -> None:
        nn.init.normal_(self.delta_obj_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.delta_obj_head.bias)

        nn.init.normal_(self.delta_bg_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.delta_bg_head.bias)

        # sigmoid(-3) ~= 0.047
        nn.init.normal_(self.attn_head.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.attn_head.bias, float(attention_bias))

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _reflect_avg_blur(x: torch.Tensor, k: int) -> torch.Tensor:
        """Low-pass image used as a cheap background replacement."""
        if k <= 1:
            return x
        if k % 2 == 0:
            k += 1
        pad = k // 2

        # reflect padding requires spatial size > pad. Fall back to replicate
        # for tiny crops, although normal training crops are 512x512.
        mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
        xp = F.pad(x, (pad, pad, pad, pad), mode=mode)
        return F.avg_pool2d(xp, kernel_size=k, stride=1)

    def _background_base(self, x: torch.Tensor) -> torch.Tensor:
        """Return a compressible background replacement base.

        residual:
            old behavior, background branch is x + delta_bg_residual.

        blur_replace:
            background branch is heavy_blur(x) + delta_bg_residual.

        lowres_replace:
            background branch is downsample/upsample(x) + delta_bg_residual.

        mean_replace:
            background branch is per-image RGB mean + delta_bg_residual.

        This is what makes the background disposable: it can be replaced by a
        low-frequency/flat version rather than only receiving a small residual.
        """
        if self.bg_mode == "residual":
            return x

        if self.bg_mode == "blur_replace":
            return self._reflect_avg_blur(x, self.bg_blur_kernel)

        if self.bg_mode == "lowres_replace":
            B, C, H, W = x.shape
            scale = max(2, int(self.bg_lowres_scale))
            h2 = max(4, H // scale)
            w2 = max(4, W // scale)
            z = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
            y = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
            return y

        if self.bg_mode == "mean_replace":
            return x.mean(dim=(2, 3), keepdim=True).expand_as(x)

        raise RuntimeError(f"Unhandled bg_mode={self.bg_mode!r}")

    @staticmethod
    def _expand_scalar_severity(severity: torch.Tensor) -> torch.Tensor:
        """Expand scalar severity into internal nonlinear conditioning features.

        The external/public severity remains a scalar normalized compression
        parameter in [0, 1]. This function only gives the condition MLP enough
        capacity to learn that high-compression regimes may require a different
        response than mild compression regimes.

        Returned basis, for s in [0, 1]:
            [s, s^2, s^3, relu((s-0.50)/0.50),
             relu((s-0.75)/0.25), relu((s-0.90)/0.10)]
        """
        if severity.ndim == 1:
            severity = severity[:, None]

        s = severity.float().clamp(0.0, 1.0)
        s2 = s * s
        s3 = s2 * s
        high50 = torch.clamp((s - 0.50) / 0.50, min=0.0, max=1.0)
        high75 = torch.clamp((s - 0.75) / 0.25, min=0.0, max=1.0)
        high90 = torch.clamp((s - 0.90) / 0.10, min=0.0, max=1.0)
        return torch.cat([s, s2, s3, high50, high75, high90], dim=1)

    def _make_condition(
        self,
        severity: torch.Tensor,
        codec_family: torch.Tensor,
    ) -> torch.Tensor:
        codec_family = codec_family.long().view(-1)

        family_vec = self.family_emb(codec_family)
        severity_basis = self._expand_scalar_severity(severity)
        cond_in = torch.cat([family_vec, severity_basis], dim=1)
        cond = self.cond_mlp(cond_in)
        return cond

    def forward(
        self,
        x: torch.Tensor,
        severity: torch.Tensor,
        codec_family: torch.Tensor,
    ):
        cond = self._make_condition(
            severity=severity,
            codec_family=codec_family,
        )

        e1 = self.enc1(x, cond)

        x2 = self.down1(e1)
        e2 = self.enc2(x2, cond)

        x3 = self.down2(e2)
        m = self.mid(x3, cond)

        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2_conv(u2)
        u2 = self._match_size(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1_conv(u1)
        u1 = self._match_size(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)

        d1 = self.final_film(d1, cond)

        delta_obj = self.max_delta_obj * torch.tanh(self.delta_obj_head(d1))
        delta_bg_residual = self.max_delta_bg * torch.tanh(self.delta_bg_head(d1))
        attention = torch.sigmoid(self.attn_head(d1))

        # Object path: preserve object pixels with a bounded residual edit.
        obj_part = torch.clamp(x + delta_obj, 0.0, 1.0)

        # Background path: either old residual behavior, or replacement by a
        # cheap low-frequency/flat background plus a bounded residual.
        bg_base = self._background_base(x)
        bg_part = torch.clamp(bg_base + delta_bg_residual, 0.0, 1.0)

        # Mix object-preserving and background-simplified images.
        edited = attention * obj_part + (1.0 - attention) * bg_part
        edited = torch.clamp(edited, 0.0, 1.0)

        effective_delta = edited - x

        # For logging/losses, expose the actual background edit relative to the
        # input, not only the small residual around bg_base.
        delta_bg = bg_part - x

        return edited, delta_obj, delta_bg, attention, effective_delta


# =============================================================================
# COCO-format dataset
# =============================================================================


class CocoBoxCropDataset(Dataset):
    """
    COCO-format image dataset returning:
        image: [3, crop_size, crop_size], float in [0, 1]
        mask:  [1, crop_size, crop_size], object box mask in {0, 1}

    This does not require pycocotools.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_json: str | Path,
        crop_size: int = 512,
        object_crop_prob: float = 0.90,
        crop_margin: float = 1.8,
        min_box_area: float = 4.0,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_json = Path(annotations_json)
        self.crop_size = int(crop_size)
        self.object_crop_prob = float(object_crop_prob)
        self.crop_margin = float(crop_margin)
        self.min_box_area = float(min_box_area)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images_dir: {self.images_dir}")

        if not self.annotations_json.exists():
            raise FileNotFoundError(f"Missing annotations_json: {self.annotations_json}")

        with open(self.annotations_json, "r") as f:
            coco = json.load(f)

        self.images = []
        self.image_id_to_info = {}

        for im in coco["images"]:
            image_id = int(im["id"])
            file_name = im["file_name"]
            width = int(im.get("width", 0))
            height = int(im.get("height", 0))
            info = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
            self.images.append(info)
            self.image_id_to_info[image_id] = info

        self.anns_by_image: Dict[int, List[List[float]]] = defaultdict(list)

        for ann in coco.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue

            image_id = int(ann["image_id"])
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = map(float, bbox)
            if w <= 1 or h <= 1 or (w * h) < self.min_box_area:
                continue

            self.anns_by_image[image_id].append([x, y, w, h])

        self.object_indices = [
            idx
            for idx, im in enumerate(self.images)
            if len(self.anns_by_image[int(im["id"])]) > 0
        ]

        if len(self.images) == 0:
            raise RuntimeError("No images found in COCO JSON.")

        print(
            f"Loaded COCO dataset: images={len(self.images)} "
            f"images_with_boxes={len(self.object_indices)} "
            f"annotations={sum(len(v) for v in self.anns_by_image.values())}"
        )

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).contiguous()

    def _choose_index(self) -> int:
        if (
            len(self.object_indices) > 0
            and random.random() < self.object_crop_prob
        ):
            return random.choice(self.object_indices)
        return random.randrange(len(self.images))

    def _compute_crop_box(
        self,
        W: int,
        H: int,
        boxes: Sequence[Sequence[float]],
    ) -> Tuple[int, int, int, int]:
        # Fallback random square-ish crop.
        if len(boxes) == 0 or random.random() > self.object_crop_prob:
            side = min(W, H)
            if side <= 0:
                return 0, 0, W, H
            if W == side:
                x0 = 0
            else:
                x0 = random.randint(0, max(0, W - side))
            if H == side:
                y0 = 0
            else:
                y0 = random.randint(0, max(0, H - side))
            return x0, y0, x0 + side, y0 + side

        x, y, w, h = random.choice(list(boxes))
        cx = x + 0.5 * w
        cy = y + 0.5 * h

        side = max(w, h) * self.crop_margin
        side = max(side, 64.0)
        side = min(side, float(max(W, H)))

        # Jitter around object center.
        jitter = 0.15 * side
        cx += random.uniform(-jitter, jitter)
        cy += random.uniform(-jitter, jitter)

        x0 = int(round(cx - 0.5 * side))
        y0 = int(round(cy - 0.5 * side))
        x1 = int(round(cx + 0.5 * side))
        y1 = int(round(cy + 0.5 * side))

        # Clamp while preserving crop size as much as possible.
        crop_w = x1 - x0
        crop_h = y1 - y0

        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > W:
            shift = x1 - W
            x0 -= shift
            x1 = W
        if y1 > H:
            shift = y1 - H
            y0 -= shift
            y1 = H

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, max(x0 + 1, x1))
        y1 = min(H, max(y0 + 1, y1))

        return x0, y0, x1, y1

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        # Ignore the dataloader index so every access gets a random crop.
        idx = self._choose_index()
        info = self.images[idx]
        image_id = int(info["id"])
        path = self.images_dir / info["file_name"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Robust fallback: pick another image.
            idx = random.randrange(len(self.images))
            info = self.images[idx]
            image_id = int(info["id"])
            path = self.images_dir / info["file_name"]
            img = Image.open(path).convert("RGB")

        W, H = img.size
        boxes = self.anns_by_image.get(image_id, [])

        mask_full = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask_full)
        for x, y, w, h in boxes:
            x0 = max(0, int(math.floor(x)))
            y0 = max(0, int(math.floor(y)))
            x1 = min(W, int(math.ceil(x + w)))
            y1 = min(H, int(math.ceil(y + h)))
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1, y1], fill=255)

        crop_box = self._compute_crop_box(W, H, boxes)
        img = img.crop(crop_box)
        mask = mask_full.crop(crop_box)

        img = img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)

        image_t = self._pil_to_tensor(img)
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        mask_t = torch.from_numpy(mask_arr[None, :, :]).contiguous()

        return {
            "image": image_t,
            "mask": mask_t,
        }


# =============================================================================
# Conditioning helpers
# =============================================================================


def parse_csv_strings(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError(f"Empty CSV string: {s!r}")
    return out


def parse_csv_floats(s: str) -> List[float]:
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError(f"Empty CSV float string: {s!r}")
    return out


def parse_sampling_probs(
    s: str,
    names: Optional[Sequence[str]] = None,
    expected_len: Optional[int] = None,
) -> Optional[List[float]]:
    """Parse comma-separated probabilities.

    Accepted formats:
      - "0.4,0.2,0.2,0.2"
      - "diff_jpeg:0.34,bmshj:0.33,bpg_like:0.33"

    If names is provided, named probabilities are returned in that order.
    Probabilities are normalized to sum to 1.
    """
    s = str(s or "").strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None

    if any(":" in p for p in parts):
        if names is None:
            raise ValueError("Named probabilities require names.")
        mapping = {}
        for p in parts:
            if ":" not in p:
                raise ValueError(f"Mixed named/unnamed probability entry: {p!r}")
            k, v = p.split(":", 1)
            mapping[k.strip()] = float(v.strip())
        missing = [n for n in names if n not in mapping]
        if missing:
            raise ValueError(f"Missing probabilities for: {missing}")
        vals = [float(mapping[n]) for n in names]
    else:
        vals = [float(p) for p in parts]

    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} probabilities, got {len(vals)}: {vals}")

    total = float(sum(vals))
    if total <= 0:
        raise ValueError(f"Probability sum must be > 0, got {vals}")

    vals = [float(v) / total for v in vals]
    return vals



def parse_family_float_lists(s: str) -> Dict[str, List[float]]:
    """Parse family-specific float lists.

    Format:
        "diff_jpeg:0,0.25,0.5;bmshj:0,0.142857,1;bpg_like:0,0.1,1"

    This is used for family-specific severity grids. It lets JPEG sample
    severity values corresponding to JPEG qualities, BMSHJ sample its discrete
    quality indices, and BPG sample QP-derived severities.
    """
    s = str(s or "").strip()
    if not s:
        return {}

    out: Dict[str, List[float]] = {}
    for entry in [e.strip() for e in s.split(";") if e.strip()]:
        if ":" not in entry:
            raise ValueError(
                f"Invalid family float-list entry {entry!r}. Expected family:v1,v2,..."
            )
        name, values = entry.split(":", 1)
        vals = [float(v.strip()) for v in values.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"No values specified for family {name!r}")
        out[name.strip()] = vals
    return out


def parse_family_prob_lists(s: str, levels_by_family: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Parse family-specific probability lists and normalize each family.

    Format mirrors parse_family_float_lists:
        "diff_jpeg:1,1,2;bmshj:1,2,2;bpg_like:1,1,2"

    Length for each family must match the corresponding severity level list.
    Missing family means uniform sampling for that family.
    """
    raw = parse_family_float_lists(s)
    out: Dict[str, List[float]] = {}
    for family, probs in raw.items():
        if family not in levels_by_family:
            raise ValueError(
                f"Severity probabilities specified for unknown family {family!r}. "
                f"Known families: {sorted(levels_by_family)}"
            )
        if len(probs) != len(levels_by_family[family]):
            raise ValueError(
                f"severity_probs_by_family[{family}] has length {len(probs)} but "
                f"severity_levels_by_family[{family}] has length {len(levels_by_family[family])}"
            )
        total = float(sum(probs))
        if total <= 0:
            raise ValueError(f"Probability sum for {family} must be > 0, got {probs}")
        out[family] = [float(p) / total for p in probs]
    return out


def build_codec_family_mapping(codec_families: Sequence[str]) -> Dict[str, int]:
    if len(set(codec_families)) != len(codec_families):
        raise ValueError(f"Duplicate codec family names: {codec_families}")
    return {name: idx for idx, name in enumerate(codec_families)}


def normalize_severity_tensor(
    raw: torch.Tensor,
    severity_min: float,
    severity_max: float,
    invert: bool = False,
) -> torch.Tensor:
    if severity_max <= severity_min:
        raise ValueError(
            f"severity_max must be > severity_min, got {severity_min=} {severity_max=}"
        )
    s = (raw.float() - float(severity_min)) / float(severity_max - severity_min)
    s = s.clamp(0.0, 1.0)
    if invert:
        s = 1.0 - s
    return s


def sample_conditions(
    batch_size: int,
    device: torch.device,
    codec_family_to_id: Dict[str, int],
    severity_levels: Optional[List[float]],
    severity_min: float,
    severity_max: float,
    invert_severity: bool,
    family_probs: Optional[List[float]] = None,
    severity_probs: Optional[List[float]] = None,
    severity_levels_by_family: Optional[Dict[str, List[float]]] = None,
    severity_probs_by_family: Optional[Dict[str, List[float]]] = None,
    id_to_codec_family: Optional[Dict[int, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        family_ids: [B], long
        severity_raw: [B], float
        severity_norm: [B], float in [0, 1]

    The old sampler used one global severity grid for all codec families.
    This updated sampler can use family-specific grids/probabilities.

    Why this matters:
        codec parameter -> bitrate is not linear.
        A severity grid derived from actual JPEG qualities, BMSHJ quality
        indices, and BPG QP values gives better coverage than uniform 0..1.
    """
    n_families = len(codec_family_to_id)
    severity_levels_by_family = severity_levels_by_family or {}
    severity_probs_by_family = severity_probs_by_family or {}

    if family_probs is not None:
        p = torch.tensor(family_probs, dtype=torch.float32, device=device)
        p = p / p.sum().clamp_min(1e-8)
        family_ids = torch.multinomial(p, num_samples=batch_size, replacement=True).long()
    else:
        family_ids = torch.randint(
            low=0,
            high=n_families,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    if severity_levels_by_family:
        if id_to_codec_family is None:
            id_to_codec_family = {v: k for k, v in codec_family_to_id.items()}

        severity_values: List[torch.Tensor] = []
        for i in range(batch_size):
            family_id = int(family_ids[i].detach().cpu().item())
            family_name = id_to_codec_family[family_id]
            levels_list = severity_levels_by_family.get(family_name, severity_levels)

            if levels_list is None:
                # Fallback continuous uniform if no grid is defined for this family.
                val = torch.empty((), dtype=torch.float32, device=device)
                val.uniform_(float(severity_min), float(severity_max))
                severity_values.append(val)
                continue

            levels = torch.tensor(levels_list, dtype=torch.float32, device=device)
            probs_list = severity_probs_by_family.get(family_name, None)
            if probs_list is not None:
                p = torch.tensor(probs_list, dtype=torch.float32, device=device)
                if p.numel() != levels.numel():
                    raise ValueError(
                        f"severity_probs_by_family[{family_name}] length {p.numel()} "
                        f"!= levels length {levels.numel()}"
                    )
                p = p / p.sum().clamp_min(1e-8)
                idx = torch.multinomial(p, num_samples=1, replacement=True)[0]
            else:
                idx = torch.randint(0, levels.numel(), size=(), device=device)
            severity_values.append(levels[idx])

        severity_raw = torch.stack(severity_values, dim=0).to(device=device, dtype=torch.float32)

    elif severity_levels is not None:
        levels = torch.tensor(severity_levels, dtype=torch.float32, device=device)
        if severity_probs is not None:
            p = torch.tensor(severity_probs, dtype=torch.float32, device=device)
            if p.numel() != levels.numel():
                raise ValueError(
                    f"severity_probs length {p.numel()} != severity_levels length {levels.numel()}"
                )
            p = p / p.sum().clamp_min(1e-8)
            idx = torch.multinomial(p, num_samples=batch_size, replacement=True)
        else:
            idx = torch.randint(0, levels.numel(), size=(batch_size,), device=device)
        severity_raw = levels[idx]
    else:
        severity_raw = torch.empty(batch_size, dtype=torch.float32, device=device)
        severity_raw.uniform_(float(severity_min), float(severity_max))

    severity_norm = normalize_severity_tensor(
        severity_raw,
        severity_min=severity_min,
        severity_max=severity_max,
        invert=invert_severity,
    )

    return family_ids, severity_raw, severity_norm


# =============================================================================
# Differentiable codec-family proxies
# =============================================================================


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def quantize_ste(x: torch.Tensor, step: torch.Tensor | float) -> torch.Tensor:
    """STE quantization for image-like tensors in [0, 1]."""
    return torch.clamp(ste_round(x / step) * step, 0.0, 1.0)


def quantize_ste_unclamped(x: torch.Tensor, step: torch.Tensor | float) -> torch.Tensor:
    """STE quantization for signed transform/residual coefficients."""
    return ste_round(x / step) * step


def avg_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def block_average(x: torch.Tensor, block: int) -> torch.Tensor:
    if block <= 1:
        return x

    B, C, H, W = x.shape
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block

    mode = "reflect" if H > 1 and W > 1 else "replicate"
    xp = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    pooled = F.avg_pool2d(xp, kernel_size=block, stride=block)
    up = F.interpolate(pooled, size=xp.shape[-2:], mode="nearest")
    return up[:, :, :H, :W]


# -----------------------------------------------------------------------------
# Better internal differentiable proxies for the two non-real families.
#
# JPEG and BMSHJ can use real differentiable/learned codecs in RealCodecProxyManager.
# JPEG2000 and BPG are still approximations, but these are structurally closer
# than simple blur+quantization:
#   - jp2_like: multi-level Haar wavelet transform + subband/dead-zone quantization
#   - bpg_like: YCbCr, chroma subsampling, intra-prediction residual quantization,
#               deblocking/smoothing
# -----------------------------------------------------------------------------


def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    H, W = x.shape[-2:]
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        mode = "reflect" if H > 1 and W > 1 else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (H, W)


def haar_dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[int, int]]:
    """One differentiable orthonormal-ish 2D Haar analysis step.

    Returns LL, (LH, HL, HH), original spatial shape. The inverse below crops
    back to the original shape, so odd image sizes are safe.
    """
    x, shape = _pad_even(x)
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]

    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh), shape


def haar_idwt2(
    ll: torch.Tensor,
    bands: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shape: Tuple[int, int],
) -> torch.Tensor:
    """Inverse of haar_dwt2()."""
    lh, hl, hh = bands
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5

    B, C, H2, W2 = ll.shape
    y = torch.empty(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
    y[..., 0::2, 0::2] = a
    y[..., 0::2, 1::2] = b
    y[..., 1::2, 0::2] = c
    y[..., 1::2, 1::2] = d

    H, W = shape
    return y[..., :H, :W]


def jp2_wavelet_proxy(x: torch.Tensor, severity: torch.Tensor, levels: int = 3) -> torch.Tensor:
    """JPEG2000-like differentiable wavelet proxy.

    This is not a real JPEG2000 codec, but it mimics the important part that
    the old proxy missed: multi-resolution wavelet subbands with stronger
    dead-zone quantization on high-frequency bands.
    """
    s = severity.float().clamp(0.0, 1.0).view(1, 1, 1, 1)

    cur = x
    pyramids: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[int, int]]] = []

    # Build a small wavelet pyramid. Stop before tiny maps for stability.
    max_levels = max(1, int(levels))
    for _ in range(max_levels):
        if min(cur.shape[-2:]) < 16:
            break
        ll, bands, shape = haar_dwt2(cur)
        pyramids.append((bands, shape))
        cur = ll

    # Coarsest LL is quantized lightly. Details are quantized harder.
    ll_step = (0.5 + 7.0 * s) / 255.0
    cur = quantize_ste_unclamped(cur, ll_step)

    n_levels = max(1, len(pyramids))
    for rev_idx, (bands, shape) in enumerate(reversed(pyramids)):
        # rev_idx=0 is the coarsest detail level, larger is finer detail.
        fine_factor = 1.0 + 0.35 * (n_levels - 1 - rev_idx)
        lh, hl, hh = bands

        # JPEG2000 uses dead-zone quantization; high-frequency HH is hit hardest.
        step_lh = (0.9 + 18.0 * s) * fine_factor / 255.0
        step_hh = (1.2 + 26.0 * s) * fine_factor / 255.0
        thr_lh = (0.15 + 4.0 * s) * fine_factor / 255.0
        thr_hh = (0.20 + 6.0 * s) * fine_factor / 255.0

        def deadzone_quant(b: torch.Tensor, step: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
            mag = torch.relu(b.abs() - thr)
            b2 = b.sign() * mag
            return quantize_ste_unclamped(b2, step)

        lh_q = deadzone_quant(lh, step_lh, thr_lh)
        hl_q = deadzone_quant(hl, step_lh, thr_lh)
        hh_q = deadzone_quant(hh, step_hh, thr_hh)

        # Severe JPEG2000 removes fine texture but should avoid JPEG-style blocks.
        keep = 1.0 - 0.10 * s
        cur = haar_idwt2(cur, (lh_q * keep, hl_q * keep, hh_q * keep), shape)

    y = torch.clamp(cur, 0.0, 1.0)

    # Mild low-pass at severe settings to emulate wavelet quantization smoothing.
    smooth_mix = 0.02 + 0.10 * s
    smooth = avg_blur(y, 3)
    y = (1.0 - smooth_mix) * y + smooth_mix * smooth
    return torch.clamp(y, 0.0, 1.0)


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 0.5
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb
    return torch.cat([r, g, b], dim=1)


def chroma_downup_420(c: torch.Tensor, scale: int = 2) -> torch.Tensor:
    H, W = c.shape[-2:]
    scale = max(2, int(scale))
    h2 = max(1, math.ceil(H / scale))
    w2 = max(1, math.ceil(W / scale))
    z = F.interpolate(c, size=(h2, w2), mode="bilinear", align_corners=False)
    return F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix."""
    mat = torch.empty((n, n), device=device, dtype=dtype)
    factor = math.pi / float(n)
    for k in range(n):
        alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = alpha * math.cos((i + 0.5) * k * factor)
    return mat


def transform_quantize_blocks(
    x: torch.Tensor,
    strength: torch.Tensor,
    block: int = 4,
    base_step: float = 0.55,
    highfreq_gain: float = 3.0,
) -> torch.Tensor:
    """Block-transform quantization for signed residuals.

    This is a closer HEVC/BPG proxy than directly quantizing pixel residuals.
    HEVC intra prediction codes transform residuals; high-frequency transform
    coefficients are effectively more fragile at high QP.
    """
    B, C, H, W = x.shape
    n = int(block)
    pad_h = (n - H % n) % n
    pad_w = (n - W % n) % n
    mode = "reflect" if H > 1 and W > 1 else "replicate"
    xp = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    Hp, Wp = xp.shape[-2:]

    patches = xp.reshape(B, C, Hp // n, n, Wp // n, n)
    patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
    flat = patches.reshape(-1, n, n)

    D = _dct_matrix(n, x.device, x.dtype)
    coeff = D @ flat @ D.t()

    yy, xx = torch.meshgrid(
        torch.arange(n, device=x.device, dtype=x.dtype),
        torch.arange(n, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    freq = (xx + yy) / max(1.0, 2.0 * (n - 1))
    freq_weight = 1.0 + highfreq_gain * freq
    step = ((base_step + 22.0 * strength.view(1, 1, 1)) / 255.0) * freq_weight.view(1, n, n)

    coeff_q = quantize_ste_unclamped(coeff, step)
    rec = D.t() @ coeff_q @ D

    rec = rec.reshape(B, C, Hp // n, Wp // n, n, n)
    rec = rec.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, C, Hp, Wp)
    return rec[:, :, :H, :W]


def bpg_hevc_like_proxy_unused_strong(x: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
    """Stronger BPG/HEVC-intra-like differentiable proxy.

    Real BPG is HEVC intra coding. This proxy captures more of that structure:
        - RGB -> YCbCr
        - 4:2:0-like chroma down/up sampling
        - multiple cheap intra predictors for luma
        - transform-domain residual quantization with STE
        - QP-like exponential strength handled before this function
        - deblocking / SAO-like smoothing

    It is still not a real BPG codec, but it is better aligned than the old
    blur/block-average proxy.
    """
    s_tensor = severity.float().clamp(0.0, 1.0)
    s = float(s_tensor.detach().cpu().item())
    s4 = s_tensor.view(1, 1, 1, 1)

    ycc = rgb_to_ycbcr(x).clamp(0.0, 1.0)
    y = ycc[:, 0:1]
    cbcr = ycc[:, 1:3]

    # Approximate several HEVC intra prediction modes with differentiable cheap predictors.
    block = 4 if s < 0.50 else (8 if s < 0.82 else 16)
    planar = block_average(y, block=block)
    vertical = F.avg_pool2d(F.pad(y, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :], kernel_size=3, stride=1, padding=1)
    horizontal = F.avg_pool2d(F.pad(y, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1], kernel_size=3, stride=1, padding=1)
    smooth = avg_blur(y, 3 if s < 0.75 else 5)

    # More planar/smooth prediction at high severity.
    w_planar = 0.45 + 0.25 * s4
    w_smooth = 0.20 + 0.20 * s4
    w_dir = (1.0 - w_planar - w_smooth).clamp_min(0.05)
    pred = w_planar * planar + 0.5 * w_dir * vertical + 0.5 * w_dir * horizontal + w_smooth * smooth

    residual = y - pred

    # Transform residual quantization, closer to HEVC than pixel residual quantization.
    tblock = 4 if s < 0.75 else 8
    residual_q = transform_quantize_blocks(
        residual,
        strength=s_tensor.view(1),
        block=tblock,
        base_step=0.45,
        highfreq_gain=3.5,
    )
    y_rec = pred + residual_q

    # Deblocking/SAO-like smoothing; preserve edges by mixing with the residual reconstruction.
    deblock = 0.02 + 0.16 * s4
    y_smooth = avg_blur(y_rec, 3 if s < 0.80 else 5)
    y_rec = (1.0 - deblock) * y_rec + deblock * y_smooth

    # Chroma is often subsampled and coarsely quantized.
    chroma_scale = 2 if s < 0.78 else 4
    cbcr_du = chroma_downup_420(cbcr, scale=chroma_scale)
    chroma_mix = 0.50 + 0.35 * s4
    cbcr_rec = (1.0 - chroma_mix) * cbcr + chroma_mix * cbcr_du
    chroma_step = (1.2 + 30.0 * s4) / 255.0
    cbcr_rec = quantize_ste(cbcr_rec, chroma_step)

    ycc_rec = torch.cat([y_rec, cbcr_rec], dim=1).clamp(0.0, 1.0)

    # Weak CTU/local structure, not JPEG-like hard blocking.
    local = block_average(ycc_rec, block=4 if s < 0.85 else 8)
    local_mix = 0.01 + 0.07 * s4
    ycc_rec = (1.0 - local_mix) * ycc_rec + local_mix * local

    rgb = ycbcr_to_rgb(ycc_rec)
    return torch.clamp(rgb, 0.0, 1.0)



def bpg_hevc_like_proxy_medium(x: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
    """Medium-strength BPG/HEVC-intra-like differentiable proxy.

    This is a safer compromise between:
      - the old blur/block BPG proxy, which was stable but crude;
      - the strong YCbCr/intra-transform proxy, which made training too hard.

    It keeps the useful HEVC-like ingredients, but with weaker quantization,
    weaker deblocking, and less destructive chroma processing.
    """
    s_tensor = severity.float().clamp(0.0, 1.0)
    s = float(s_tensor.detach().cpu().item())
    s4 = s_tensor.view(1, 1, 1, 1)

    ycc = rgb_to_ycbcr(x).clamp(0.0, 1.0)
    y = ycc[:, 0:1]
    cbcr = ycc[:, 1:3]

    # Cheap intra-like prediction. Keep this weaker than the strong version.
    block = 4 if s < 0.70 else 8
    planar = block_average(y, block=block)
    smooth = avg_blur(y, 3 if s < 0.85 else 5)
    pred_mix = 0.18 + 0.32 * s4
    pred = (1.0 - pred_mix) * y + pred_mix * (0.70 * planar + 0.30 * smooth)

    residual = y - pred

    # Transform residual quantization, but much softer than the strong proxy.
    # This keeps BPG-like behavior without making the training collapse.
    tblock = 4
    residual_q = transform_quantize_blocks(
        residual,
        strength=s_tensor.view(1),
        block=tblock,
        base_step=0.22,
        highfreq_gain=2.0,
    )
    y_rec = pred + residual_q

    # Weak deblocking/smoothing.
    deblock = 0.01 + 0.08 * s4
    y_smooth = avg_blur(y_rec, 3)
    y_rec = (1.0 - deblock) * y_rec + deblock * y_smooth

    # Mild chroma down/up + quantization. Strong BPG destroys chroma more;
    # here we keep it milder to avoid hurting detector features too much.
    chroma_scale = 2
    cbcr_du = chroma_downup_420(cbcr, scale=chroma_scale)
    chroma_mix = 0.20 + 0.35 * s4
    cbcr_rec = (1.0 - chroma_mix) * cbcr + chroma_mix * cbcr_du
    chroma_step = (0.8 + 16.0 * s4) / 255.0
    cbcr_rec = quantize_ste(cbcr_rec, chroma_step)

    ycc_rec = torch.cat([y_rec, cbcr_rec], dim=1).clamp(0.0, 1.0)

    # Very weak local CTU-like structure.
    local = block_average(ycc_rec, block=4)
    local_mix = 0.005 + 0.035 * s4
    ycc_rec = (1.0 - local_mix) * ycc_rec + local_mix * local

    rgb = ycbcr_to_rgb(ycc_rec)
    return torch.clamp(rgb, 0.0, 1.0)


def _arg_float(obj, name: str, default: float) -> float:
    """Read a float CLI argument from argparse.Namespace-like object."""
    if obj is None:
        return float(default)
    return float(getattr(obj, name, default))


def _range_check(vmin: float, vmax: float, name: str) -> Tuple[float, float]:
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin:
        raise ValueError(f"{name}: max must be > min, got min={vmin}, max={vmax}")
    return vmin, vmax


def codec_proxy_single(
    x: torch.Tensor,
    family_name: str,
    severity: torch.Tensor,
    proxy_args: Optional[argparse.Namespace] = None,
) -> torch.Tensor:
    """
    Range-aware differentiable codec-family proxy.

    severity is always normalized:
        0 = mild / high quality / high bitrate
        1 = severe / low quality / low bitrate

    The proxy now interprets that normalized severity through each codec's
    parameter range:

        diff_jpeg:
            JPEG quality in [diffjpeg_quality_min, diffjpeg_quality_max]
            higher quality = less severe.
            In the main run this branch is replaced by real Diff-JPEG.

        jp2_like:
            OpenJPEG compression ratio -r in [jp2_rate_min, jp2_rate_max]
            higher rate/ratio = more severe.
            The proxy converts the implied rate to a nonlinear wavelet strength.

        bmshj:
            quality in [bmshj_quality_min, bmshj_quality_max]
            higher quality = less severe.
            In the main run this branch is replaced by real CompressAI BMSHJ.

        bpg_like:
            BPG QP in [bpg_qp_min, bpg_qp_max]
            higher QP = more severe.
            The proxy converts QP to an HEVC-like exponential quantization strength.
    """
    s_tensor = severity.float().clamp(0.0, 1.0)
    s = float(s_tensor.detach().cpu().item())
    s4 = s_tensor.view(1, 1, 1, 1)

    if family_name == "diff_jpeg":
        # Fallback only. Real Diff-JPEG is used when --use-real-diffjpeg is set.
        qmin, qmax = _range_check(
            _arg_float(proxy_args, "diffjpeg_quality_min", 10),
            min(_arg_float(proxy_args, "diffjpeg_quality_max", 99), 99),
            "Diff-JPEG quality range",
        )

        # severity 0 -> qmax, severity 1 -> qmin.
        q = qmax - s4 * (qmax - qmin)
        q_sev = ((qmax - q) / max(qmax - qmin, 1e-6)).clamp(0.0, 1.0)
        q_sev_scalar = float(q_sev.detach().cpu().item())

        q_step = (2.0 + 22.0 * q_sev) / 255.0
        block_mix = 0.05 + 0.40 * q_sev

        y = quantize_ste(x, q_step)
        blocky = block_average(y, block=8)
        y = (1.0 - block_mix) * y + block_mix * blocky

        if q_sev_scalar >= 0.50:
            smooth = avg_blur(y, k=3)
            blur_mix = 0.05 + 0.10 * q_sev
            y = (1.0 - blur_mix) * y + blur_mix * smooth

        return torch.clamp(y, 0.0, 1.0)

    if family_name == "jp2_like":
        rmin, rmax = _range_check(
            _arg_float(proxy_args, "jp2_rate_min", 1.0),
            _arg_float(proxy_args, "jp2_rate_max", 192.0),
            "JPEG2000 rate range",
        )

        # OpenJPEG -r is a compression ratio. Higher value = more severe.
        # severity 0 -> rmin, severity 1 -> rmax.
        r = rmin + s4 * (rmax - rmin)
        lin = ((r - rmin) / max(rmax - rmin, 1e-6)).clamp(0.0, 1.0)

        # Compression ratios are perceptually nonlinear. Make the wavelet proxy
        # react to the actual range rather than using a purely abstract 0..1.
        if rmin > 0 and rmax > rmin:
            log_r = torch.log(r.clamp_min(1e-6) / float(rmin)) / max(math.log(float(rmax) / float(rmin)), 1e-6)
            log_r = log_r.clamp(0.0, 1.0)
            jp2_strength = (0.45 * lin + 0.55 * log_r).clamp(0.0, 1.0)
        else:
            jp2_strength = torch.sqrt(lin.clamp_min(1e-8)).clamp(0.0, 1.0)

        return jp2_wavelet_proxy(x, severity=jp2_strength.view(-1)[0], levels=3)

    if family_name == "bmshj":
        # Fallback only. Real BMSHJ is used when --use-real-bmshj is set.
        qmin, qmax = _range_check(
            _arg_float(proxy_args, "bmshj_quality_min", 1),
            _arg_float(proxy_args, "bmshj_quality_max", 8),
            "BMSHJ quality range",
        )

        # severity 0 -> qmax, severity 1 -> qmin.
        q = qmax - s4 * (qmax - qmin)
        q_sev = ((qmax - q) / max(qmax - qmin, 1e-6)).clamp(0.0, 1.0)
        q_scalar = float(q_sev.detach().cpu().item())

        q_step = (1.0 + 12.0 * q_sev) / 255.0

        scale = 1.0 - 0.50 * (q_scalar ** 1.25)
        scale = max(0.45, min(1.0, scale))

        H, W = x.shape[-2:]
        h2 = max(16, int(round(H * scale)))
        w2 = max(16, int(round(W * scale)))

        z = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
        y = F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)

        smooth = avg_blur(y, k=3 if q_scalar < 0.70 else 5)
        smooth_mix = 0.03 + 0.22 * q_sev
        y = (1.0 - smooth_mix) * y + smooth_mix * smooth

        y = quantize_ste(y, q_step)
        return torch.clamp(y, 0.0, 1.0)

    if family_name == "bpg_like":
        qpmin, qpmax = _range_check(
            _arg_float(proxy_args, "bpg_qp_min", 28.0),
            _arg_float(proxy_args, "bpg_qp_max", 48.0),
            "BPG QP range",
        )
        qp = qpmin + s4 * (qpmax - qpmin)
        qp_sev = ((qp - qpmin) / max(qpmax - qpmin, 1e-6)).clamp(0.0, 1.0)
        qp_sev_scalar = float(qp_sev.detach().cpu().item())

        mode = str(getattr(proxy_args, "bpg_proxy_mode", "medium")).strip().lower()

        if mode == "old":
            # OLD stable BPG proxy: smoothing + local 4x4 block structure + STE quantization.
            q_step = (1.0 + 15.0 * qp_sev) / 255.0
            k = 3 if qp_sev_scalar < 0.50 else 5
            smooth = avg_blur(x, k=k)
            blocky = block_average(smooth, block=4)
            block_mix = 0.03 + 0.22 * qp_sev
            y = (1.0 - block_mix) * smooth + block_mix * blocky
            y = quantize_ste(y, q_step)
            return torch.clamp(y, 0.0, 1.0)

        if mode == "medium":
            # Safer better BPG proxy: HEVC-like, but much softer than the strong version.
            return bpg_hevc_like_proxy_medium(x, severity=qp_sev.view(-1)[0])

        if mode == "strong":
            # Strong HEVC-like proxy. This is available for ablation, but it may make training too hard.
            return bpg_hevc_like_proxy_unused_strong(x, severity=qp_sev.view(-1)[0])

        raise ValueError(f"Unknown --bpg-proxy-mode {mode!r}. Use old, medium, or strong.")

    raise KeyError(f"Unknown codec family: {family_name}")


def codec_proxy_batch(
    x: torch.Tensor,
    family_ids: torch.Tensor,
    severity_norm: torch.Tensor,
    id_to_codec_family: Dict[int, str],
    proxy_args: Optional[argparse.Namespace] = None,
) -> torch.Tensor:
    """Handmade proxy batch wrapper, passing codec min/max range args."""
    ys = []
    for i in range(x.shape[0]):
        family_id = int(family_ids[i].detach().cpu().item())
        family_name = id_to_codec_family[family_id]
        y = codec_proxy_single(
            x[i : i + 1],
            family_name=family_name,
            severity=severity_norm[i],
            proxy_args=proxy_args,
        )
        ys.append(y)
    return torch.cat(ys, dim=0)


def _import_diffjpeg_functions():
    """Import necla-ml/Diff-JPEG with a few common module layouts.

    The expected public API is:
        diff_jpeg_coding(image_rgb=image, jpeg_quality=quality, ste=True)
        DiffJPEGCoding(ste=True)
    """
    errors = []
    candidates = [
        "diff_jpeg",
        "diff_jpeg.diff_jpeg",
        "diff_jpeg.jpeg",
        "diff_jpeg.codec",
        "DiffJPEG",
    ]
    for module_name in candidates:
        try:
            mod = __import__(module_name, fromlist=["diff_jpeg_coding", "DiffJPEGCoding"])
            fn = getattr(mod, "diff_jpeg_coding", None)
            cls = getattr(mod, "DiffJPEGCoding", None)
            if fn is not None or cls is not None:
                return fn, cls, module_name
        except Exception as e:
            errors.append(f"{module_name}: {e}")

    raise ImportError(
        "Could not import necla-ml/Diff-JPEG. Install it with:\\n"
        "  python -m pip install git+https://github.com/necla-ml/Diff-JPEG.git\\n"
        "Tried module names:\\n  " + "\\n  ".join(errors)
    )


class RealCodecProxyManager(nn.Module):
    """Codec-family proxy manager.

    Families:
      - diff_jpeg: optionally uses necla-ml/Diff-JPEG with STE.
      - bmshj: optionally uses real CompressAI BMSHJ forward and likelihood bpp.
      - jp2_like / bpg_like: use handmade differentiable proxies.

    Severity convention:
      severity=0 -> mild / high quality
      severity=1 -> severe / low quality

    Diff-JPEG mapping:
      severity=0 -> JPEG quality diffjpeg_quality_max, default 99
      severity=1 -> JPEG quality diffjpeg_quality_min, default 10
      Note: necla-ml/Diff-JPEG range tops at 99, not 100.

    BMSHJ mapping:
      severity=0 -> quality bmshj_quality_max, default 8
      severity=1 -> quality bmshj_quality_min, default 1
    """

    def __init__(self, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        self.use_real_diffjpeg = bool(args.use_real_diffjpeg)
        self.use_real_bmshj = bool(args.use_real_bmshj)

        self.diff_jpeg_fn = None
        self.diff_jpeg_module = None

        if self.use_real_diffjpeg:
            fn, cls, module_name = _import_diffjpeg_functions()
            self.diff_jpeg_fn = fn
            if cls is not None:
                try:
                    self.diff_jpeg_module = cls(ste=True)
                    if isinstance(self.diff_jpeg_module, nn.Module):
                        self.diff_jpeg_module = self.diff_jpeg_module.to(device)
                except Exception as e:
                    print(f"[WARN] Could not instantiate DiffJPEGCoding(ste=True): {e}")
                    self.diff_jpeg_module = None

            print("Real Diff-JPEG proxy enabled:")
            print(f"  import module: {module_name}")
            print(f"  quality range: [{args.diffjpeg_quality_min}, {args.diffjpeg_quality_max}]")
            print("  severity 0 -> high quality, severity 1 -> low quality")
            print("  STE: True")

        self.bmshj_codecs = nn.ModuleDict()
        if self.use_real_bmshj:
            try:
                from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior
            except Exception as e:
                raise ImportError(
                    "Could not import CompressAI BMSHJ models. Install CompressAI, e.g.\\n"
                    "  python -m pip install compressai\\n"
                    "Use a version compatible with your PyTorch/CUDA environment."
                ) from e

            model_name = str(args.bmshj_model).strip().lower()
            if model_name in ["bmshj2018-hyperprior", "bmshj2018_hyperprior", "hyperprior"]:
                make_codec = bmshj2018_hyperprior
            elif model_name in ["bmshj2018-factorized", "bmshj2018_factorized", "factorized"]:
                make_codec = bmshj2018_factorized
            else:
                raise ValueError(
                    f"Unknown --bmshj-model {args.bmshj_model!r}. "
                    "Use bmshj2018-hyperprior or bmshj2018-factorized."
                )

            qmin = int(args.bmshj_quality_min)
            qmax = int(args.bmshj_quality_max)
            if qmin > qmax:
                raise ValueError("--bmshj-quality-min must be <= --bmshj-quality-max")

            print("Real CompressAI BMSHJ proxy enabled:")
            print(f"  model: {args.bmshj_model}")
            print(f"  qualities loaded: {list(range(qmin, qmax + 1))}")
            print(f"  bpp scale: {args.bmshj_bpp_scale}")
            print(f"  mode: {'eval' if args.bmshj_eval_mode else 'train/noisy-quant'}")

            for q in range(qmin, qmax + 1):
                codec = make_codec(quality=q, pretrained=True).to(device)
                for p in codec.parameters():
                    p.requires_grad_(False)

                # For training the pre-editor, train mode gives differentiable
                # noisy quantization in CompressAI. eval mode is available if
                # you explicitly want deterministic behavior, but gradients
                # through hard rounding can be weaker.
                if args.bmshj_eval_mode:
                    codec.eval()
                else:
                    codec.train()

                # update() is mainly required for arithmetic coding/compress(),
                # but harmless when available.
                try:
                    codec.update(force=True)
                except Exception:
                    pass

                self.bmshj_codecs[str(q)] = codec

    @staticmethod
    def _scalar_severity(severity: torch.Tensor) -> float:
        return float(severity.detach().float().clamp(0.0, 1.0).cpu().item())

    def jpeg_quality_from_severity(self, severity: torch.Tensor) -> int:
        s = self._scalar_severity(severity)
        qmin = int(self.args.diffjpeg_quality_min)
        qmax = int(self.args.diffjpeg_quality_max)
        # Diff-JPEG range tops at 99, not 100.
        qmax = min(qmax, 99)
        q = int(round(qmax - s * (qmax - qmin)))
        return max(qmin, min(qmax, q))

    def bmshj_quality_from_severity(self, severity: torch.Tensor) -> int:
        s = self._scalar_severity(severity)
        qmin = int(self.args.bmshj_quality_min)
        qmax = int(self.args.bmshj_quality_max)
        q = int(round(qmax - s * (qmax - qmin)))
        return max(qmin, min(qmax, q))

    def _call_diffjpeg(self, x: torch.Tensor, jpeg_quality: int) -> torch.Tensor:
        # necla-ml/Diff-JPEG requires jpeg_quality to be a torch.Tensor
        # with one quality value per image in the batch: shape [B].
        # Our manager usually calls this with B=1, but keep it generic.
        q_tensor = torch.full(
            (x.shape[0],),
            float(jpeg_quality),
            device=x.device,
            dtype=x.dtype,
        )

        # Use the module if possible, otherwise the functional API.
        if self.diff_jpeg_module is not None:
            mod = self.diff_jpeg_module
            try:
                y = mod(image_rgb=x, jpeg_quality=q_tensor)
            except TypeError:
                try:
                    y = mod(x, q_tensor)
                except TypeError:
                    y = mod(image_rgb=x, jpeg_quality=q_tensor, ste=True)
            return torch.clamp(y, 0.0, 1.0)

        if self.diff_jpeg_fn is None:
            raise RuntimeError("Real Diff-JPEG was requested but no callable was imported.")

        try:
            y = self.diff_jpeg_fn(image_rgb=x, jpeg_quality=q_tensor, ste=True)
        except TypeError:
            y = self.diff_jpeg_fn(x, q_tensor, ste=True)

        return torch.clamp(y, 0.0, 1.0)

    def _call_bmshj(self, x: torch.Tensor, quality: int) -> Tuple[torch.Tensor, torch.Tensor]:
        codec = self.bmshj_codecs[str(int(quality))]

        # CompressAI likelihoods should be computed in fp32 for stability.
        # This also avoids AMP issues in old torch/compressai environments.
        amp_off = (
            torch.amp.autocast("cuda", enabled=False)
            if x.is_cuda
            else nullcontext()
        )

        with amp_off:
            x32 = x.float().clamp(0.0, 1.0)
            out = codec(x32)

            if not isinstance(out, dict) or "x_hat" not in out or "likelihoods" not in out:
                raise RuntimeError(
                    "CompressAI BMSHJ forward did not return expected keys "
                    "['x_hat', 'likelihoods']."
                )

            x_hat = torch.clamp(out["x_hat"], 0.0, 1.0)

            num_pixels = x32.shape[0] * x32.shape[-2] * x32.shape[-1]
            bpp_terms = []
            for likelihood in out["likelihoods"].values():
                likelihood = likelihood.float().clamp_min(1e-9)
                bpp_terms.append(torch.log(likelihood).sum() / (-math.log(2.0) * num_pixels))
            bpp = torch.stack(bpp_terms).sum()

        return x_hat.to(dtype=x.dtype), bpp.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        family_ids: torch.Tensor,
        severity_norm: torch.Tensor,
        id_to_codec_family: Dict[int, str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys: List[torch.Tensor] = []
        bpps: List[torch.Tensor] = []

        for i in range(x.shape[0]):
            family_id = int(family_ids[i].detach().cpu().item())
            family_name = id_to_codec_family[family_id]
            xi = x[i : i + 1]
            sev_i = severity_norm[i]

            if family_name == "diff_jpeg" and self.use_real_diffjpeg:
                q = self.jpeg_quality_from_severity(sev_i)
                yi = self._call_diffjpeg(xi, q)
                bpp_i = xi.new_tensor(0.0)

            elif family_name == "bmshj" and self.use_real_bmshj:
                q = self.bmshj_quality_from_severity(sev_i)
                yi, bpp_i = self._call_bmshj(xi, q)

            else:
                yi = codec_proxy_single(xi, family_name=family_name, severity=sev_i, proxy_args=self.args)
                bpp_i = xi.new_tensor(0.0)

            ys.append(yi)
            bpps.append(bpp_i)

        return torch.cat(ys, dim=0), torch.stack(bpps).mean()


# =============================================================================
# Loss helpers
# =============================================================================


def soften_mask(mask: torch.Tensor, blur_kernel: int = 31) -> torch.Tensor:
    """
    mask: [B, 1, H, W] in {0, 1}
    """
    if blur_kernel <= 1:
        return mask.float().clamp(0.0, 1.0)

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Slight dilation before blur.
    dil_k = max(3, blur_kernel // 4)
    if dil_k % 2 == 0:
        dil_k += 1

    m = F.max_pool2d(mask.float(), kernel_size=dil_k, stride=1, padding=dil_k // 2)
    m = F.avg_pool2d(m, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
    return m.clamp(0.0, 1.0)


def weighted_mean(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    while weight.ndim < x.ndim:
        weight = weight.expand(-1, x.shape[1], -1, -1)
    return (x * weight).sum() / (weight.sum() * x.shape[1] + eps)


def weighted_l1(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return weighted_mean((a - b).abs(), weight)


def tv_loss(x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()

    if weight is None:
        return dx.mean() + dy.mean()

    wx = weight[:, :, :, 1:]
    wy = weight[:, :, 1:, :]
    return weighted_mean(dx, wx) + weighted_mean(dy, wy)


def reflect_avg_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    xp = F.pad(x, (pad, pad, pad, pad), mode=mode)
    return F.avg_pool2d(xp, kernel_size=k, stride=1)


def background_simplified_target(
    x: torch.Tensor,
    mode: str,
    blur_kernel: int,
    lowres_scale: int,
) -> torch.Tensor:
    """Compressible background target.

    This is intentionally NOT a clean-background reconstruction target.
    It is a low-frequency/flat target used only when lambda_bg_flat > 0.
    """
    mode = str(mode)

    if mode == "blur":
        return reflect_avg_blur(x, blur_kernel)

    if mode == "lowres":
        B, C, H, W = x.shape
        scale = max(2, int(lowres_scale))
        h2 = max(4, H // scale)
        w2 = max(4, W // scale)
        z = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
        return F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)

    if mode == "mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x)

    if mode == "zero":
        return torch.zeros_like(x)

    raise ValueError(
        f"Invalid bg_flat_mode={mode!r}. "
        "Expected one of: blur, lowres, mean, zero."
    )


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def attention_loss(attention: torch.Tensor, mask_soft: torch.Tensor) -> torch.Tensor:
    """
    AMP-safe attention supervision.

    PyTorch does not allow F.binary_cross_entropy() under autocast when the
    input is already sigmoid output. Since this model returns attention after
    sigmoid, we compute BCE manually in float32.

    attention:
        sigmoid probability map in [0, 1]

    mask_soft:
        soft object mask in [0, 1]
    """
    target = mask_soft.float().clamp(0.0, 1.0)
    pred = attention.float().clamp(1e-6, 1.0 - 1e-6)

    bce = -(
        target * torch.log(pred)
        + (1.0 - target) * torch.log1p(-pred)
    ).mean()

    dsc = dice_loss(pred, target)
    return bce + dsc


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3) / 8.0

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3) / 8.0

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    return torch.sqrt(gx * gx + gy * gy + 1e-6)


class FrozenYOLOFeatureConsistency(nn.Module):
    """
    Frozen Ultralytics YOLO feature consistency loss using forward hooks.

    Teacher:
        detector features from clean image

    Student:
        detector features from compressed(preedited image)

    The YOLO detector is frozen.
    Gradients flow through the student detector forward into the degraded image,
    then into the pre-editor.

    Important for this project:
        feature loss is applied only on object regions using the GT-derived
        training mask, downsampled to each hooked feature-map resolution.

    This lets:
        objects     -> preserve YOLO-relevant features after compression
        background  -> become smoother / simpler / more compressible
    """

    def __init__(
        self,
        weights: str,
        device: torch.device,
        hook_layers: Optional[List[int]] = None,
        loss_type: str = "smooth_l1",
        normalize: bool = True,
        print_layers: bool = False,
    ):
        super().__init__()

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "Could not import ultralytics. Install ultralytics or set --lambda-det 0."
            ) from e

        yolo = YOLO(weights)

        # Ultralytics wrapper has .model, which is the actual torch module.
        self.model = yolo.model.to(device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.loss_type = str(loss_type)
        self.normalize = bool(normalize)

        if not hasattr(self.model, "model"):
            raise RuntimeError(
                "Unexpected Ultralytics YOLO model structure: model has no attribute 'model'."
            )

        # In Ultralytics, model.model is usually a ModuleList/Sequential.
        self.layers = self.model.model

        if print_layers:
            print("Ultralytics YOLO layers:")
            for i, layer in enumerate(self.layers):
                f = getattr(layer, "f", None)
                print(f"  {i:03d}: {layer.__class__.__name__}, f={f} -> {layer}")

        if hook_layers is None:
            hook_layers = self._default_hook_layers()

        self.hook_layers = list(hook_layers)
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks = []

        for idx in self.hook_layers:
            if idx < 0 or idx >= len(self.layers):
                raise ValueError(
                    f"Invalid hook layer index {idx}. "
                    f"Model has {len(self.layers)} layers."
                )

            handle = self.layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

        print(f"YOLO feature hooks registered on layers: {self.hook_layers}")

    def _default_hook_layers(self) -> List[int]:
        """
        Choose safe mid/high-level layers automatically.

        Better:
            inspect with --print-detector-layers and set --detector-hook-layers
            to the layers feeding the final Detect head, e.g. Detect.f=[15,18,21].
        """
        n = len(self.layers)

        # Prefer feature-pyramid / neck-ish layers, not early texture layers.
        candidates = [
            int(0.45 * n),
            int(0.65 * n),
            int(0.85 * n),
        ]

        out = []
        for idx in candidates:
            idx = max(0, min(n - 1, idx))
            if idx not in out:
                out.append(idx)

        return out

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            feat = None

            if torch.is_tensor(output):
                feat = output
            elif isinstance(output, (list, tuple)):
                tensor_outputs = [
                    o for o in output
                    if torch.is_tensor(o) and o.is_floating_point()
                ]
                if len(tensor_outputs) > 0:
                    feat = tensor_outputs[0]
            elif isinstance(output, dict):
                tensor_outputs = [
                    o for o in output.values()
                    if torch.is_tensor(o) and o.is_floating_point()
                ]
                if len(tensor_outputs) > 0:
                    feat = tensor_outputs[0]

            # Only use standard image feature maps [B, C, H, W].
            if (
                feat is not None
                and feat.is_floating_point()
                and feat.ndim == 4
            ):
                self._features[idx] = feat

        return hook

    def clear_features(self) -> None:
        self._features = {}

    def extract_features(
        self,
        x: torch.Tensor,
        detach: bool,
    ) -> Dict[int, torch.Tensor]:
        self.clear_features()

        if detach:
            with torch.no_grad():
                _ = self.model(x)
            return {k: v.detach() for k, v in self._features.items()}

        _ = self.model(x)
        return dict(self._features)

    @staticmethod
    def _resize_object_mask(
        object_mask: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        object_mask:
            [B, 1, H_img, W_img], soft object mask in [0, 1]

        feature:
            [B, C, H_feat, W_feat]

        returns:
            [B, 1, H_feat, W_feat]
        """
        if object_mask.ndim != 4:
            raise ValueError(f"Expected object_mask [B,1,H,W], got {object_mask.shape}")

        if object_mask.shape[0] != feature.shape[0]:
            raise ValueError(
                f"Mask batch {object_mask.shape[0]} != feature batch {feature.shape[0]}"
            )

        m = F.interpolate(
            object_mask.float(),
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return m.clamp(0.0, 1.0)

    def _masked_tensor_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes detector feature loss only on object-mask regions.
        """
        s = student.float()
        t = teacher.detach().float()

        if self.normalize:
            scale = t.detach().abs().mean().clamp_min(1e-3)
            s = s / scale
            t = t / scale

        if self.loss_type == "l1":
            loss_map = (s - t).abs()
        elif self.loss_type == "mse":
            loss_map = (s - t).pow(2)
        elif self.loss_type == "smooth_l1":
            loss_map = F.smooth_l1_loss(s, t, beta=0.5, reduction="none")
        else:
            raise ValueError(f"Unknown detector feature loss type: {self.loss_type}")

        # object_mask: [B,1,H,W], loss_map: [B,C,H,W]
        denom = object_mask.sum() * loss_map.shape[1]
        if float(denom.detach().cpu().item()) < 1e-6:
            return loss_map.new_tensor(0.0)

        return (loss_map * object_mask).sum() / denom.clamp_min(1e-6)

    def feature_loss(
        self,
        student_feats: Dict[int, torch.Tensor],
        teacher_feats: Dict[int, torch.Tensor],
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = []

        for idx in self.hook_layers:
            if idx not in student_feats or idx not in teacher_feats:
                continue

            s = student_feats[idx]
            t = teacher_feats[idx].detach()

            # Only compare normal feature maps.
            if s.ndim != 4 or t.ndim != 4:
                continue

            if s.shape != t.shape:
                if s.shape[0] == t.shape[0] and s.shape[1] == t.shape[1]:
                    t = F.interpolate(
                        t,
                        size=s.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    continue

            mask_f = self._resize_object_mask(object_mask, s)
            loss = self._masked_tensor_loss(s, t, mask_f)
            losses.append(loss)

        if len(losses) == 0:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).mean()

    def forward(
        self,
        clean: torch.Tensor,
        degraded: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        clean:
            original image in [0, 1], [B, 3, H, W]

        degraded:
            compressed(preedited image) in [0, 1], [B, 3, H, W]

        object_mask:
            soft GT-derived object mask in [0, 1], [B, 1, H, W]
            Used only during training.
        """
        teacher_feats = self.extract_features(clean, detach=True)
        student_feats = self.extract_features(degraded, detach=False)

        return self.feature_loss(
            student_feats=student_feats,
            teacher_feats=teacher_feats,
            object_mask=object_mask,
        )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class FrozenDetectron2FPNConsistency(nn.Module):
    """Frozen Detectron2 FPN feature distillation loss.

    This is the closest differentiable surrogate to the paper's Det setup:
        clean image -> Faster R-CNN X101-FPN -> FPN features P2..P6
        degraded image -> same frozen detector -> FPN features P2..P6

    It does not compute discrete mAP. Instead, it preserves the detector's
    multi-scale features after compression, which is differentiable and much
    better aligned with COCO Det AP than a generic image loss.

    Inputs are RGB tensors in [0, 1]. Detectron2 models expect BGR 0..255.
    """

    def __init__(
        self,
        config_file: str,
        device: torch.device,
        weights: str = "",
        feature_names: Optional[List[str]] = None,
        loss_type: str = "smooth_l1",
        normalize: bool = True,
        mask_weight: float = 0.7,
        full_weight: float = 0.3,
        input_format: str = "BGR",
    ):
        super().__init__()

        try:
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.modeling import build_model
        except Exception as e:
            raise RuntimeError(
                "Could not import detectron2. Install detectron2 or use "
                "--detector-backend yolo / --detector-backend none."
            ) from e

        cfg = get_cfg()

        # Accept either a model-zoo config name or a local config path.
        cfg_path = str(config_file)
        if Path(cfg_path).exists():
            cfg.merge_from_file(cfg_path)
        else:
            cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

        if weights:
            cfg.MODEL.WEIGHTS = str(weights)
        else:
            # IMPORTANT:
            # Detectron2 model-zoo configs often contain an ImageNet-pretrained
            # backbone path in cfg.MODEL.WEIGHTS, e.g.
            #   detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl
            # If we keep that value, the FPN/RPN/ROI-head weights are random.
            # For FPN feature distillation we need the actual COCO detector
            # checkpoint, not only the ImageNet backbone.
            if Path(cfg_path).exists():
                if not str(cfg.MODEL.WEIGHTS):
                    raise ValueError(
                        "Local Detectron2 config was provided without --detectron2-weights, "
                        "and cfg.MODEL.WEIGHTS is empty. Please pass --detectron2-weights."
                    )
            else:
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        cfg.MODEL.DEVICE = str(device)
        cfg.freeze()

        self.cfg = cfg
        self.model = build_model(cfg).to(device).eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.loss_type = str(loss_type)
        self.normalize = bool(normalize)
        self.mask_weight = float(mask_weight)
        self.full_weight = float(full_weight)
        self.input_format = str(input_format).upper()

        if feature_names is None or len(feature_names) == 0:
            feature_names = ["p2", "p3", "p4", "p5", "p6"]
        self.feature_names = list(feature_names)

        print("Detectron2 FPN feature consistency loaded:")
        print(f"  config: {config_file}")
        print(f"  weights: {cfg.MODEL.WEIGHTS}")
        print(f"  features: {self.feature_names}")
        print(f"  mask_weight={self.mask_weight}, full_weight={self.full_weight}")

    def _rgb01_to_detectron_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB [0,1] batch to Detectron2 image tensor before normalization.

        Detectron2's GeneralizedRCNN.preprocess_image uses ImageList.from_tensors,
        which performs an in-place copy into a padded tensor. With some old
        Detectron2 + newer PyTorch combinations, that can fail when gradients
        must flow to the input image:
            RuntimeError: A view was created in no_grad mode and is being
            modified inplace with grad mode enabled.

        We avoid that path and implement the equivalent preprocessing with
        differentiable out-of-place tensor ops.
        """
        if self.input_format == "BGR":
            return x[:, [2, 1, 0], :, :] * 255.0
        if self.input_format == "RGB":
            return x * 255.0
        raise ValueError(f"Invalid input_format={self.input_format!r}. Use BGR or RGB.")

    def _preprocess_tensor_differentiable(self, x: torch.Tensor) -> torch.Tensor:
        """Detectron2-style normalization/padding without ImageList in-place ops."""
        y = self._rgb01_to_detectron_tensor(x)

        pixel_mean = self.model.pixel_mean.to(device=y.device, dtype=y.dtype)
        pixel_std = self.model.pixel_std.to(device=y.device, dtype=y.dtype)
        y = (y - pixel_mean) / pixel_std

        stride = int(getattr(self.model.backbone, "size_divisibility", 0) or 0)
        if stride > 1:
            h, w = y.shape[-2:]
            pad_h = (stride - (h % stride)) % stride
            pad_w = (stride - (w % stride)) % stride
            if pad_h > 0 or pad_w > 0:
                # F.pad is out-of-place and differentiable w.r.t. y.
                y = F.pad(y, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        return y

    def extract_features(self, x: torch.Tensor, detach: bool) -> Dict[str, torch.Tensor]:
        if detach:
            with torch.no_grad():
                images_tensor = self._preprocess_tensor_differentiable(x)
                feats = self.model.backbone(images_tensor)
            return {k: v.detach() for k, v in feats.items() if torch.is_tensor(v)}

        images_tensor = self._preprocess_tensor_differentiable(x)
        feats = self.model.backbone(images_tensor)
        return {k: v for k, v in feats.items() if torch.is_tensor(v)}

    @staticmethod
    def _resize_mask(mask: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            mask.float(),
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

    def _loss_map(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = student.float()
        t = teacher.detach().float()

        if self.normalize:
            scale = t.detach().abs().mean().clamp_min(1e-3)
            s = s / scale
            t = t / scale

        if self.loss_type == "l1":
            return (s - t).abs()
        if self.loss_type == "mse":
            return (s - t).pow(2)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(s, t, beta=0.5, reduction="none")
        raise ValueError(f"Unknown detector feature loss type: {self.loss_type}")

    def _masked_mean(self, loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # loss_map [B,C,H,W], mask [B,1,H,W]
        denom = mask.sum() * loss_map.shape[1]
        if float(denom.detach().cpu().item()) < 1e-6:
            return loss_map.new_tensor(0.0)
        return (loss_map * mask).sum() / denom.clamp_min(1e-6)

    def forward(
        self,
        clean: torch.Tensor,
        degraded: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        teacher_feats = self.extract_features(clean, detach=True)
        student_feats = self.extract_features(degraded, detach=False)

        losses = []
        for name in self.feature_names:
            if name not in student_feats or name not in teacher_feats:
                continue

            s = student_feats[name]
            t = teacher_feats[name].detach()

            if s.ndim != 4 or t.ndim != 4:
                continue

            # Detectron2 padding should make shapes match. Keep this for safety.
            if s.shape != t.shape:
                if s.shape[0] == t.shape[0] and s.shape[1] == t.shape[1]:
                    t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    continue

            lm = self._loss_map(s, t)

            parts = []
            weights = []

            if self.mask_weight > 0:
                mf = self._resize_mask(object_mask, s)
                parts.append(self._masked_mean(lm, mf))
                weights.append(self.mask_weight)

            if self.full_weight > 0:
                parts.append(lm.mean())
                weights.append(self.full_weight)

            if parts:
                wsum = max(sum(weights), 1e-8)
                losses.append(sum(w * p for w, p in zip(weights, parts)) / wsum)

        if not losses:
            return degraded.new_tensor(0.0)

        return torch.stack(losses).mean()



# =============================================================================
# Train / validation
# =============================================================================


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
    # -------------------------------------------------------------------------
    # Annotation-box masks
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
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

        # Edited compression must keep gradients to the pre-editor.
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
        comp_l1_preedit = (compressed_edited - images).abs().mean()

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
        "comp_edit_l1": comp_l1_preedit.detach(),
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

    print(f"Loading initial pre-editor checkpoint: {ckpt_path}")
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

    model = ConditionalAttentionPreEditor(
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


# =============================================================================
# Utils / CLI
# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone conditional family/severity pre-editor trainer."
    )

    # Data
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--annotations-json", type=str, required=True)
    p.add_argument("--val-images-dir", type=str, default="")
    p.add_argument("--val-annotations-json", type=str, default="")

    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--object-crop-prob", type=float, default=0.60)
    p.add_argument("--crop-margin", type=float, default=4.0)

    # Conditioning
    p.add_argument(
        "--codec-families",
        type=str,
        default="diff_jpeg,bmshj,bpg_like",
        help="Comma-separated codec families.",
    )
    p.add_argument(
        "--severity-min",
        type=float,
        default=0.0,
        help="Raw severity value mapped to normalized 0.",
    )
    p.add_argument(
        "--severity-max",
        type=float,
        default=1.0,
        help="Raw severity value mapped to normalized 1.",
    )
    p.add_argument(
        "--severity-levels",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help=(
            "Comma-separated NORMALIZED severity levels in [0, 1] to sample. "
            "0 = mild degradation / high bitrate, 1 = severe degradation / low bitrate. "
            "Use empty string for continuous uniform sampling."
        ),
    )
    p.add_argument(
        "--val-severity-levels",
        type=str,
        default="",
        help="Optional validation raw severity levels.",
    )
    p.add_argument(
        "--severity-probs",
        type=str,
        default="",
        help=(
            "Optional comma-separated sampling probabilities aligned with --severity-levels. "
            "Example for 0,0.2,0.4,0.6,0.8,1.0: 0.03,0.07,0.15,0.20,0.25,0.30"
        ),
    )
    p.add_argument(
        "--codec-family-probs",
        type=str,
        default="",
        help=(
            "Optional family sampling probabilities. Either aligned with --codec-families "
            "or named, e.g. diff_jpeg:0.34,bmshj:0.33,bpg_like:0.33"
        ),
    )
    p.add_argument(
        "--severity-levels-by-family",
        type=str,
        default="",
        help=(
            "Optional family-specific severity grids. Format: "
            "diff_jpeg:0,0.5,1;bmshj:0,0.142857,1;bpg_like:0,0.1,1. "
            "If set, it overrides the global --severity-levels during training."
        ),
    )
    p.add_argument(
        "--severity-probs-by-family",
        type=str,
        default="",
        help=(
            "Optional family-specific sampling probabilities aligned with "
            "--severity-levels-by-family. Missing family = uniform for that family."
        ),
    )
    p.add_argument(
        "--val-severity-levels-by-family",
        type=str,
        default="",
        help=(
            "Optional family-specific validation severity grids, same format as "
            "--severity-levels-by-family."
        ),
    )
    p.add_argument(
        "--invert-severity",
        action="store_true",
        help="Use if raw value is quality where larger means less severe.",
    )

    # Real codec-family proxies.
    p.add_argument(
        "--use-real-diffjpeg",
        action="store_true",
        help=(
            "Use necla-ml/Diff-JPEG with STE for the diff_jpeg family. "
            "If disabled, use the internal JPEG-like proxy."
        ),
    )
    p.add_argument(
        "--diffjpeg-quality-min",
        type=int,
        default=10,
        help="JPEG quality at severity=1. Diff-JPEG valid range should be <= 99.",
    )
    p.add_argument(
        "--diffjpeg-quality-max",
        type=int,
        default=99,
        help="JPEG quality at severity=0. necla-ml/Diff-JPEG tops at 99, not 100.",
    )
    p.add_argument(
        "--use-real-bmshj",
        action="store_true",
        help=(
            "Use real CompressAI BMSHJ forward for the bmshj family and "
            "likelihood bpp as rate term."
        ),
    )
    p.add_argument(
        "--bmshj-model",
        type=str,
        default="bmshj2018-hyperprior",
        choices=["bmshj2018-hyperprior", "bmshj2018-factorized"],
    )
    p.add_argument("--bmshj-quality-min", type=int, default=1)
    p.add_argument("--bmshj-quality-max", type=int, default=8)
    p.add_argument(
        "--bmshj-eval-mode",
        action="store_true",
        help=(
            "Use BMSHJ codec in eval mode. Default is train/noisy-quant mode, "
            "which usually gives better gradients for pre-editor training."
        ),
    )
    p.add_argument(
        "--bmshj-bpp-scale",
        type=float,
        default=0.03,
        help="Scale for CompressAI likelihood bpp before adding to rate loss.",
    )
    p.add_argument(
        "--rate-tv-scale",
        type=float,
        default=1.0,
        help="Scale for TV/high-frequency rate proxy. BMSHJ bpp is added separately.",
    )

    # Real/proxy codec parameter ranges used by the handcrafted proxies.
    # Manual eval severity can still be used; these ranges make training know
    # what severity=0 and severity=1 mean for each real codec family.
    p.add_argument("--jp2-rate-min", type=float, default=1.0)
    p.add_argument("--jp2-rate-max", type=float, default=192.0)
    p.add_argument("--bpg-qp-min", type=float, default=0.0)
    p.add_argument("--bpg-qp-max", type=float, default=51.0)
    p.add_argument(
        "--bpg-proxy-mode",
        type=str,
        default="medium",
        choices=["old", "medium", "strong"],
        help=(
            "BPG-like differentiable proxy variant: "
            "old=smooth/block proxy, medium=stable YCbCr/intra/transform proxy, "
            "strong=more aggressive experimental proxy."
        ),
    )

    # Model
    p.add_argument("--base-ch", type=int, default=48)
    p.add_argument("--family-emb-dim", type=int, default=8)
    p.add_argument("--cond-dim", type=int, default=64)
    p.add_argument("--max-delta-obj", type=float, default=0.09)
    p.add_argument("--max-delta-bg", type=float, default=0.03)
    p.add_argument("--attention-bias", type=float, default=-4.0)
    p.add_argument(
        "--bg-mode",
        type=str,
        default="blur_replace",
        choices=["residual", "blur_replace", "lowres_replace", "mean_replace"],
        help=(
            "Background composition mode. "
            "residual = old x + delta_bg behavior. "
            "blur_replace/lowres_replace/mean_replace = background branch replaces "
            "background with a cheaper low-frequency/flat version plus residual."
        ),
    )
    p.add_argument(
        "--bg-blur-kernel",
        type=int,
        default=51,
        help="Kernel used by blur_replace and bg_flat_mode=blur.",
    )
    p.add_argument(
        "--bg-lowres-scale",
        type=int,
        default=16,
        help="Downsampling factor used by lowres_replace and bg_flat_mode=lowres.",
    )

    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint used to initialize/fine-tune the pre-editor model weights.",
    )

    # Loss weights
    p.add_argument(
        "--attn-blur",
        type=int,
        default=7,
        help=(
            "Blur/dilation size for attention target built from annotation boxes. "
            "Use a small value to keep attention tight around GT boxes."
        ),
    )
    p.add_argument(
        "--det-mask-blur",
        type=int,
        default=31,
        help=(
            "Blur/dilation size for detector feature mask built from annotation boxes. "
            "Usually larger than attn-blur because YOLO features need object context."
        ),
    )
    p.add_argument("--attn-area-mult", type=float, default=1.0)

    p.add_argument("--lambda-obj-l1", type=float, default=4.0)
    p.add_argument("--lambda-obj-edge", type=float, default=3.0)
    p.add_argument("--lambda-bg-l1", type=float, default=0.0)
    p.add_argument(
        "--lambda-bg-flat",
        type=float,
        default=0.75,
        help=(
            "Weight for forcing background toward a cheap low-frequency/flat target. "
            "Unlike lambda-bg-l1, this does not preserve clean background appearance."
        ),
    )
    p.add_argument(
        "--bg-flat-mode",
        type=str,
        default="blur",
        choices=["blur", "lowres", "mean", "zero"],
        help="Cheap target used when lambda-bg-flat > 0.",
    )
    p.add_argument("--lambda-bg-smooth", type=float, default=1.00)
    p.add_argument("--lambda-rate", type=float, default=2.00)
    p.add_argument("--lambda-attn", type=float, default=1.00)
    p.add_argument("--lambda-attn-sparse", type=float, default=0.75)
    p.add_argument("--lambda-attn-area", type=float, default=2.00)
    p.add_argument("--lambda-delta", type=float, default=0.02)
    p.add_argument("--lambda-delta-tv", type=float, default=0.05)

    # Severity-dependent dynamic loss weights.
    # At high severity, task/object/FPN loss is upweighted and rate/background
    # simplification can be downweighted. This is intended to improve q1/q2 AP.
    p.add_argument("--task-severity-gain", type=float, default=0.0)
    p.add_argument("--rate-severity-drop", type=float, default=0.0)
    p.add_argument("--bg-severity-drop", type=float, default=0.0)
    p.add_argument("--min-rate-mult", type=float, default=0.25)
    p.add_argument("--min-bg-mult", type=float, default=0.25)

    # Frozen detector feature consistency.
    p.add_argument(
        "--detector-backend",
        type=str,
        default="yolo",
        choices=["none", "yolo", "detectron2"],
        help="Detector feature-distillation backend.",
    )
    p.add_argument("--detector-weights", type=str, default="")
    p.add_argument("--lambda-det", type=float, default=2.0)
    p.add_argument(
        "--detector-hook-layers",
        type=str,
        default="",
        help=(
            "Comma-separated Ultralytics YOLO layer indices for feature hooks. "
            "Example: 10,15,20. Empty = automatic mid/high-level layers."
        ),
    )
    p.add_argument(
        "--detector-feature-loss",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "l1", "mse"],
        help="Loss used for object-masked YOLO hook feature consistency.",
    )
    p.add_argument(
        "--print-detector-layers",
        action="store_true",
        help="Print Ultralytics YOLO layer indices before training.",
    )

    # Detectron2 Faster R-CNN / FPN feature distillation.
    p.add_argument(
        "--detectron2-config",
        type=str,
        default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        help="Detectron2 model-zoo config or local config path.",
    )
    p.add_argument(
        "--detectron2-weights",
        type=str,
        default="",
        help="Optional local Detectron2 weights. Empty = model-zoo checkpoint.",
    )
    p.add_argument(
        "--detectron2-feature-names",
        type=str,
        default="p2,p3,p4,p5,p6",
        help="Comma-separated FPN feature names to distill.",
    )
    p.add_argument(
        "--detectron2-input-format",
        type=str,
        default="BGR",
        choices=["BGR", "RGB"],
        help="Input color order expected by the Detectron2 model.",
    )
    p.add_argument(
        "--d2-fpn-mask-weight",
        type=float,
        default=0.7,
        help="Weight for object/context-masked FPN feature loss.",
    )
    p.add_argument(
        "--d2-fpn-full-weight",
        type=float,
        default=0.3,
        help="Weight for full-image FPN feature loss, useful to preserve context.",
    )

    p.add_argument(
        "--assert-objstrong",
        action="store_true",
        help=(
            "Abort unless the critical object-strong/background-replacement "
            "settings are exactly the intended ones. Useful to catch stale scripts."
        ),
    )

    # Validation
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-steps", type=int, default=50)

    # Logging
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training progress every N optimizer steps inside each epoch.",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
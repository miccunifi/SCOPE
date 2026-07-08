from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch

from roi_codec_model import ROICompressionAE
from roi_utils import merge_boxes_by_intersection, roi_quality_by_scale
from tile_utils import Tile
from utils import (
    box_area_xyxy,
    expand_box_xyxy,
    jpeg_encode_bytes,
    jpeg_encode_decode,
)

from preedit_inference import apply_preedit_bgr


@dataclass
class TransmissionResult:
    reconstructed_bgr: np.ndarray
    estimated_bytes: int
    num_rois: int
    chosen_quality: int | None = None


def uniform_jpeg_transmission(img_bgr: np.ndarray, quality: int) -> TransmissionResult:
    reconstructed, nbytes = jpeg_encode_decode(img_bgr, quality)
    return TransmissionResult(reconstructed, nbytes, 0, quality)


def downscale_jpeg_transmission(
    img_bgr: np.ndarray,
    scale: float,
    quality: int,
) -> TransmissionResult:
    h, w = img_bgr.shape[:2]
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))

    small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    decoded_small, nbytes = jpeg_encode_decode(small, quality)
    reconstructed = cv2.resize(decoded_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return TransmissionResult(reconstructed, nbytes, 0, quality)


def roi_jpeg_transmission(
    img_bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    bg_quality: int,
    roi_quality: int,
    roi_margin: float = 0.15,
    max_rois: int = 100,
    metadata_bytes_per_roi: int = 32,
    merge_rois: bool = True,
    merge_intersection_threshold: float = 0.10,
) -> TransmissionResult:
    height, width = img_bgr.shape[:2]

    low_quality_frame, bg_bytes = jpeg_encode_decode(img_bgr, bg_quality)
    reconstructed = low_quality_frame.copy()

    expanded_boxes = [
        expand_box_xyxy(box, roi_margin, width, height)
        for box in boxes_xyxy
    ]

    if merge_rois:
        expanded_boxes = merge_boxes_by_intersection(
            expanded_boxes,
            threshold=merge_intersection_threshold,
        )

    expanded_boxes = sorted(expanded_boxes, key=lambda b: box_area_xyxy(b), reverse=True)
    expanded_boxes = expanded_boxes[:max_rois]

    total_bytes = int(bg_bytes)
    used_rois = 0

    for box in expanded_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_bytes = jpeg_encode_bytes(crop, roi_quality)
        total_bytes += len(crop_bytes) + metadata_bytes_per_roi

        crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        crop_decoded = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
        if crop_decoded is None:
            continue

        if crop_decoded.shape[:2] != (y2 - y1, x2 - x1):
            crop_decoded = cv2.resize(crop_decoded, (x2 - x1, y2 - y1))

        reconstructed[y1:y2, x1:x2] = crop_decoded
        used_rois += 1

    return TransmissionResult(reconstructed, total_bytes, used_rois, None)


def scale_aware_roi_jpeg_transmission(
    img_bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    bg_quality: int,
    roi_margin: float = 0.15,
    max_rois: int = 100,
    metadata_bytes_per_roi: int = 32,
    merge_rois: bool = True,
    merge_intersection_threshold: float = 0.10,
    tiny_quality: int = 85,
    small_quality: int = 75,
    medium_quality: int = 65,
    large_quality: int = 55,
    tiny_thr: float = 0.001,
    small_thr: float = 0.005,
    medium_thr: float = 0.02,
) -> TransmissionResult:
    height, width = img_bgr.shape[:2]

    low_quality_frame, bg_bytes = jpeg_encode_decode(img_bgr, bg_quality)
    reconstructed = low_quality_frame.copy()

    expanded_boxes = [
        expand_box_xyxy(box, roi_margin, width, height)
        for box in boxes_xyxy
    ]

    if merge_rois:
        expanded_boxes = merge_boxes_by_intersection(
            expanded_boxes,
            threshold=merge_intersection_threshold,
        )

    expanded_boxes = sorted(expanded_boxes, key=lambda b: box_area_xyxy(b), reverse=True)
    expanded_boxes = expanded_boxes[:max_rois]

    total_bytes = int(bg_bytes)
    used_rois = 0

    for box in expanded_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        q = roi_quality_by_scale(
            box_xyxy=box,
            width=width,
            height=height,
            tiny_thr=tiny_thr,
            small_thr=small_thr,
            medium_thr=medium_thr,
            tiny_quality=tiny_quality,
            small_quality=small_quality,
            medium_quality=medium_quality,
            large_quality=large_quality,
        )

        crop_bytes = jpeg_encode_bytes(crop, q)
        total_bytes += len(crop_bytes) + metadata_bytes_per_roi

        crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        crop_decoded = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
        if crop_decoded is None:
            continue

        if crop_decoded.shape[:2] != (y2 - y1, x2 - x1):
            crop_decoded = cv2.resize(crop_decoded, (x2 - x1, y2 - y1))

        reconstructed[y1:y2, x1:x2] = crop_decoded
        used_rois += 1

    return TransmissionResult(reconstructed, total_bytes, used_rois, None)


def tile_importance_transmission(
    img_bgr: np.ndarray,
    selected_tiles: List[Tile],
    bg_quality: int,
    tile_quality: int,
    metadata_bytes_per_tile: int = 8,
) -> TransmissionResult:
    low_quality_frame, bg_bytes = jpeg_encode_decode(img_bgr, bg_quality)
    reconstructed = low_quality_frame.copy()

    total_bytes = int(bg_bytes)
    used_tiles = 0

    for tile in selected_tiles:
        x1, y1, x2, y2 = tile
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_bytes = jpeg_encode_bytes(crop, tile_quality)
        total_bytes += len(crop_bytes) + metadata_bytes_per_tile

        crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        crop_decoded = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
        if crop_decoded is None:
            continue

        if crop_decoded.shape[:2] != (y2 - y1, x2 - x1):
            crop_decoded = cv2.resize(crop_decoded, (x2 - x1, y2 - y1))

        reconstructed[y1:y2, x1:x2] = crop_decoded
        used_tiles += 1

    return TransmissionResult(reconstructed, total_bytes, used_tiles, None)


_COMPRESSAI_MODEL_CACHE = {}


def _get_device_str(device: str | int | None = None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def load_compressai_model(
    model_name: str,
    quality: int,
    device: str | int | None = None,
):
    try:
        from compressai.zoo import bmshj2018_hyperprior, cheng2020_attn, mbt2018_mean
    except ImportError as e:
        raise ImportError(
            "CompressAI baseline requires: pip install compressai torch torchvision"
        ) from e

    device_str = _get_device_str(device)
    model_name = model_name.lower()

    key = (model_name, int(quality), device_str)
    if key in _COMPRESSAI_MODEL_CACHE:
        return _COMPRESSAI_MODEL_CACHE[key]

    if model_name == "bmshj2018-hyperprior":
        net = bmshj2018_hyperprior(quality=quality, pretrained=True)
    elif model_name == "cheng2020-attn":
        net = cheng2020_attn(quality=quality, pretrained=True)
    elif model_name == "mbt2018-mean":
        net = mbt2018_mean(quality=quality, pretrained=True)
    else:
        raise ValueError(f"Unsupported CompressAI model: {model_name}")

    net = net.eval().to(device_str)
    _COMPRESSAI_MODEL_CACHE[key] = net
    return net


def compressai_transmission(
    img_bgr: np.ndarray,
    model_name: str,
    quality: int,
    device: str | int | None = None,
) -> TransmissionResult:
    """
    Uses CompressAI pretrained models.
    Estimated bytes are computed from likelihoods/bpp, not actual entropy bitstream.
    """
    import torch.nn.functional as F

    device_str = _get_device_str(device)
    net = load_compressai_model(model_name, quality, device_str)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device_str)

    _, _, h, w = x.shape
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    with torch.no_grad():
        out = net(x_pad)
        x_hat = out["x_hat"].clamp(0, 1)

        total_bits = 0.0
        for likelihood in out["likelihoods"].values():
            total_bits += torch.log(likelihood).sum() / (-np.log(2.0))
        estimated_bytes = int(float(total_bits.item()) / 8.0)

    x_hat = x_hat[:, :, :h, :w]
    arr = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return TransmissionResult(bgr, estimated_bytes, 0, quality)


def compressai_roi_transmission(
    img_bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    compressai_model: str,
    compressai_quality: int,
    roi_quality: int,
    roi_margin: float = 0.15,
    max_rois: int = 100,
    metadata_bytes_per_roi: int = 32,
    merge_rois: bool = True,
    merge_intersection_threshold: float = 0.10,
    device: str | int | None = None,
) -> TransmissionResult:
    height, width = img_bgr.shape[:2]

    bg_result = compressai_transmission(
        img_bgr=img_bgr,
        model_name=compressai_model,
        quality=compressai_quality,
        device=device,
    )

    reconstructed = bg_result.reconstructed_bgr.copy()
    total_bytes = int(bg_result.estimated_bytes)

    expanded_boxes = [
        expand_box_xyxy(box, roi_margin, width, height)
        for box in boxes_xyxy
    ]

    if merge_rois:
        expanded_boxes = merge_boxes_by_intersection(
            expanded_boxes,
            threshold=merge_intersection_threshold,
        )

    expanded_boxes = sorted(
        expanded_boxes,
        key=lambda b: box_area_xyxy(b),
        reverse=True,
    )[:max_rois]

    used_rois = 0

    for box in expanded_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_bytes = jpeg_encode_bytes(crop, roi_quality)
        total_bytes += len(crop_bytes) + metadata_bytes_per_roi

        crop_arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        crop_decoded = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)

        if crop_decoded is None:
            continue

        if crop_decoded.shape[:2] != (y2 - y1, x2 - x1):
            crop_decoded = cv2.resize(crop_decoded, (x2 - x1, y2 - y1))

        reconstructed[y1:y2, x1:x2] = crop_decoded
        used_rois += 1

    return TransmissionResult(
        reconstructed_bgr=reconstructed,
        estimated_bytes=total_bytes,
        num_rois=used_rois,
        chosen_quality=compressai_quality,
    )


_ROI_CODEC_CACHE = {}


def load_roi_codec(
    checkpoint_path: str,
    latent_ch: int,
    device: str | int | None = None,
) -> ROICompressionAE:
    device_str = _get_device_str(device)
    key = (checkpoint_path, latent_ch, device_str)

    if key in _ROI_CODEC_CACHE:
        return _ROI_CODEC_CACHE[key]

    model = ROICompressionAE(latent_ch=latent_ch, quantize=True)
    ckpt = torch.load(checkpoint_path, map_location=device_str)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device_str).eval()
    _ROI_CODEC_CACHE[key] = model
    return model


def learned_roi_codec_transmission(
    img_bgr: np.ndarray,
    boxes_xyxy: List[List[float]],
    bg_codec: str,
    bg_quality: int,
    roi_codec_ckpt: str,
    roi_latent_ch: int,
    roi_crop_size: int,
    roi_margin: float = 0.15,
    max_rois: int = 100,
    bits_per_latent: int = 8,
    metadata_bytes_per_roi: int = 32,
    merge_rois: bool = True,
    merge_intersection_threshold: float = 0.10,
    device: str | int | None = None,
) -> TransmissionResult:
    """
    Full-frame low-bitrate background + learned ROI codec.

    bg_codec options:
      - "jpeg"
      - "compressai_bmshj2018-hyperprior"
      - "compressai_cheng2020-attn"
    """
    height, width = img_bgr.shape[:2]
    device_str = _get_device_str(device)

    if bg_codec == "jpeg":
        bg_result = uniform_jpeg_transmission(img_bgr, bg_quality)
    elif bg_codec == "compressai_bmshj2018-hyperprior":
        bg_result = compressai_transmission(
            img_bgr=img_bgr,
            model_name="bmshj2018-hyperprior",
            quality=bg_quality,
            device=device_str,
        )
    elif bg_codec == "compressai_cheng2020-attn":
        bg_result = compressai_transmission(
            img_bgr=img_bgr,
            model_name="cheng2020-attn",
            quality=bg_quality,
            device=device_str,
        )
    else:
        raise ValueError(f"Unknown bg_codec: {bg_codec}")

    reconstructed = bg_result.reconstructed_bgr.copy()
    total_bytes = int(bg_result.estimated_bytes)

    roi_codec = load_roi_codec(
        checkpoint_path=roi_codec_ckpt,
        latent_ch=roi_latent_ch,
        device=device_str,
    )

    expanded_boxes = [
        expand_box_xyxy(box, roi_margin, width, height)
        for box in boxes_xyxy
    ]

    if merge_rois:
        expanded_boxes = merge_boxes_by_intersection(
            expanded_boxes,
            threshold=merge_intersection_threshold,
        )

    expanded_boxes = sorted(
        expanded_boxes,
        key=lambda b: box_area_xyxy(b),
        reverse=True,
    )[:max_rois]

    used_rois = 0

    with torch.no_grad():
        for box in expanded_boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_resized = cv2.resize(
                crop,
                (roi_crop_size, roi_crop_size),
                interpolation=cv2.INTER_LINEAR,
            )
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(crop_rgb).float() / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0).to(device_str)

            decoded, z, z_q = roi_codec(x)

            roi_bytes = ROICompressionAE.fixed_rate_bytes(
                z_q=z_q,
                bits_per_latent=bits_per_latent,
            )
            total_bytes += int(roi_bytes) + metadata_bytes_per_roi
            decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
            decoded_np = (decoded_np.clip(0, 1) * 255.0).round().astype(np.uint8)
            decoded_bgr = cv2.cvtColor(decoded_np, cv2.COLOR_RGB2BGR)

            decoded_bgr = cv2.resize(
                decoded_bgr,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_LINEAR,
            )

            reconstructed[y1:y2, x1:x2] = decoded_bgr
            used_rois += 1

    return TransmissionResult(
        reconstructed_bgr=reconstructed,
        estimated_bytes=total_bytes,
        num_rois=used_rois,
        chosen_quality=bg_quality,
    )

def preedit_jpeg_transmission(
    img_bgr,
    preedit_ckpt: str,
    jpeg_quality: int,
    device=None,
    base_ch: int = 32,
    max_delta: float = 0.05,
    tile_size: int = 512,
):
    edited = apply_preedit_bgr(
        img_bgr=img_bgr,
        ckpt_path=preedit_ckpt,
        device=device,
        base_ch=base_ch,
        max_delta=max_delta,
        tile_size=tile_size,
    )

    reconstructed, nbytes = jpeg_encode_decode(
        edited,
        jpeg_quality,
    )

    return TransmissionResult(
        reconstructed_bgr=reconstructed,
        estimated_bytes=nbytes,
        num_rois=0,
        chosen_quality=jpeg_quality,
    )


def preedit_compressai_transmission(
    img_bgr,
    preedit_ckpt: str,
    compressai_model: str,
    compressai_quality: int,
    device=None,
    base_ch: int = 32,
    max_delta: float = 0.05,
    tile_size: int = 512,
):
    edited = apply_preedit_bgr(
        img_bgr=img_bgr,
        ckpt_path=preedit_ckpt,
        device=device,
        base_ch=base_ch,
        max_delta=max_delta,
        tile_size=tile_size,
    )

    result = compressai_transmission(
        img_bgr=edited,
        model_name=compressai_model,
        quality=compressai_quality,
        device=device,
    )

    return TransmissionResult(
        reconstructed_bgr=result.reconstructed_bgr,
        estimated_bytes=result.estimated_bytes,
        num_rois=0,
        chosen_quality=compressai_quality,
    )
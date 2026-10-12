from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

                # For training the SCOPE, train mode gives differentiable
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

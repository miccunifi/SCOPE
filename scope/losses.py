from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


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

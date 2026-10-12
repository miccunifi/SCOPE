from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class QuantizeSTE(torch.autograd.Function):
    """
    Straight-through estimator for rounding.

    Forward: round(x)
    Backward: identity gradient
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    return QuantizeSTE.apply(x)


@dataclass
class RateEstimate:
    fixed_bytes: int
    sparse_bytes: int
    active_values: int
    total_values: int
    active_ratio: float


class ROICompressionAE(nn.Module):
    """
    Small detection-stable ROI autoencoder.

    Input/output:
        B x 3 x H x W, typically 128x128.

    Latent:
        B x latent_ch x H/8 x W/8.

    For crop_size=128 and latent_ch=8:
        latent = 8 x 16 x 16 = 2048 values.
    """

    def __init__(
        self,
        latent_ch: int = 8,
        quantize: bool = True,
        latent_scale: float = 8.0,
    ):
        super().__init__()
        self.latent_ch = latent_ch
        self.quantize = quantize
        self.latent_scale = latent_scale

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, latent_ch, kernel_size=3, stride=1, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Scale latent before quantization so rounding is not too destructive.
        return self.encoder(x) * self.latent_scale

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        # Undo latent scale before decoding.
        return self.decoder(z_q / self.latent_scale)

    def quantize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Additive uniform noise approximates quantization.
            return z + torch.empty_like(z).uniform_(-0.5, 0.5)

        if self.quantize:
            return quantize_ste(z)

        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_hat: reconstructed crop
            z: continuous latent
            z_q: quantized/noisy latent
        """
        z = self.encode(x)
        z_q = self.quantize_latent(z)
        x_hat = self.decode(z_q)
        return x_hat, z, z_q

    @staticmethod
    def fixed_rate_bytes(
        z_q: torch.Tensor,
        bits_per_latent: int = 8,
    ) -> int:
        """
        Fixed-rate estimate:
            number of latent values * bits_per_latent.
        """
        num_values = z_q.numel()
        return int(num_values * bits_per_latent / 8)

    @staticmethod
    def sparse_rate_bytes(
        z_q: torch.Tensor,
        bits_per_active_value: int = 8,
        active_threshold: float = 0.5,
        index_bits: int = 16,
    ) -> int:
        """
        Sparse latent estimate:
            transmit only active latent values and their indices.

        bytes = active_values * (value_bits + index_bits) / 8

        This is not a full entropy coder, but it is a more realistic
        bandwidth proxy than assuming all latents cost 8 bits.
        """
        active = torch.abs(z_q) > active_threshold
        active_values = int(active.sum().item())
        bits = active_values * (bits_per_active_value + index_bits)
        return int(bits / 8)

    @staticmethod
    def estimate_rate(
        z_q: torch.Tensor,
        bits_per_latent: int = 8,
        bits_per_active_value: int = 8,
        active_threshold: float = 0.5,
        index_bits: int = 16,
    ) -> RateEstimate:
        total_values = int(z_q.numel())
        active_values = int((torch.abs(z_q) > active_threshold).sum().item())

        fixed_bytes = ROICompressionAE.fixed_rate_bytes(
            z_q=z_q,
            bits_per_latent=bits_per_latent,
        )

        sparse_bytes = ROICompressionAE.sparse_rate_bytes(
            z_q=z_q,
            bits_per_active_value=bits_per_active_value,
            active_threshold=active_threshold,
            index_bits=index_bits,
        )

        active_ratio = active_values / max(1, total_values)

        return RateEstimate(
            fixed_bytes=fixed_bytes,
            sparse_bytes=sparse_bytes,
            active_values=active_values,
            total_values=total_values,
            active_ratio=active_ratio,
        )


def latent_rate_loss(
    z: torch.Tensor,
    mode: str = "log",
) -> torch.Tensor:
    """
    Differentiable proxy for compressibility.

    Modes:
        l1:
            encourages small latents.
        log:
            encourages sparse / low-magnitude latents more smoothly.
        l2:
            stronger penalty on large latents.
    """
    if mode == "l1":
        return torch.abs(z).mean()

    if mode == "log":
        return torch.log1p(torch.abs(z)).mean()

    if mode == "l2":
        return torch.square(z).mean()

    raise ValueError(f"Unknown rate loss mode: {mode}")
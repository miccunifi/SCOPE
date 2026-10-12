from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SCOPE(nn.Module):
    """
    Codec-family and severity conditioned image model.

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

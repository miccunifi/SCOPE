from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenYOLOFeatureConsistency(nn.Module):
    """
    Frozen Ultralytics YOLO feature consistency loss using forward hooks.

    Teacher:
        detector features from clean image

    Student:
        detector features from compressed(scopeed image)

    The YOLO detector is frozen.
    Gradients flow through the student detector forward into the degraded image,
    then into the SCOPE.

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
            compressed(scopeed image) in [0, 1], [B, 3, H, W]

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

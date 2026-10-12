"""CPU-only training-path integration tests with synthetic COCO data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scope.data import CocoBoxCropDataset
from scope.model import SCOPE
from scope.train_scope import build_argparser
from scope.training import compute_losses


class LightweightCodec(torch.nn.Module):
    def forward(self, images, *, family_ids, severity_norm, id_to_codec_family):
        assert family_ids.shape[0] == images.shape[0]
        assert all(int(value) in id_to_codec_family for value in family_ids)
        return torch.clamp(images * 0.92 + 0.02, 0.0, 1.0), images.mean() * 0.01


class LightweightTeacher(torch.nn.Module):
    def forward(self, *, clean, degraded, object_mask):
        return (((clean - degraded) ** 2) * object_mask).mean()


class TrainingSmokeTests(unittest.TestCase):
    def test_coco_crop_and_object_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            data = np.zeros((80, 80, 3), dtype=np.uint8)
            data[10:40, 12:42] = (255, 128, 20)
            Image.fromarray(data).save(root / "images/example.png")
            (root / "gt.json").write_text(json.dumps({
                "images": [{"id": 10, "file_name": "example.png", "width": 80, "height": 80}],
                "annotations": [{"id": 1, "image_id": 10, "category_id": 1, "bbox": [12, 10, 30, 30]}],
            }))
            dataset = CocoBoxCropDataset(root / "images", root / "gt.json", crop_size=64, object_crop_prob=1.0)
            sample = dataset[0]
            image, mask = sample["image"], sample["mask"]
            self.assertEqual(tuple(image.shape), (3, 64, 64))
            self.assertEqual(tuple(mask.shape), (1, 64, 64))
            self.assertGreater(float(mask.sum()), 0)
            self.assertTrue(torch.isfinite(image).all())

    def test_training_loss_backprop_and_optimizer_step(self):
        torch.manual_seed(42)
        args = build_argparser().parse_args([
            "--images-dir", "dummy", "--annotations-json", "dummy.json", "--output-dir", "dummy_run",
            "--device", "cpu", "--base-ch", "8", "--lambda-det", "1",
        ])
        model = SCOPE(num_codec_families=1, base_ch=8, bg_mode="residual")
        images = torch.rand(1, 3, 64, 64)
        masks = torch.zeros(1, 1, 64, 64)
        masks[:, :, 15:48, 15:48] = 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.enc1.conv1.weight.detach().clone()
        loss, logs = compute_losses(
            args=args, model=model, codec_proxy_manager=LightweightCodec(),
            detector_loss_module=LightweightTeacher(), images=images, masks=masks,
            family_ids=torch.tensor([0]), severity_norm=torch.tensor([0.5]),
            id_to_codec_family={0: "diff_jpeg"},
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("det", logs)
        self.assertGreater(float(logs["det"]), 0)
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.enc1.conv1.weight.detach()))

    def test_default_loss_without_teacher(self):
        args = build_argparser().parse_args([
            "--images-dir", "x", "--annotations-json", "y", "--output-dir", "z", "--device", "cpu",
        ])
        args.bpg_proxy_mode = "old"
        model = SCOPE(num_codec_families=1, base_ch=8, bg_mode="residual")
        image = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        loss, logs = compute_losses(
            args=args, model=model, codec_proxy_manager=LightweightCodec(),
            detector_loss_module=None, images=image, masks=mask,
            family_ids=torch.tensor([0]), severity_norm=torch.tensor([0.25]),
            id_to_codec_family={0: "diff_jpeg"},
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logs["det"]), 0)


if __name__ == "__main__":
    unittest.main()

"""Benchmark the actual SCOPE timing path using a synthetic CPU checkpoint."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from scope.evaluation.dataset import CocoDataset
from scope.evaluation.timing import benchmark_experiment
from scope.model import SCOPE


class TimingIntegrationTests(unittest.TestCase):
    def test_timing_loads_model_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[8:24, 8:24] = 160
            self.assertTrue(cv2.imwrite(str(root / "images/one.png"), image))
            (root / "gt.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "one.png", "width": 32, "height": 32}],
                "annotations": [],
            }))
            weights = root / "scope.pt"
            model = SCOPE(num_codec_families=1, base_ch=8, bg_mode="residual")
            torch.save({"model": model.state_dict(), "codec_family_to_id": {"diff_jpeg": 0},
                        "args": {"base_ch": 8, "bg_mode": "residual"}}, weights)
            exp = {"name": "timing_scope", "mode": "scope_jpeg", "codec_family": "block_dct",
                   "scope_ckpt": str(weights), "base_ch": 8, "bg_mode": "residual",
                   "condition": {"severity": 0.25}}
            dataset = CocoDataset(str(root / "images"), str(root / "gt.json"))
            rows, summary = benchmark_experiment(exp, dataset, root / "timings", "cpu",
                                                limit=1, warmup=0, repeat=2, save_outputs=True,
                                                max_saved_outputs=1, severity_bin_size=None)
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["num_measurements"], 2)
            self.assertTrue((root / "timings/timing_scope/scope_timing_per_image.csv").is_file())
            self.assertTrue((root / "timings/timing_scope/edited/one.png").is_file())
            self.assertGreater(summary["scope_forward_ms_mean"], 0)


if __name__ == "__main__":
    unittest.main()

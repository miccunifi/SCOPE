"""Validate the real-smoke subset config generator without running detectors."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.smoke_real import prepare_subset


class RealSmokeConfigTests(unittest.TestCase):
    def test_subset_filters_images_annotations_and_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            for name in ("1.png", "2.png"):
                (root / "images" / name).write_bytes(b"image placeholder")
            coco = {
                "images": [{"id": 1, "file_name": "1.png"}, {"id": 2, "file_name": "2.png"}],
                "annotations": [{"id": 1, "image_id": 1}, {"id": 2, "image_id": 2}],
                "categories": [{"id": 1, "name": "object"}],
            }
            (root / "gt.json").write_text(json.dumps(coco))
            config = root / "eval.yaml"
            config.write_text(yaml.safe_dump({
                "dataset": {"images_dir": "images", "annotations_json": "gt.json"},
                "models": {"server_model": "detector.pt"},
                "experiments": [
                    {"name": "jpeg_q10", "mode": "uniform", "jpeg_quality": 10},
                    {"name": "scope_jpeg_q10", "mode": "scope_jpeg", "jpeg_quality": 10, "scope_ckpt": "best.pt"},
                ],
            }))
            output = prepare_subset(config, ["jpeg_q10"], 1, root, root / "results", root)
            loaded = yaml.safe_load(output.read_text())
            self.assertEqual(len(loaded["experiments"]), 1)
            self.assertEqual(loaded["experiments"][0]["name"], "jpeg_q10")
            subset = json.loads(Path(loaded["dataset"]["annotations_json"]).read_text())
            self.assertEqual([im["id"] for im in subset["images"]], [1])
            self.assertEqual([ann["image_id"] for ann in subset["annotations"]], [1])
            self.assertEqual(Path(loaded["models"]["server_model"]), root / "detector.pt")
            self.assertEqual(len(json.loads((root / "gt.json").read_text())["images"]), 2)

    def test_rejects_unknown_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "eval.yaml"
            config.write_text(yaml.safe_dump({"experiments": [{"name": "a", "mode": "uniform"}]}))
            with self.assertRaises(ValueError):
                prepare_subset(config, ["missing"], 1, root, root / "results", root)


if __name__ == "__main__":
    unittest.main()

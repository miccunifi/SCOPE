"""Run the real evaluation loop with synthetic images and a stubbed detector/COCO metric backend."""
from __future__ import annotations

import csv
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scope.evaluation.dataset import CocoDataset


class FakeDetector:
    def __init__(self):
        self.calls = []

    def predict_full(self, *, image_path, image_id, category_id_offset, conf, iou, max_det):
        self.calls.append((Path(image_path), image_id, conf, iou, max_det))
        self.calls[-1][0].stat()
        return [{"image_id": image_id, "category_id": 0, "bbox": [2, 2, 10, 10],
                 "score": 0.9, "_xyxy": [2, 2, 12, 12]}]


class EvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_coco = types.ModuleType("pycocotools.coco")
        fake_coco.COCO = type("COCO", (), {})
        fake_eval = types.ModuleType("pycocotools.cocoeval")
        fake_eval.COCOeval = type("COCOeval", (), {})
        fake_ultra = types.ModuleType("ultralytics")
        fake_ultra.YOLO = type("YOLO", (), {})
        fake_pkg = types.ModuleType("pycocotools")
        fake_pkg.__path__ = []
        cls.modules = patch.dict(sys.modules, {
            "pycocotools": fake_pkg, "pycocotools.coco": fake_coco,
            "pycocotools.cocoeval": fake_eval, "ultralytics": fake_ultra,
        })
        cls.modules.start()
        cls.evaluation = importlib.import_module("scope.evaluation.evaluate")

    @classmethod
    def tearDownClass(cls):
        for module in ("scope.evaluation.evaluate", "scope.evaluation.coco_metrics", "scope.evaluation.yolo_utils"):
            sys.modules.pop(module, None)
        cls.modules.stop()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[10:30, 10:32] = (20, 125, 245)
        assert cv2.imwrite(str(self.root / "images/one.png"), image)
        (self.root / "gt.json").write_text(json.dumps({
            "images": [{"id": 7, "file_name": "one.png", "width": 64, "height": 48}],
            "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [2, 2, 10, 10]}],
            "categories": [{"id": 1, "name": "object"}],
        }))
        self.dataset = CocoDataset(str(self.root / "images"), str(self.root / "gt.json"))
        self.detector = FakeDetector()

    def tearDown(self):
        self.tmp.cleanup()

    def run_experiment(self, exp):
        def coco_stub(annotation, predictions):
            self.assertTrue(Path(annotation).is_file())
            values = json.loads(Path(predictions).read_text())
            self.assertEqual(len(values), 1)
            self.assertNotIn("_xyxy", values[0])
            self.assertEqual(values[0]["category_id"], 1)
            return {"AP": 0.42, "AP50": 0.65}

        with patch.object(self.evaluation, "evaluate_coco_map", side_effect=coco_stub):
            return self.evaluation.run_single_experiment(
                exp, self.dataset, self.root / "results", self.detector, 0,
                {"category_map": "coco91", "conf": 0.001}, "cpu",
            )

    def verify_output(self, exp_name, summary):
        directory = self.root / "results" / exp_name
        self.assertTrue((directory / "reconstructed/one.png").is_file())
        self.assertTrue((directory / "summary.json").is_file())
        self.assertTrue((directory / "predictions_coco.json").is_file())
        with (directory / "bandwidth.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["image_id"]), 7)
        self.assertGreater(int(rows[0]["estimated_bytes"]), 0)
        self.assertGreater(float(summary["dataset_bpp"]), 0)
        self.assertEqual(summary["AP"], 0.42)
        self.assertEqual(len(self.detector.calls), 1)

    def test_baseline_jpeg_generates_all_outputs(self):
        summary = self.run_experiment({
            "name": "jpeg_q40", "mode": "uniform", "codec_family": "block_dct", "jpeg_quality": 40,
        })
        self.verify_output("jpeg_q40", summary)

    def test_scope_jpeg_dispatch_and_conditioning(self):
        original_image = cv2.imread(str(self.root / "images/one.png"))
        with patch.object(self.evaluation, "scope_jpeg_transmission", return_value={
            "reconstructed_bgr": original_image, "estimated_bytes": 150, "debug": None,
        }) as encode:
            summary = self.run_experiment({
                "name": "scope_jpeg_q40", "mode": "scope_jpeg", "codec_family": "block_dct",
                "jpeg_quality": 40, "scope_ckpt": "not_loaded_when_mocked.pt", "condition": {"severity": 0.6},
            })
        encode.assert_called_once()
        self.assertEqual(encode.call_args.args[2], 40)
        self.assertAlmostEqual(summary["severity"], 0.6)
        self.verify_output("scope_jpeg_q40", summary)


if __name__ == "__main__":
    unittest.main()

"""Small CPU smoke checks for the evaluation data and codec interfaces."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scope.evaluation.conditioning import (
    get_experiment_severity, infer_codec_family_from_experiment,
)
from scope.evaluation.dataset import CocoDataset
from scope.evaluation.utils import jpeg_encode_decode


class EvaluationContracts(unittest.TestCase):
    def test_coco_dataset_resolves_image_paths_and_gt_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            path = root / "images/one.jpg"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((16, 20, 3), dtype=np.uint8)))
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps({
                "images": [{"id": 10, "file_name": "one.jpg", "width": 20, "height": 16}],
                "annotations": [
                    {"id": 1, "image_id": 10, "bbox": [2, 3, 5, 7], "category_id": 1},
                    {"id": 2, "image_id": 10, "bbox": [0, 0, 2, 2], "category_id": 1, "iscrowd": 1},
                ],
                "categories": [{"id": 1, "name": "person"}],
            }))
            dataset = CocoDataset(str(root / "images"), str(annotations))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.images[0].path, path)
            self.assertEqual(dataset.get_gt_boxes_xyxy(10), [[2.0, 3.0, 7.0, 10.0]])

    def test_jpeg_round_trip_reports_real_positive_bytes(self):
        image = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        image = np.repeat(image[..., None], 3, axis=2)
        decoded, encoded_bytes = jpeg_encode_decode(image, quality=40)
        self.assertEqual(decoded.shape, image.shape)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertGreater(encoded_bytes, 0)

    def test_codec_conditioning_maps_to_trained_family(self):
        examples = [
            ({"mode": "scope_jpeg", "codec_family": "block_dct", "condition": {"severity": 0.25}}, "diff_jpeg", 0.25),
            ({"mode": "scope_bpg", "codec_family": "intra_smooth", "condition": {"severity": 0.75}}, "bpg_like", 0.75),
            ({"mode": "scope_compressai", "codec_family": "bmshj", "condition": {"severity": 0.5}}, "bmshj", 0.5),
        ]
        for config, family, severity in examples:
            with self.subTest(mode=config["mode"]):
                self.assertEqual(infer_codec_family_from_experiment(config), family)
                self.assertEqual(get_experiment_severity(config), severity)


if __name__ == "__main__":
    unittest.main()

"""Isolated tests for preflight path validation; no external data is needed."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.validate_environment import (
    Reporter, check_config, check_dataset, check_output, check_result_directories, find_executable,
    resolve_local_path,
)


class PreflightTests(unittest.TestCase):
    def test_relative_paths_resolve_against_repository_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resolve_local_path("models/yolo.pt", root), root / "models/yolo.pt")
            self.assertEqual(resolve_local_path(root / "absolute.pt", root), root / "absolute.pt")

    def test_dataset_validator_detects_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "coco.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "missing.png"}]}))
            report = Reporter(root)
            check_dataset(report, "images", "coco.json", "sample", check_images=-1)
            self.assertTrue(any(x.status == "FAIL" and x.subject == "sample.image-files" for x in report.checks))
            (root / "images/missing.png").write_bytes(b"x")
            report = Reporter(root)
            check_dataset(report, "images", "coco.json", "sample", check_images=-1)
            self.assertFalse(any(x.status == "FAIL" for x in report.checks))

    def test_missing_annotation_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Reporter(Path(tmp))
            check_dataset(report, "images", "missing.json", "sample")
            self.assertTrue(any(x.status == "FAIL" and x.subject == "sample.annotations" for x in report.checks))

    def test_unwritable_output_target_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "not_a_directory").write_text("x")
            report = Reporter(root)
            check_output(report, root / "not_a_directory", "output")
            self.assertEqual(report.checks[-1].status, "FAIL")

    def test_codec_executable_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "myencoder"
            file.write_text("#!/bin/sh\nexit 0\n")
            file.chmod(0o755)
            self.assertEqual(find_executable("myencoder", root), None)
            self.assertEqual(find_executable("./myencoder", root), str(file))
            with patch.dict(os.environ, {"PATH": str(root)}):
                self.assertEqual(find_executable("myencoder", root), str(file))

    def test_eval_config_detects_missing_checkpoint_and_codec(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "coco.json").write_text(json.dumps({"images": []}))
            config = {
                "output_dir": "output", "dataset": {"images_dir": "images", "annotations_json": "coco.json"},
                "models": {"server_model": "detector.pt"},
                "experiments": [
                    {"name": "bpg_q28", "mode": "bpg", "bpg_qp": 28},
                    {"name": "scope_bpg_q28", "mode": "scope_bpg", "scope_ckpt": "scope.pt", "codec_family": "intra_smooth"},
                ],
                "bpg_encoder_bin": "missing_bpgenc", "bpg_decoder_bin": "missing_bpgdec",
            }
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            report = Reporter(root)
            check_config(report, path, purpose="eval", check_packages=False)
            failed = {item.subject for item in report.checks if item.status == "FAIL"}
            self.assertIn("detector-weights", failed)
            self.assertIn("scope-checkpoint", failed)
            self.assertIn("codec:bpg_encoder_bin", failed)
            self.assertIn("codec:bpg_decoder_bin", failed)
            self.assertNotIn("config-format", failed)

    def test_saved_outputs_require_reconstructions_and_bandwidth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = root / "eval" / "scope_jpeg_q10"
            exp.mkdir(parents=True)
            (exp / "summary.json").write_text(json.dumps({"AP": 0.2, "dataset_bpp": 0.1}))
            report = Reporter(root)
            check_result_directories(report, [Path("eval")], purpose="saved")
            self.assertTrue(any(item.status == "FAIL" and item.subject.endswith(".reconstructed") for item in report.checks))
            (exp / "reconstructed").mkdir()
            (exp / "bandwidth.csv").write_text("image_id,estimated_bytes\n1,100\n")
            report = Reporter(root)
            check_result_directories(report, [Path("eval")], purpose="saved")
            self.assertFalse([item for item in report.checks if item.status == "FAIL"])
            check_result_directories(report, [Path("eval")], purpose="analysis")
            self.assertFalse([item for item in report.checks if item.status == "FAIL"])

    def test_timing_does_not_require_unused_codec_binaries_or_detector(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "coco.json").write_text(json.dumps({"images": []}))
            (root / "model.pt").write_bytes(b"placeholder")
            config = {
                "output_dir": "output", "dataset": {"images_dir": "images", "annotations_json": "coco.json"},
                "models": {"server_model": "not_required.pt"},
                "experiments": [{"name": "time", "mode": "scope_bpg", "scope_ckpt": "model.pt"}],
                "bpg_encoder_bin": "missing_bpgenc", "bpg_decoder_bin": "missing_bpgdec",
            }
            path = root / "timing.yaml"
            path.write_text(yaml.safe_dump(config))
            report = Reporter(root)
            check_config(report, path, purpose="timing", check_packages=False)
            self.assertFalse([x for x in report.checks if x.status == "FAIL"], report.checks)


if __name__ == "__main__":
    unittest.main()

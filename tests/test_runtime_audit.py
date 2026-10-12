"""Exercise the optional runtime checkpoint audit without external weights."""
from __future__ import annotations

import tempfile
import contextlib
import io
import unittest
from pathlib import Path

import torch
import yaml

from scope.model import SCOPE
from tools.check_runtime import audit_checkpoints


class RuntimeAuditTests(unittest.TestCase):
    def test_checkpoint_loader_accepts_matching_codec_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best.pt"
            model = SCOPE(num_codec_families=1, base_ch=8, bg_mode="residual")
            torch.save({
                "model": model.state_dict(),
                "codec_family_to_id": {"diff_jpeg": 0},
                "args": {"base_ch": 8, "bg_mode": "residual"},
            }, checkpoint)
            config = root / "eval.yaml"
            config.write_text(yaml.safe_dump({
                "experiments": [{"name": "scope_jpeg_q50", "mode": "scope_jpeg",
                                 "scope_ckpt": "best.pt", "codec_family": "block_dct"}],
            }))
            with contextlib.redirect_stdout(io.StringIO()):
                checked, failures = audit_checkpoints([config], root=root)
            self.assertEqual((checked, failures), (1, 0))
            config.write_text(yaml.safe_dump({
                "experiments": [{"name": "scope_bpg_q40", "mode": "scope_bpg",
                                 "scope_ckpt": "best.pt", "codec_family": "intra_smooth"}],
            }))
            with contextlib.redirect_stdout(io.StringIO()):
                _, failures = audit_checkpoints([config], root=root)
            self.assertEqual(failures, 1)

    def test_checkpoint_audit_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "eval.yaml"
            config.write_text(yaml.safe_dump({
                "experiments": [{"name": "scope_jpeg_q10", "mode": "scope_jpeg",
                                 "scope_ckpt": "missing.pt", "codec_family": "block_dct"}],
            }))
            with contextlib.redirect_stdout(io.StringIO()):
                _, failures = audit_checkpoints([config], root=root)
            self.assertEqual(failures, 1)


if __name__ == "__main__":
    unittest.main()


class UnseenJpeg2000Tests(unittest.TestCase):
    def test_jpeg2000_ablation_conditions_on_trained_families(self):
        from scope.evaluation.conditioning import infer_codec_family_from_experiment

        root = Path(__file__).resolve().parents[1]
        main = yaml.safe_load((root / "configs/eval.yaml").read_text())
        ablation = yaml.safe_load((root / "configs/ablation.yaml").read_text())
        main_jp2 = [exp for exp in main["experiments"] if exp["mode"] == "scope_jpeg2000"]
        self.assertEqual(main_jp2, [])

        ablation_jp2 = [
            exp for exp in ablation["experiments"]
            if exp["mode"] == "scope_jpeg2000"
        ]
        self.assertEqual(len(ablation_jp2), 18)
        families = [infer_codec_family_from_experiment(exp) for exp in ablation_jp2]
        self.assertEqual({name: families.count(name) for name in set(families)}, {
            "diff_jpeg": 6, "bmshj": 6, "bpg_like": 6,
        })

    def test_checkpoint_audit_accepts_jpeg2000_cross_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            model = SCOPE(num_codec_families=3, base_ch=8, bg_mode="residual")
            torch.save({
                "model": model.state_dict(),
                "codec_family_to_id": {"diff_jpeg": 0, "bmshj": 1, "bpg_like": 2},
                "args": {"base_ch": 8, "bg_mode": "residual"},
            }, checkpoint)
            configs = root / "ablation.yaml"
            configs.write_text(yaml.safe_dump({"experiments": [
                {"name": "jp2_condition_jpeg", "mode": "scope_jpeg2000",
                 "scope_ckpt": "model.pt", "codec_family": "block_dct"},
                {"name": "jp2_condition_learned", "mode": "scope_jpeg2000",
                 "scope_ckpt": "model.pt", "codec_family": "learned"},
                {"name": "jp2_condition_bpg", "mode": "scope_jpeg2000",
                 "scope_ckpt": "model.pt", "codec_family": "intra_smooth"},
            ]}))
            with contextlib.redirect_stdout(io.StringIO()):
                checked, failures = audit_checkpoints([configs], root=root)
            self.assertEqual((checked, failures), (1, 0))

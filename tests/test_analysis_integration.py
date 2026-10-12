"""Run the BDBR command end to end using synthetic saved summaries."""
from __future__ import annotations

import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scope.evaluation.ablation.cli import main


class AnalysisIntegrationTests(unittest.TestCase):
    def test_jpeg_ablation_and_cheng_baseline_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "eval_yolo"
            rates = [0.12, 0.20, 0.32, 0.48]
            for codec in ("jpeg", "cheng"):
                for method in ("baseline", "scope", "no_task") if codec == "jpeg" else ("baseline", "scope"):
                    for i, bitrate in enumerate(rates):
                        quality = 0.30 + 0.08 * i + (0.03 if method == "scope" else (0.01 if method == "no_task" else 0.0))
                        # Use normalized ablation names to exercise parsing of the real result folders.
                        suffix = f"{codec}_q{i+1}"
                        name = suffix if method == "baseline" else f"scope_{suffix}"
                        folder = source / name if method != "no_task" else root / "no_task_yolo" / name
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "summary.json").write_text(json.dumps({
                            "experiment": name, "AP": quality, "dataset_bpp": bitrate,
                            "avg_bpp_per_image": bitrate,
                            "codec_family": "learned" if codec == "cheng" else "block_dct",
                            "compressai_model": "cheng2020-attn" if codec == "cheng" else "",
                            "mode": "scope_compressai" if codec == "cheng" and method != "baseline" else "uniform",
                        }))
            out = root / "tables"
            argv = ["scope.evaluation.ablation_bdbr", "--detector-dir", f"full:yolo={source}",
                    f"no_task:yolo={root / 'no_task_yolo'}", "--out-dir", str(out),
                    "--table-methods", "full,no_task", "--allow-missing-methods",
                    "--interp", "linear", "--table-points", "3", "--min-bd-points", "2"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((out / "collected_results.csv").is_file())
            self.assertTrue((out / "interpolated_equal_rate_points.csv").is_file())
            self.assertTrue((out / "latex" / "bdbr_summary.tex").is_file())
            with (out / "pairwise_equal_rate_and_bdbr.csv").open() as f:
                pairs = list(csv.DictReader(f))
            cheng = [p for p in pairs if p["codec"] == "cheng"]
            self.assertTrue(cheng)
            self.assertTrue(all(p["test_method"] == "full" and p["reference_method"] == "baseline" for p in cheng), cheng)
            self.assertTrue(any(p["codec"] == "jpeg" and p["reference_method"] == "full" for p in pairs))
            self.assertTrue(any(p["codec"] == "cheng" and p["bd_status"] == "ok" for p in pairs), cheng)


if __name__ == "__main__":
    unittest.main()

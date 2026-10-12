from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ChengBaselineOnlyTests(unittest.TestCase):
    def test_cheng_only_compares_scope_with_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "results"
            source.mkdir()
            for quality in range(1, 6):
                name = f"cheng_q{quality}"
                rate = quality * 0.1
                accuracy = 0.2 + quality * 0.1
                for experiment, bpp in (
                    (name, rate),
                    (f"scope_{name}", rate * 0.8),
                ):
                    folder = source / experiment
                    folder.mkdir()
                    (folder / "summary.json").write_text(json.dumps({
                        "experiment": experiment,
                        "AP": accuracy,
                        "dataset_bpp": bpp,
                    }))

            out_dir = work / "tables"
            command = [
                sys.executable, "-m", "scope.evaluation.ablation_bdbr",
                "--detector-dir", f"full:yolo={source}",
                "--out-dir", str(out_dir), "--interp", "linear",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            with (out_dir / "pairwise_equal_rate_and_bdbr.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            cheng_rows = [row for row in rows if row["codec"] == "cheng"]
            self.assertEqual(len(cheng_rows), 1)
            self.assertEqual(cheng_rows[0]["test_method"], "full")
            self.assertEqual(cheng_rows[0]["reference_method"], "baseline")
            self.assertEqual(cheng_rows[0]["bd_status"], "ok")
            self.assertAlmostEqual(float(cheng_rows[0]["bdbr_percent"]), -20.0, places=6)

            table = (out_dir / "latex" / "interpolated_cheng.tex").read_text()
            self.assertIn("Baseline & SCOPE", table)
            self.assertNotIn("w/o task", table)
            self.assertNotIn("w/o bg", table)


if __name__ == "__main__":
    unittest.main()

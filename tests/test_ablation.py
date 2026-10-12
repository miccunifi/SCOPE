from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scope.evaluation.ablation.curves import compute_bd_rate
from scope.evaluation.ablation.results import strip_scope_prefix


ROOT = Path(__file__).resolve().parents[1]


class AblationAnalysisTests(unittest.TestCase):
    def test_scope_and_legacy_prefixes(self):
        for name in ('scope_jpeg_q10', 'preedit_jpeg_q10'):
            self.assertEqual(strip_scope_prefix(name, 'scope_,preedit_'), ('scope', 'jpeg_q10'))
        self.assertEqual(strip_scope_prefix('jpeg_q10', 'scope_,preedit_'), ('baseline', 'jpeg_q10'))

    def test_known_bd_rate(self):
        reference_rate = np.array([0.1, 0.2, 0.4, 0.8])
        accuracy = np.array([0.3, 0.45, 0.6, 0.75])
        result = compute_bd_rate(
            reference_rate, accuracy, reference_rate * 0.8, accuracy,
            interpolation='linear', dense_points=101, min_points=3,
        )
        self.assertEqual(result['bd_status'], 'ok')
        self.assertAlmostEqual(result['bdbr_percent'], -20.0, places=7)

    def test_analysis_pipeline_with_mixed_prefixes(self):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            directories = {key: work / key for key in ('full', 'no_task', 'no_bg')}
            for directory in directories.values():
                directory.mkdir()

            for level, rate in enumerate((0.1, 0.2, 0.3, 0.4), start=1):
                name = f'jpeg_q{level * 10}'
                baseline = directories['full'] / name
                baseline.mkdir()
                (baseline / 'summary.json').write_text(json.dumps({
                    'experiment': name, 'AP': 0.3 + 0.1 * level,
                    'dataset_bpp': rate,
                }))
                for method, gain in (('full', 0.04), ('no_task', 0.02), ('no_bg', 0.01)):
                    prefix = 'preedit_' if method == 'no_bg' else 'scope_'
                    exp_name = prefix + name
                    folder = directories[method] / exp_name
                    folder.mkdir()
                    (folder / 'summary.json').write_text(json.dumps({
                        'experiment': exp_name, 'AP': 0.3 + 0.1 * level + gain,
                        'dataset_bpp': rate * 1.02,
                    }))

            out_dir = work / 'output'
            command = [sys.executable, '-m', 'scope.evaluation.ablation',
                       '--detector-dir', *(f'{key}:yolo={path}' for key, path in directories.items()),
                       '--out-dir', str(out_dir), '--interp', 'linear']
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            with (out_dir / 'interpolated_equal_rate_points.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row['full'] and row['no_bg'] for row in rows))

            with (out_dir / 'pairwise_equal_rate_and_bdbr.csv').open() as stream:
                bdbr = list(csv.DictReader(stream))
            self.assertEqual(len(bdbr), 15)
            self.assertEqual(sum(row['bd_status'] == 'ok' for row in bdbr), 5)
            self.assertTrue((out_dir / 'latex' / 'bdbr_summary.tex').exists())


if __name__ == '__main__':
    unittest.main()

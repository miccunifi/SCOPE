from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from scope import SCOPE
from scope.evaluation.conditioning import (
    get_experiment_severity,
    infer_codec_family_from_experiment,
)
from scope.inference import apply_scope_bgr, load_scope_model


ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        model = SCOPE(num_codec_families=3, base_ch=8).eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            outputs = model(x, severity=torch.tensor([0.5]), codec_family=torch.tensor([1]))
        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[0].shape, x.shape)
        self.assertTrue(torch.isfinite(outputs[0]).all())

    def test_checkpoint_loading_and_inference(self):
        model = SCOPE(num_codec_families=2, base_ch=8).eval()
        mapping = {'diff_jpeg': 0, 'bmshj': 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'weights.pt'
            torch.save({
                'model': model.state_dict(),
                'codec_family_to_id': mapping,
                'args': {'base_ch': 8, 'attention_bias': -3.0, 'bg_mode': 'residual', 'bg_blur_kernel': 41},
            }, path)
            restored, restored_mapping = load_scope_model(str(path), torch.device('cpu'))
            self.assertEqual(restored_mapping, mapping)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            result = apply_scope_bgr(image, str(path), 'diff_jpeg', 0.25, 'cpu')
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)


class ConfigurationTests(unittest.TestCase):
    CONFIG_FILES = ('eval.yaml', 'ablation.yaml', 'timing.yaml')

    def test_unique_experiments_and_valid_severity(self):
        for filename in self.CONFIG_FILES:
            with self.subTest(filename=filename):
                cfg = yaml.safe_load((ROOT / 'configs' / filename).read_text())
                experiments = cfg['experiments']
                self.assertGreater(len(experiments), 0)
                self.assertEqual(len({exp['name'] for exp in experiments}), len(experiments))
                for exp in experiments:
                    self.assertIn('mode', exp)
                    self.assertIn('codec_family', exp)
                    if exp['mode'].startswith('scope_'):
                        self.assertIn('scope_ckpt', exp)
                        self.assertGreaterEqual(get_experiment_severity(exp), 0)
                        self.assertLessEqual(get_experiment_severity(exp), 1)
                        infer_codec_family_from_experiment(exp)

    def test_experiment_overrides_inherit_common_settings(self):
        for filename in self.CONFIG_FILES:
            cfg = yaml.safe_load((ROOT / 'configs' / filename).read_text())
            for experiment in cfg['experiments']:
                if experiment['mode'].startswith('scope_'):
                    for key, value in cfg['scope_common'].items():
                        self.assertEqual(experiment[key], value)


if __name__ == '__main__':
    unittest.main()

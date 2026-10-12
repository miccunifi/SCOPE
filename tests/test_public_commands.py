"""Regression checks for the public direct-file commands and docs paths."""
from __future__ import annotations

import ast
import runpy
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {
    "training/train_scope.py": "scope.train_scope",
    "evaluation/evaluate_yolo26x_codecs.py": "scope.evaluation.evaluate",
    "evaluation/evaluate_rtdetr_l_saved.py": "scope.evaluation.saved_rtdetr",
    "evaluation/evaluate_fasterrcnn_x101_fpn_saved.py": "scope.evaluation.saved_multi_detector",
    "evaluation/benchmark_scope.py": "scope.evaluation.timing",
    "evaluation/evaluate_categories.py": "scope.evaluation.rate_distortion",
    "evaluation/visualize_results.py": "scope.evaluation.visualize",
    "analysis/compute_bdbr.py": "scope.evaluation.ablation_bdbr",
}

class PublicEntryPointTests(unittest.TestCase):
    def test_public_wrapper_names_and_implementation_imports(self):
        for rel, module in PUBLIC.items():
            with self.subTest(script=rel):
                path = ROOT / rel
                self.assertTrue(path.is_file())
                tree = ast.parse(path.read_text())
                imports = [x for x in ast.walk(tree) if isinstance(x, ast.ImportFrom)]
                self.assertTrue(any(i.module == module and any(a.name == "main" for a in i.names) for i in imports), rel)
                self.assertTrue(any(isinstance(n, ast.If) and '__main__' in ast.unparse(n.test) for n in tree.body), rel)

    def test_markdown_command_paths_exist(self):
        pat = re.compile(r"\b(?:python(?:3)?|bash)\s+((?:training|evaluation|analysis|tools)/[\w.-]+\.(?:py|sh))")
        found = set()
        for rel in ("README.md", "docs/PIPELINE.md", "docs/TESTING.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for command in pat.findall(text):
                with self.subTest(document=rel, command=command):
                    self.assertTrue((ROOT / command).is_file())
                    found.add(command)
        self.assertTrue(PUBLIC.keys() <= found, f"Missing documentation commands: {PUBLIC.keys()-found}")

    def test_readme_embedded_figures_and_link_targets(self):
        import re
        for rel in ("README.md", "docs/PIPELINE.md"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
                with self.subTest(doc=rel, image=path):
                    self.assertTrue(((ROOT / rel).parent / path).is_file())

    def test_main_eval_config_is_dynamically_valid(self):
        config = yaml.safe_load((ROOT / "configs/eval.yaml").read_text())
        names = [x["name"] for x in config["experiments"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(names)
        # Direct JPEG2000 and zero-shot cross-family conditions belong in ablation.yaml.
        self.assertFalse(any("jp2" in n for n in names))

    def test_jpeg2000_ablation_only_uses_trained_families(self):
        config = yaml.safe_load((ROOT / "configs/ablation.yaml").read_text())
        allowed = {"diff_jpeg", "bmshj", "bpg_like", "block_dct", "learned", "intra_smooth"}
        scoped = [e for e in config["experiments"] if e["mode"].startswith("scope_")]
        self.assertTrue(scoped)
        for e in scoped:
            self.assertNotIn(e.get("codec_family", ""), ("jp2_like", "wavelet"))

    def test_fixed_fasterrcnn_detector_rejects_wrong_backend(self):
        # The generic implementation may need pycocotools. Replace it only for this CLI contract test.
        stub = types.ModuleType('scope.evaluation.saved_multi_detector')
        stub.main = lambda: None
        with patch.dict(sys.modules, {'scope.evaluation.saved_multi_detector': stub}):
            namespace = runpy.run_path(str(ROOT / 'evaluation/evaluate_fasterrcnn_x101_fpn_saved.py'), run_name='__test__')
            validate = namespace['_ensure_fixed_detector']
            valid = validate(['--source-eval-dir', 'out'])
            self.assertEqual(valid[-4:], ['--backend', 'detectron2', '--detector', 'fasterrcnn_x101_fpn'])
            with self.assertRaises(SystemExit):
                validate(['--backend', 'torchvision'])
            with self.assertRaises(SystemExit):
                validate(['--detector', 'retinanet'])

if __name__ == '__main__':
    unittest.main()

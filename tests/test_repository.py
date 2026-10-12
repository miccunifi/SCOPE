"""Dependency-free repository structure and configuration checks."""

from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = ("eval.yaml", "ablation.yaml", "timing.yaml")
ENTRY_POINTS = (
    "scope.train_scope", "scope.evaluation.evaluate", "scope.evaluation.timing",
    "scope.evaluation.saved_rtdetr", "scope.evaluation.saved_multi_detector",
    "scope.evaluation.rate_distortion", "scope.evaluation.visualize",
    "scope.evaluation.ablation_bdbr", "scope.evaluation.ablation",
)


def module_file(module: str) -> Path | None:
    stem = ROOT.joinpath(*module.split("."))
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


class RepositoryTests(unittest.TestCase):
    def test_all_python_sources_compile(self):
        for source_dir in (ROOT / "scope", ROOT / "training", ROOT / "evaluation", ROOT / "analysis", ROOT / "tools", ROOT / "tests"):
            for path in source_dir.rglob("*.py"):
                with self.subTest(path=path.relative_to(ROOT)):
                    compile(path.read_bytes(), str(path), "exec")

    def test_relative_imports_reference_real_modules(self):
        for path in (ROOT / "scope").rglob("*.py"):
            parts = path.relative_to(ROOT).with_suffix("").parts
            package = parts[:-1]
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue
                with self.subTest(path=path.relative_to(ROOT), line=node.lineno):
                    level_up = node.level - 1
                    self.assertLess(level_up, len(package))
                    target = package[:len(package) - level_up] + tuple(node.module.split(".") if node.module else ())
                    stem = ROOT.joinpath(*target)
                    self.assertTrue(stem.with_suffix(".py").is_file() or (stem / "__init__.py").is_file(), str(target))

    def test_absolute_scope_imports_reference_modules(self):
        for path in (ROOT / "scope").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    if module == "scope" or module.startswith("scope."):
                        with self.subTest(path=path.relative_to(ROOT), line=node.lineno, module=module):
                            self.assertIsNotNone(module_file(module))

    def test_entrypoint_modules_exist(self):
        for name in ENTRY_POINTS:
            with self.subTest(name=name):
                self.assertIsNotNone(module_file(name))

    def test_training_shell_matches_python_cli(self):
        shell = ROOT / "training/train_scope.sh"
        subprocess.run(["bash", "-n", str(shell)], check=True, capture_output=True)
        content = shell.read_text(encoding="utf-8")
        self.assertIn('TRAIN_SCRIPT="${TRAIN_SCRIPT:-${PROJECT_ROOT}/training/train_scope.py}"', content)
        self.assertIn('python "${TRAIN_SCRIPT}"', content)
        tree = ast.parse((ROOT / "scope/train_scope.py").read_text(encoding="utf-8"))
        recognized = {
            value.value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for value in node.args if isinstance(value, ast.Constant)
            and isinstance(value.value, str) and value.value.startswith("--")
        }
        referenced = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", content))
        self.assertFalse(referenced - recognized, f"Unknown CLI flags: {referenced - recognized}")

    def test_yaml_experiment_contract(self):
        allowed = {
            "uniform", "jpeg", "bpg", "jpeg2000", "jp2", "compressai",
            "scope_jpeg", "scope_bpg", "scope_jpeg2000", "scope_jp2", "scope_compressai",
        }
        for filename in CONFIG_FILES:
            with self.subTest(config=filename):
                cfg = yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
                self.assertGreater(len(cfg["experiments"]), 0)
                names = [item["name"] for item in cfg["experiments"]]
                self.assertEqual(len(names), len(set(names)))
                for exp in cfg["experiments"]:
                    self.assertIn(exp["mode"], allowed, exp["name"])
                    if exp["mode"].startswith("scope_"):
                        self.assertIn("scope_ckpt", exp)
                        for key, value in cfg["scope_common"].items():
                            self.assertEqual(exp[key], value, (filename, exp["name"], key))
                    severity = (exp.get("condition") or {}).get("severity")
                    if severity is not None:
                        self.assertGreaterEqual(float(severity), 0)
                        self.assertLessEqual(float(severity), 1)

    def test_training_and_evaluation_naming(self):
        self.assertTrue((ROOT / "training/train_scope.sh").is_file())
        self.assertTrue((ROOT / "training/train_scope.py").is_file())
        self.assertFalse((ROOT / "scope/train_scope.sh").exists())
        self.assertTrue((ROOT / "scope/evaluation/ablation/cli.py").exists())


if __name__ == "__main__":
    unittest.main()

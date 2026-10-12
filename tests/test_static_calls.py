"""Import-symbol and directly resolvable call-contract regression tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.audit_code import audit, module_index, signature_of, validate_direct_call


ROOT = Path(__file__).resolve().parents[1]


class StaticCallsTests(unittest.TestCase):
    def test_internal_imports_and_direct_calls(self):
        findings = audit(ROOT)
        self.assertFalse(findings, "\n" + "\n".join(str(f) for f in findings))

    def test_auditor_finds_missing_symbol_and_wrong_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scope").mkdir()
            (root / "scope/__init__.py").write_text("")
            (root / "scope/base.py").write_text("def foo(x, *, severity):\n    return x + severity\n")
            (root / "scope/consumer.py").write_text(
                "from .base import foo, missing\n"
                "def run():\n    return foo(1, incorrect=2)\n"
            )
            findings = audit(root)
            self.assertTrue(any(f.kind == "missing-symbol" and "missing" in f.detail for f in findings), findings)
            self.assertTrue(any(f.kind == "signature" and "incorrect" in f.detail for f in findings), findings)

    def test_auditor_handles_package_exports_and_local_shadowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scope").mkdir()
            (root / "scope/__init__.py").write_text("from .base import foo\n")
            (root / "scope/base.py").write_text("def foo(x):\n    return x\n")
            (root / "scope/consumer.py").write_text(
                "from scope import foo\n"
                "def run(foo):\n    return foo(1, 2, 3)\n"
            )
            self.assertFalse(audit(root))

    def test_all_sources_are_indexed(self):
        modules = module_index(ROOT)
        self.assertIn("scope.train_scope", modules)
        self.assertIn("scope.evaluation.ablation.cli", modules)
        self.assertIn("tools.validate_environment", modules)


if __name__ == "__main__":
    unittest.main()

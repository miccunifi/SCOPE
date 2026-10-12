"""Exercise the actual training shell script without starting GPU training."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scope.train_scope import build_argparser

ROOT = Path(__file__).resolve().parents[1]


class ShellIntegrationTests(unittest.TestCase):
    def test_training_shell_passes_real_parser_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "training").mkdir()
            (root / "training/train_scope.py").write_text("# test placeholder\n")
            for name in ("train", "validation"):
                (root / name).mkdir()
                (root / f"{name}.json").write_text('{"images": []}')
            capture = root / "captured_args.txt"
            python = root / "bin/python"
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SCOPE_TEST_CAPTURE"\n')
            python.chmod(0o755)
            env = dict(os.environ, PROJECT_ROOT=str(root), TRAIN_IMAGES=str(root / "train"),
                       TRAIN_JSON=str(root / "train.json"), VAL_IMAGES=str(root / "validation"),
                       VAL_JSON=str(root / "validation.json"), OUT_DIR=str(root / "outputs"),
                       SCOPE_TEST_CAPTURE=str(capture), PATH=str(root / "bin") + os.pathsep + os.environ.get("PATH", ""))
            run = subprocess.run(["bash", str(ROOT / "training/train_scope.sh")],
                                 cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            argv = capture.read_text().splitlines()
            self.assertEqual(argv[0], str(root / "training/train_scope.py"))
            args = build_argparser().parse_args(argv[1:])
            self.assertEqual(Path(args.images_dir), root / "train")
            self.assertEqual(Path(args.annotations_json), root / "train.json")
            self.assertEqual(Path(args.output_dir), root / "outputs")
            self.assertTrue(args.use_real_diffjpeg)
            self.assertTrue(args.use_real_bmshj)
            self.assertTrue((root / "outputs").is_dir())


if __name__ == "__main__":
    unittest.main()

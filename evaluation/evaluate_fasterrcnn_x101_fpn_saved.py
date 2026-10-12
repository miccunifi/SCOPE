#!/usr/bin/env python3
"""Evaluate saved reconstructions with Faster R-CNN ResNeXt-101-32x8d-FPN.

Uses the Detectron2 COCO model used in the SCOPE paper. The generic
saved-image evaluation implementation lives in
``scope.evaluation.saved_multi_detector``.

Run from the SCOPE repository root:
    python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py --help
    python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py \
      --source-eval-dir outputs_scope_yolo \
      --images-dir datasets/coco-2017/validation/data \
      --annotations-json datasets/coco-2017/validation/labels.json \
      --output-dir outputs_scope_fasterrcnn

The script supplies --backend=detectron2 and --detector=fasterrcnn_x101_fpn.
Optional --detectron2-weights overrides the model-zoo weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scope.evaluation.saved_multi_detector import main as _evaluate_saved


_FIXED_OPTIONS = {
    "--backend": "detectron2",
    "--detector": "fasterrcnn_x101_fpn",
}


def _ensure_fixed_detector(args: list[str]) -> list[str]:
    """Fill in the paper detector defaults; reject contradictory options."""
    final_args = list(args)
    for option, expected in _FIXED_OPTIONS.items():
        explicit_values: list[str] = []
        for idx, arg in enumerate(final_args):
            if arg == option:
                if idx + 1 >= len(final_args) or final_args[idx + 1].startswith("--"):
                    raise SystemExit(f"{option} requires a value (expected {expected}).")
                explicit_values.append(final_args[idx + 1])
            elif arg.startswith(option + "="):
                explicit_values.append(arg.split("=", 1)[1])

        for actual in explicit_values:
            if actual != expected:
                raise SystemExit(
                    f"{option}={actual!r} does not match this script's detector. "
                    f"Use {option}={expected!r}, or call the generic evaluator directly."
                )
        if not explicit_values:
            final_args.extend([option, expected])
    return final_args


def main() -> None:
    sys.argv[1:] = _ensure_fixed_detector(sys.argv[1:])
    _evaluate_saved()


if __name__ == "__main__":
    main()

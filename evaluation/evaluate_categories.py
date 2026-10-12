#!/usr/bin/env python3
"""Evaluate COCO per-category and severity metrics.

Run from the SCOPE repository root:
    python evaluation/evaluate_categories.py --help

This is a named CLI entry point. The implementation remains in scope.evaluation.rate_distortion.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct execution adds evaluation/ to sys.path, not the project root.
# Make the sibling `scope` package importable without requiring installation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scope.evaluation.rate_distortion import main


if __name__ == "__main__":
    main()

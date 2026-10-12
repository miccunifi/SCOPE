#!/usr/bin/env python3
"""Benchmark SCOPE pre-editing speed.

Run from the SCOPE repository root:
    python evaluation/benchmark_scope.py --help

This is a named CLI entry point. The implementation remains in scope.evaluation.timing.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct execution adds evaluation/ to sys.path, not the project root.
# Make the sibling `scope` package importable without requiring installation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scope.evaluation.timing import main


if __name__ == "__main__":
    main()

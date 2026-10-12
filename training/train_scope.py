#!/usr/bin/env python3
"""Train SCOPE from scratch using the full CLI implemented in scope.train_scope.

Direct usage from the repository root:
    python training/train_scope.py --help
    bash training/train_scope.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scope.train_scope import main

if __name__ == "__main__":
    main()

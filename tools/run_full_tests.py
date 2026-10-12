"""Unified SCOPE tests and optional research-machine preflight."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import validate_environment
    from tools.audit_code import audit
    from tools.check_runtime import audit_checkpoints, audit_entrypoints, audit_imports
else:
    from . import validate_environment
    from .audit_code import audit
    from .check_runtime import audit_checkpoints, audit_entrypoints, audit_imports


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SCOPE regression and integration tests.")
    parser.add_argument("--environment", action="store_true", help="Check real dataset, weight, output and codec paths.")
    parser.add_argument("--imports", action="store_true", help="Import every SCOPE module in this environment.")
    parser.add_argument("--entrypoints", action="store_true", help="Run all Python entrypoints with --help.")
    parser.add_argument("--checkpoints", action="store_true", help="Load real evaluation checkpoints on CPU and validate codec mappings.")
    parser.add_argument("--strict-external", action="store_true", help="Missing external dependencies become failures.")
    parser.add_argument("--check-images", type=int, default=20, metavar="N", help="Number of COCO image files to check; -1 for all.")
    args = parser.parse_args(argv)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"), MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS", "2"))
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env=env, cwd=validate_environment.PROJECT_ROOT)
    failures = int(result.returncode != 0)
    findings = audit()
    for finding in findings:
        print(f"[FAIL] {finding.file}:{finding.line} {finding.kind}: {finding.detail}")
    print(f"Static call/import audit: {len(findings)} issue(s)")
    failures += int(bool(findings))
    if args.imports:
        failures += audit_imports(args.strict_external)[2]
    if args.entrypoints:
        failures += audit_entrypoints(args.strict_external)[2]
    if args.checkpoints:
        failures += audit_checkpoints(list(validate_environment.DEFAULT_CONFIGS.values()))[1]
    if args.environment:
        opts = validate_environment.parse_args(["--mode", "all", "--check-images", str(args.check_images)])
        failures += validate_environment.run(opts).summary()
    print(f"SCOPE full checks: {'FAILED' if failures else 'PASSED'} ({failures} failing section(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

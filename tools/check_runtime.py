"""Runtime import, CLI and checkpoint validation for the SCOPE repository."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .audit_code import ROOT, module_index
from .validate_environment import DEFAULT_CONFIGS, SCOPE_MODES, resolve_local_path

ENTRYPOINTS = (
    "training/train_scope.py",
    "evaluation/evaluate_yolo26x_codecs.py",
    "evaluation/evaluate_rtdetr_l_saved.py",
    "evaluation/evaluate_fasterrcnn_x101_fpn_saved.py",
    "evaluation/benchmark_scope.py",
    "evaluation/evaluate_categories.py",
    "evaluation/visualize_results.py",
    "analysis/compute_bdbr.py",
)



def audit_imports(strict_external: bool = False) -> tuple[int, int, int]:
    passed = skipped = failed = 0
    for name in sorted(module_index(ROOT)):
        if name.startswith("tests.") or name.startswith("tools."):
            continue
        try:
            importlib.import_module(name)
            passed += 1
            print(f"[ OK ] {name}")
        except ModuleNotFoundError as exc:
            if exc.name and not (exc.name == "scope" or exc.name.startswith("scope.")):
                skipped += 1
                print(f"[{'FAIL' if strict_external else 'SKIP'}] {name}: missing external module {exc.name}")
                if strict_external:
                    failed += 1
            else:
                failed += 1
                print(f"[FAIL] {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"Imports: {passed} OK, {skipped} missing external dependency, {failed} failure(s)")
    return passed, skipped, failed


def audit_entrypoints(strict_external: bool = False) -> tuple[int, int, int]:
    passed = skipped = failed = 0
    for name in ENTRYPOINTS:
        try:
            proc = subprocess.run([sys.executable, str(ROOT / name), "--help"], cwd=ROOT,
                                  text=True, capture_output=True, timeout=40)
            if proc.returncode == 0 and "usage:" in proc.stdout.lower():
                passed += 1
                print(f"[ OK ] {name} --help")
            elif "No module named" in proc.stderr or "ModuleNotFoundError" in proc.stderr:
                match = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
                if "No module named 'scope" in match or 'No module named "scope' in match:
                    failed += 1
                    print(f"[FAIL] {name}: {match}")
                else:
                    skipped += 1
                    if strict_external:
                        failed += 1
                    print(f"[{'FAIL' if strict_external else 'SKIP'}] {name}: {match}")
            else:
                failed += 1
                print(f"[FAIL] {name}: exit {proc.returncode}: {proc.stderr[-600:]}")
        except subprocess.TimeoutExpired:
            failed += 1
            print(f"[FAIL] {name}: --help timed out after 40s")
    print(f"Entrypoints: {passed} OK, {skipped} missing dependency, {failed} failure(s)")
    return passed, skipped, failed


def audit_checkpoints(paths: list[Path], *, root: Path = ROOT) -> tuple[int, int]:
    import torch
    from scope.inference import load_scope_model
    from scope.evaluation.conditioning import infer_codec_family_from_experiment

    checked = failures = 0
    seen = set()
    for config in paths:
        config = resolve_local_path(config, root)
        if not config.is_file():
            print(f"[FAIL] config not found: {config}")
            failures += 1
            continue
        try:
            data: dict[str, Any] = yaml.safe_load(config.read_text())
            for exp in data["experiments"]:
                if exp.get("mode") not in SCOPE_MODES:
                    continue
                checkpoint = exp.get("scope_ckpt") or data.get("scope_ckpt") or (data.get("scope_common") or {}).get("scope_ckpt")
                if not checkpoint:
                    print(f"[FAIL] {config.name}/{exp['name']}: no checkpoint")
                    failures += 1
                    continue
                checkpoint = resolve_local_path(checkpoint, root)
                if not checkpoint.is_file():
                    print(f"[FAIL] {config.name}/{exp['name']}: missing {checkpoint}")
                    failures += 1
                    continue
                settings = {k: exp[k] for k in ("base_ch", "family_emb_dim", "cond_dim", "bg_mode", "max_delta_obj", "max_delta_bg", "bg_blur_kernel", "bg_lowres_scale", "attention_bias") if k in exp}
                key = (str(checkpoint), json.dumps(settings, sort_keys=True))
                if key not in seen:
                    model, mapping = load_scope_model(str(checkpoint), torch.device("cpu"), exp)
                    model.eval()
                    seen.add(key)
                    checked += 1
                else:
                    model, mapping = load_scope_model(str(checkpoint), torch.device("cpu"), exp)
                family = infer_codec_family_from_experiment(exp)
                if family not in mapping:
                    hint = (
                        " JPEG2000 was unseen by this checkpoint; use "
                        "block_dct, learned, or intra_smooth in the ablation config."
                        if family == "jp2_like" and exp.get("mode") in {"scope_jpeg2000", "scope_jp2"}
                        else ""
                    )
                    print(f"[FAIL] {config.name}/{exp['name']}: family {family!r} not in checkpoint: {list(mapping)}.{hint}")
                    failures += 1
            print(f"[ OK ] {config.name}: configuration processed")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {config}: {type(exc).__name__}: {exc}")
    print(f"Checkpoints: {checked} unique model configurations loaded, {failures} failure(s)")
    return checked, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit runtime imports, entrypoints and SCOPE checkpoint compatibility.")
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--entrypoints", action="store_true")
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--strict-external", action="store_true", help="Treat missing external Python dependencies as errors.")
    parser.add_argument("--config", action="append", type=Path, default=[], help="YAML for checkpoint compatibility (repeatable).")
    args = parser.parse_args(argv)
    failures = 0
    if not any((args.imports, args.entrypoints, args.checkpoints)):
        parser.error("Select --imports, --entrypoints or --checkpoints")
    if args.imports:
        failures += audit_imports(args.strict_external)[2]
    if args.entrypoints:
        failures += audit_entrypoints(args.strict_external)[2]
    if args.checkpoints:
        configs = args.config or list(DEFAULT_CONFIGS.values())
        failures += audit_checkpoints(configs)[1]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preflight validation for the SCOPE training and evaluation workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "eval": PROJECT_ROOT / "configs/eval.yaml",
    "ablation": PROJECT_ROOT / "configs/ablation.yaml",
    "timing": PROJECT_ROOT / "configs/timing.yaml",
}
SCOPE_MODES = {"scope_jpeg", "scope_bpg", "scope_jpeg2000", "scope_jp2", "scope_compressai"}
SUPPORTED_MODES = SCOPE_MODES | {"uniform", "jpeg", "bpg", "jpeg2000", "jp2", "compressai"}
CODEC_BINARIES = {
    "bpg": (("bpg_encoder_bin", "bpgenc"), ("bpg_decoder_bin", "bpgdec")),
    "jp2": (("jp2_encoder_bin", "opj_compress"), ("jp2_decoder_bin", "opj_decompress")),
}
BASE_PACKAGES = ("torch", "numpy", "cv2", "yaml", "PIL", "tqdm")


@dataclass(frozen=True)
class Check:
    status: str
    subject: str
    detail: str


class Reporter:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checks: list[Check] = []
        self._seen: set[tuple[str, str, str]] = set()

    def add(self, status: str, subject: str, detail: str) -> None:
        item = Check(status, subject, str(detail))
        key = (item.status, item.subject, item.detail)
        if key not in self._seen:
            self.checks.append(item)
            self._seen.add(key)

    def ok(self, subject: str, detail: str) -> None:
        self.add("OK", subject, detail)

    def fail(self, subject: str, detail: str) -> None:
        self.add("FAIL", subject, detail)

    def warn(self, subject: str, detail: str) -> None:
        self.add("WARN", subject, detail)

    def check_path(self, subject: str, path: Path, kind: str) -> bool:
        valid = (path.is_file() if kind == "file" else path.is_dir())
        (self.ok if valid else self.fail)(subject, f"{path} ({kind})")
        return valid

    def check_package(self, name: str, required: bool = True) -> bool:
        try:
            installed = importlib.util.find_spec(name) is not None
        except (ValueError, ImportError, ModuleNotFoundError):
            installed = False
        if installed:
            self.ok(f"package:{name}", "available")
        else:
            (self.fail if required else self.warn)(f"package:{name}", "not installed")
        return installed

    def summary(self, show_ok: bool = False) -> int:
        for check in self.checks:
            if check.status != "OK" or show_ok:
                print(f"[{check.status:4}] {check.subject}: {check.detail}")
        counts = {status: sum(x.status == status for x in self.checks) for status in ("OK", "WARN", "FAIL")}
        print(f"Preflight: {counts['OK']} OK, {counts['WARN']} warning(s), {counts['FAIL']} failure(s)")
        return 1 if counts["FAIL"] else 0


def resolve_local_path(value: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else root / path


def find_executable(value: str, root: Path = PROJECT_ROOT) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if Path(expanded).is_absolute() or "/" in expanded or "\\" in expanded:
        candidate = resolve_local_path(expanded, root)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
    return shutil.which(expanded)


def check_executable(report: Reporter, label: str, value: str) -> None:
    located = find_executable(value, report.root)
    if located:
        report.ok(label, located)
    else:
        report.fail(label, f"{value!r} is not an executable on PATH or at the configured path")


def check_structure(report: Reporter) -> None:
    mandatory = (
        "README.md", "requirements.txt", "environment.yml", "scope/__init__.py",
        "scope/train_scope.py", "training/train_scope.py", "training/train_scope.sh", "scope/training.py", "scope/model.py",
        "scope/data.py", "scope/codecs.py", "scope/conditioning.py", "scope/losses.py",
        "scope/teachers.py", "scope/inference.py", "scope/evaluation/evaluate.py",
        "scope/evaluation/timing.py", "scope/evaluation/saved_rtdetr.py",
        "scope/evaluation/saved_multi_detector.py", "scope/evaluation/rate_distortion.py",
        "scope/evaluation/ablation_bdbr.py", "scope/evaluation/ablation/cli.py",
        "evaluation/evaluate_yolo26x_codecs.py",
        "evaluation/evaluate_rtdetr_l_saved.py",
        "evaluation/evaluate_fasterrcnn_x101_fpn_saved.py",
        "evaluation/benchmark_scope.py",
        "analysis/compute_bdbr.py",
    )
    for name in mandatory:
        report.check_path(f"source:{name}", report.root / name, "file")
    for name, path in DEFAULT_CONFIGS.items():
        report.check_path(f"config:{name}", report.root / "configs" / path.name, "file")

    shell = report.root / "training/train_scope.sh"
    if shell.is_file():
        if not find_executable("bash", report.root):
            report.fail("bash", "bash not found")
        else:
            result = subprocess.run(["bash", "-n", str(shell)], capture_output=True, text=True)
            (report.ok if result.returncode == 0 else report.fail)(
                "training-shell-syntax", "valid" if result.returncode == 0 else result.stderr.strip()
            )


def check_dataset(
    report: Reporter,
    images_dir: str,
    annotations_json: str,
    name: str,
    check_images: int = 0,
) -> None:
    image_root = resolve_local_path(images_dir, report.root)
    annotations = resolve_local_path(annotations_json, report.root)
    images_exist = report.check_path(f"{name}.images", image_root, "dir")
    if not report.check_path(f"{name}.annotations", annotations, "file"):
        return
    try:
        with annotations.open(encoding="utf-8") as stream:
            content = json.load(stream)
        images = content["images"]
        if not isinstance(images, list):
            raise ValueError("'images' must be a list")
        if any("id" not in row or "file_name" not in row for row in images):
            raise ValueError("images must contain id and file_name")
        image_ids = [row["id"] for row in images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("duplicate image ids")
        report.ok(f"{name}.annotations-format", f"{len(images)} COCO image entries")
        if check_images and images_exist:
            sample = images if check_images < 0 else images[:check_images]
            missing = [row["file_name"] for row in sample if not (image_root / row["file_name"]).is_file()]
            if missing:
                report.fail(f"{name}.image-files", f"{len(missing)}/{len(sample)} missing; examples: {missing[:5]}")
            else:
                report.ok(f"{name}.image-files", f"{len(sample)} files checked")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        report.fail(f"{name}.annotations-format", str(exc))


def check_output(report: Reporter, path: Path, label: str) -> None:
    path = path.resolve()
    if path.exists() and not path.is_dir():
        report.fail(label, f"output path exists but is not a directory: {path}")
        return
    parent = path
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        report.fail(label, f"cannot create output in {parent}")
    else:
        report.ok(label, f"can create/write under {path} (permission estimate; no files created)")


def check_checkpoint(report: Reporter, path_value: str, label: str) -> None:
    report.check_path(label, resolve_local_path(path_value, report.root), "file")


def check_config(
    report: Reporter,
    config_path: Path,
    *,
    purpose: str,
    check_images: int = 0,
    check_packages: bool = True,
) -> None:
    config_path = resolve_local_path(config_path, report.root)
    if not report.check_path("config", config_path, "file"):
        return
    try:
        config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("root must be a mapping")
        experiments = config["experiments"]
        if not isinstance(experiments, list) or not experiments:
            raise ValueError("experiments must be a non-empty list")
        names = [item["name"] for item in experiments]
        if len(names) != len(set(names)):
            raise ValueError("experiment names must be unique")
        for item in experiments:
            mode = item["mode"]
            if mode not in SUPPORTED_MODES:
                raise ValueError(f"{item['name']}: unsupported mode {mode!r}")
            if mode in SCOPE_MODES and not (item.get("scope_ckpt") or config.get("scope_ckpt") or config.get("scope_common", {}).get("scope_ckpt")):
                raise ValueError(f"{item['name']}: missing scope_ckpt")
            severity = (item.get("condition") or {}).get("severity", item.get("severity"))
            if severity is not None and not 0 <= float(severity) <= 1:
                raise ValueError(f"{item['name']}: severity outside [0, 1]")
        report.ok("config-format", f"{config_path}: {len(experiments)} experiments")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        report.fail("config-format", f"{config_path}: {exc}")
        return

    check_output(report, resolve_local_path(config.get("output_dir", "outputs_scope"), report.root), "output-dir")
    if purpose == "timing":
        active = [item for item in experiments if item["mode"] in SCOPE_MODES]
    else:
        active = experiments
        dataset = config.get("dataset") or {}
        if "images_dir" not in dataset or "annotations_json" not in dataset:
            report.fail("dataset", "missing dataset.images_dir or dataset.annotations_json")
        else:
            check_dataset(report, dataset["images_dir"], dataset["annotations_json"], "validation", check_images)
        detector = (config.get("models") or {}).get("server_model")
        if not detector:
            report.fail("detector", "missing models.server_model")
        else:
            check_checkpoint(report, detector, "detector-weights")

    if purpose == "timing":
        dataset = config.get("dataset") or {}
        if "images_dir" not in dataset or "annotations_json" not in dataset:
            report.fail("dataset", "missing dataset.images_dir or dataset.annotations_json")
        else:
            check_dataset(report, dataset["images_dir"], dataset["annotations_json"], "timing", check_images)

    ckpts = {str(item.get("scope_ckpt") or config.get("scope_ckpt") or (config.get("scope_common") or {}).get("scope_ckpt")) for item in active if item["mode"] in SCOPE_MODES}
    for ckpt in sorted(ckpts):
        check_checkpoint(report, ckpt, "scope-checkpoint")

    if purpose != "timing":
        codec_specs = (
            ({"bpg", "scope_bpg"}, CODEC_BINARIES["bpg"]),
            ({"jpeg2000", "jp2", "scope_jpeg2000", "scope_jp2"}, CODEC_BINARIES["jp2"]),
        )
        for modes, binaries in codec_specs:
            for key, fallback in binaries:
                configured = {str(item.get(key, config.get(key, fallback))) for item in active if item["mode"] in modes}
                for executable in sorted(configured):
                    check_executable(report, f"codec:{key}", executable)

    if check_packages:
        for package in BASE_PACKAGES:
            report.check_package(package)
        if purpose != "timing":
            report.check_package("ultralytics")
            report.check_package("pycocotools")
        if any(item["mode"] in {"compressai", "scope_compressai"} for item in active) and purpose != "timing":
            report.check_package("compressai")

    if str(config.get("device", "")).startswith("cuda") and check_packages and report.check_package("torch", required=True):
        try:
            import torch
            (report.ok if torch.cuda.is_available() else report.fail)("cuda", "available" if torch.cuda.is_available() else "CUDA requested but unavailable")
        except Exception as exc:
            report.fail("cuda", str(exc))


def check_result_directories(report: Reporter, roots: list[Path], *, purpose: str) -> None:
    for location in roots:
        root = resolve_local_path(location, report.root)
        if not report.check_path(f"{purpose}.results", root, "dir"):
            continue
        experiments = sorted(path for path in root.iterdir() if path.is_dir() and (path / "summary.json").is_file())
        if not experiments:
            report.fail(f"{purpose}.experiments", f"no subdirectories containing summary.json under {root}")
            continue
        report.ok(f"{purpose}.experiments", f"{root}: {len(experiments)} experiment summaries")
        for exp in experiments:
            if purpose == "saved":
                report.check_path(f"{exp.name}.reconstructed", exp / "reconstructed", "dir")
                report.check_path(f"{exp.name}.bandwidth", exp / "bandwidth.csv", "file")
            else:
                try:
                    data = json.loads((exp / "summary.json").read_text(encoding="utf-8"))
                    if not isinstance(data, dict) or "AP" not in data:
                        raise ValueError("summary.json must be a mapping containing AP")
                    has_rate = any(key in data for key in ("dataset_bpp", "avg_bpp_per_image", "avg_kb_per_image"))
                    if not has_rate and not (exp / "bandwidth.csv").is_file():
                        raise ValueError("no rate statistics in summary.json and bandwidth.csv missing")
                    report.ok(f"{exp.name}.analysis", "summary/bitrate metadata available")
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    report.fail(f"{exp.name}.analysis", str(exc))


def check_training(report: Reporter, check_images: int = 0, check_packages: bool = True) -> None:
    train_images = os.environ.get("TRAIN_IMAGES", str(report.root / "datasets/coco-2017/train/data"))
    train_json = os.environ.get("TRAIN_JSON", str(report.root / "datasets/coco-2017/train/labels.json"))
    val_images = os.environ.get("VAL_IMAGES", str(report.root / "datasets/coco-2017/validation/data"))
    val_json = os.environ.get("VAL_JSON", str(report.root / "datasets/coco-2017/validation/labels.json"))
    out_dir = os.environ.get("OUT_DIR", str(report.root / "runs"))
    check_dataset(report, train_images, train_json, "train", check_images)
    check_dataset(report, val_images, val_json, "train-validation", check_images)
    check_output(report, resolve_local_path(out_dir, report.root), "train-output")
    if os.environ.get("D2_WEIGHTS"):
        weights = os.environ["D2_WEIGHTS"]
        if weights.startswith(("http://", "https://")):
            report.warn("detectron2-weights", "remote URL cannot be validated offline")
        else:
            check_checkpoint(report, weights, "detectron2-weights")
    d2_config = os.environ.get("D2_CONFIG", "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    if d2_config.startswith(("http://", "https://")):
        report.warn("detectron2-config", "remote config URL cannot be validated offline")
    elif not d2_config.startswith(("COCO-Detection/", "COCO-InstanceSegmentation/", "COCO-Keypoints/")):
        report.check_path("detectron2-config", resolve_local_path(d2_config, report.root), "file")
    if check_packages:
        for package in (*BASE_PACKAGES, "torchvision", "kornia", "compressai", "detectron2"):
            report.check_package(package)
        try:
            diffjpeg_found = importlib.util.find_spec("diff_jpeg") or importlib.util.find_spec("DiffJPEG")
        except (ValueError, ImportError, ModuleNotFoundError):
            diffjpeg_found = False
        if not diffjpeg_found:
            report.fail("package:Diff-JPEG", "required by train_scope.sh --use-real-diffjpeg; install necla-ml/Diff-JPEG")
        else:
            report.ok("package:Diff-JPEG", "candidate import module found (API not exercised)")
        try:
            import torch
            (report.ok if torch.cuda.is_available() else report.warn)("training-cuda", "available" if torch.cuda.is_available() else "unavailable; training may be very slow")
        except Exception:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SCOPE source, configurations, runtime paths, and dependencies.")
    parser.add_argument("--mode", choices=("static", "train", "eval", "timing", "saved", "analysis", "all"), default="static")
    parser.add_argument("--config", action="append", default=[], help="YAML path; may be repeated (eval/timing).")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PROJECT_ROOT", PROJECT_ROOT)), help="Project root for relative config paths.")
    parser.add_argument("--source-eval-dir", action="append", type=Path, default=[], help="Completed evaluation output directory; repeat for more detectors/ablations (saved/analysis modes).")
    parser.add_argument("--check-images", type=int, default=0, metavar="N", help="Check the first N COCO images; -1 checks all; 0 checks only annotations.")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip optional runtime package/CUDA checks.")
    parser.add_argument("--show-ok", action="store_true", help="Display successful checks too.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Reporter:
    report = Reporter(args.root)
    check_structure(report)
    if args.mode in {"train", "all"}:
        check_training(report, args.check_images, not args.skip_dependencies)
    if args.mode in {"saved", "analysis"} or (args.mode == "all" and args.source_eval_dir):
        if not args.source_eval_dir:
            report.fail("source-eval-dir", "provide at least one --source-eval-dir")
        else:
            check_result_directories(report, args.source_eval_dir, purpose="saved" if args.mode == "saved" else "analysis")
    if args.mode in {"eval", "timing", "all"}:
        if args.config:
            config_paths = [resolve_local_path(name, report.root) for name in args.config]
        elif args.mode == "timing":
            config_paths = [report.root / "configs/timing.yaml"]
        else:
            config_paths = [report.root / "configs/eval.yaml", report.root / "configs/ablation.yaml"]
            if args.mode == "all":
                config_paths.append(report.root / "configs/timing.yaml")
        for path in config_paths:
            purpose = "timing" if path.name == "timing.yaml" or args.mode == "timing" else "eval"
            print(f"Checking {purpose}: {path}")
            check_config(report, path, purpose=purpose, check_images=args.check_images, check_packages=not args.skip_dependencies)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    return report.summary(show_ok=args.show_ok)


if __name__ == "__main__":
    sys.exit(main())

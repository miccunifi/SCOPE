"""Run selected real compression/detection experiments on a small COCO subset."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .validate_environment import PROJECT_ROOT, resolve_local_path


def prepare_subset(config_path: Path, experiment_names: list[str], limit: int, tmp: Path,
                   output_dir: Path, root: Path = PROJECT_ROOT) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("experiments"), list):
        raise ValueError(f"Invalid evaluation YAML: {config_path}")
    experiments = {e["name"]: e for e in config["experiments"]}
    missing = set(experiment_names) - experiments.keys()
    if missing:
        raise ValueError(f"Unknown experiment(s): {', '.join(sorted(missing))}")
    annotation_path = resolve_local_path(config["dataset"]["annotations_json"], root)
    image_root = resolve_local_path(config["dataset"]["images_dir"], root)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    sampled = coco["images"][:limit]
    if not sampled:
        raise ValueError("No COCO images in annotation file")
    for row in sampled:
        path = image_root / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = {record["id"] for record in sampled}
    subset = dict(coco)
    subset["images"] = sampled
    subset["annotations"] = [a for a in coco.get("annotations", []) if a["image_id"] in ids]
    subset.setdefault("info", {})
    subset.setdefault("licenses", [])
    subset_path = tmp / "subset_coco.json"
    subset_path.write_text(json.dumps(subset))
    config["dataset"]["images_dir"] = str(image_root)
    config["dataset"]["annotations_json"] = str(subset_path)
    config["output_dir"] = str(output_dir)
    config["experiments"] = [exp for exp in config["experiments"] if exp["name"] in experiment_names]
    config["models"]["server_model"] = str(resolve_local_path(config["models"]["server_model"], root))
    for exp in config["experiments"]:
        if exp["mode"].startswith("scope_") and "scope_ckpt" in exp:
            exp["scope_ckpt"] = str(resolve_local_path(exp["scope_ckpt"], root))
    config_file = tmp / "smoke_config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real compression + detector + COCO smoke test (subset metrics are not publishable).")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--experiments", nargs="+", required=True, help="Exact experiment names in the YAML file.")
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_scope_smoke"))
    args = parser.parse_args(argv)
    if args.images <= 0:
        parser.error("--images must be positive")
    config = resolve_local_path(args.config)
    destination = resolve_local_path(args.output_dir)
    with tempfile.TemporaryDirectory(prefix="scope_real_smoke_") as tmp:
        smoke_cfg = prepare_subset(config, args.experiments, args.images, Path(tmp), destination)
        result = subprocess.run([sys.executable, "-m", "scope.evaluation.evaluate", "--config", str(smoke_cfg)],
                                cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print(f"Real evaluation smoke test completed: {destination}")
        print("Subset AP is diagnostic only; do not use it in the paper.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

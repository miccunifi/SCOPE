# 🔎 SCOPE: Manual validation and reproducibility

This document describes checks that can be run **without** the former `tests/` and `tools/` directories. The released layout does **not** contain `tools.run_full_tests`, `tools.validate_environment`, an automated test suite, or an exhaustive import/function-call auditor.

These are practical checks for the training and evaluation pipeline; they are **not a substitute for a real end-to-end run** using the intended model weights and data.

All commands assume the **SCOPE repository root** is the working directory.

## ✅ 1. Check code and shell syntax

```bash
# Compile all package Python files without running models
python -m compileall -q scope

# Check the training launcher's Bash syntax
bash -n scope/train_scope.sh
```

Both should finish without errors. Compilation checks Python syntax only: it **cannot** guarantee that imported functions exist or that numerical operations work.

Check package entry points after installing the runtime dependencies:

```bash
python -m scope.train_scope --help
python -m scope.evaluation.evaluate --help
python -m scope.evaluation.timing --help
python -m scope.evaluation.ablation_bdbr --help
python -m scope.evaluation.saved_rtdetr --help
python -m scope.evaluation.saved_multi_detector --help
```

A `--help` failure may reflect a missing external package during module import; it does not necessarily mean the command-line arguments are incorrect.

## 📁 2. Check input paths and executables

The defaults in `scope/train_scope.sh` and values in `configs/*.yaml` refer to external assets that are not distributed with the repository.

```bash
# COCO inputs
ls datasets/coco-2017/train/labels.json
ls datasets/coco-2017/validation/labels.json
ls datasets/coco-2017/train/data | head
ls datasets/coco-2017/validation/data | head

# Default evaluation detector
ls models/yolo26x.pt

# External codecs used by BPG and JPEG2000
command -v bpgenc
command -v bpgdec
command -v opj_compress
command -v opj_decompress
```

If a path differs, update the relevant YAML value or the training environment variable. A missing codec executable only prevents experiments that depend on that codec. The training teacher's Detectron2 model-zoo config name is not necessarily a local file path.

For the configured SCOPE checkpoint, inspect `scope_common.scope_ckpt` in the YAML. The old `runs/preedit_*/best_val.pt` naming is valid; what matters is that the **actual file exists and is compatible with the model**. Avoid renaming a checkpoint to make its path look more consistent.

## 🧩 3. Validate active YAML experiments

The following standalone Python snippet uses the already-required PyYAML dependency. It verifies that the configuration files parse, active experiment names are unique, required experiment keys exist, and explicitly supplied severity values are in range.

```bash
python - <<'PY'
from pathlib import Path
import yaml

for filename in ('eval.yaml', 'ablation.yaml', 'timing.yaml'):
    path = Path('configs') / filename
    with path.open() as handle:
        config = yaml.safe_load(handle)

    experiments = config.get('experiments', [])
    names = [item.get('name') for item in experiments]
    assert all(names), f'{filename}: missing experiment name'
    assert len(names) == len(set(names)), f'{filename}: duplicate experiment name'

    for experiment in experiments:
        assert experiment.get('mode'), f'{filename}/{experiment["name"]}: missing mode'
        condition = experiment.get('condition') or {}
        if 'severity' in condition:
            severity = float(condition['severity'])
            assert 0 <= severity <= 1, (
                f'{filename}/{experiment["name"]}: severity {severity} outside [0,1]'
            )

    print(f'[OK] {filename}: {len(experiments)} active experiments')
PY
```

**Do not enforce an old experiment count such as 58.** The active count can change when previously commented-out JPEG2000 blocks are disabled or enabled; commented YAML text does not run.

## 🌍 4. Verify the JPEG2000 conditioning contract

The published SCOPE checkpoint was trained with `diff_jpeg`, `bmshj`, and `bpg_like` only. It does **not** contain a JPEG2000 (`jp2_like`) family embedding.

Your intended configuration split is:

- `configs/eval.yaml`: direct JPEG2000 baselines; unsupported wavelet-conditioned SCOPE experiments disabled.
- `configs/ablation.yaml`: actual JPEG2000 compression with **seen-family conditions** `block_dct`, `learned`, `intra_smooth`.

You can review the active configuration entries with:

```bash
python - <<'PY'
from pathlib import Path
import yaml

for filename in ('eval.yaml', 'ablation.yaml'):
    config = yaml.safe_load((Path('configs') / filename).read_text())
    print('\n', filename)
    for exp in config['experiments']:
        name = str(exp.get('name', '')).lower()
        mode = str(exp.get('mode', '')).lower()
        if 'jp2' in name or 'jpeg2000' in mode:
            print(exp['name'], mode, exp.get('codec_family'))
            if filename == 'ablation.yaml' and mode.startswith('scope_'):
                assert exp.get('codec_family') in {
                    'block_dct', 'learned', 'intra_smooth',
                }, f'Unsupported JPEG2000 SCOPE family in {exp["name"]}'
            if filename == 'eval.yaml' and mode.startswith('scope_'):
                raise AssertionError(
                    f'Unexpected SCOPE JPEG2000 experiment in main eval: {exp["name"]}'
                )
PY
```

This checks the **configuration**; it cannot prove that the checkpoint contents match. A real inference load is still necessary. Only load weights from a trusted source.

## 🧠 5. Check the training configuration

```bash
bash -n scope/train_scope.sh
python -m scope.train_scope --help
```

The shell launcher checks that its configured COCO directories and annotation files exist before starting. Run it with paths appropriate to your machine:

```bash
TRAIN_IMAGES=/data/coco/train2017 \
TRAIN_JSON=/data/coco/annotations/instances_train2017.json \
VAL_IMAGES=/data/coco/val2017 \
VAL_JSON=/data/coco/annotations/instances_val2017.json \
OUT_DIR=runs/scope_new_run \
bash scope/train_scope.sh
```

**This starts training**; it is not a dry-run command. Before committing GPU time, inspect the launcher arguments, selected detector model, codec proxies, and output directory.

## 📊 6. Check evaluation outputs

After running a real evaluation:

```bash
python -m scope.evaluation.evaluate --config configs/eval.yaml
```

Inspect `output_dir` in the YAML and verify that each requested experiment has:

```text
<output_dir>/<experiment>/
├── reconstructed/
├── predictions_coco.json
├── bandwidth.csv
└── summary.json
```

`summary.json` should contain a valid AP metric and rate statistic such as `dataset_bpp`. `bandwidth.csv` should contain per-image encoded bytes/rates, and the number of rows should match the number of evaluated images. The `reconstructed/` PNGs are decoded images, **not** the bitstreams used for bitrate accounting.

When comparing YOLO, RT-DETR, and Faster R-CNN, ensure that all detectors use the same COCO split, category-ID convention, experiment outputs, and intended model weights.

## 📈 7. Audit BDBR curve provenance

The production BDBR command reads every `summary.json` recursively under each provided source directory. Changing an experiment's YAML status later does **not** delete or hide its old output folder.

```bash
python -m scope.evaluation.ablation_bdbr \
  --detector-dir \
    full:yolo=PATH_TO_YOLO_RESULTS \
    full:rtdetr=PATH_TO_RTDETR_RESULTS \
    full:fastrcnn=PATH_TO_FASTERRCNN_RESULTS \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --table-methods full \
  --out-dir outputs_scope_bdbr
```

Open `outputs_scope_bdbr/collected_results.csv` and inspect `summary_path`, `experiment`, `codec`, `detector`, `curve_type`, `AP`, and `dataset_bpp` for all rows. Separate results from different checkpoints and from runs generated before configuration changes.

The BDBR status rows should be checked for missing curves, too few Pareto points, or a missing common AP interval. **An interpolated rate/AP point is not an additional experiment.** Cheng is compared **only as SCOPE versus direct compression** in the integrated BDBR analysis; JPEG2000 conditioning is analyzed separately.

## ⚠️ 8. Limits of these checks

Syntax checks, YAML inspection, filesystem listings and `--help` can catch many reproducibility problems, but **they do not guarantee that every function call, dynamic import, runtime-generated path, or codec interface works**.

A full reproducibility check still requires real training/inference, real codec tools, a detector model, and evaluation on the complete COCO validation split. Do not report a small diagnostic subset's AP or BDBR as a paper reproduction. Record your checkpoint, configuration, actual codec, detector, rate metric, interpolation method, and the common AP interval for each comparison.

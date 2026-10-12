# ✅ Reproducing and Verifying SCOPE Results

This guide provides a practical route from a **fresh SCOPE checkout** to evaluated codec curves and BDBR tables. You can either **train from scratch** or use the released **pretrained SCOPE checkpoint**. Commands are run from the repository root.

## ✅ Automated source and integration tests

The repository includes automated checks, without requiring a downloaded COCO dataset or released model weights for the default test suite. Tests exercise actual SCOPE forward/backward passes on synthetic images, codec subprocess interfaces with fake executables, evaluation outputs, timing, BDBR interpolation and table generation, and the public training shell command.

```bash
# Synthetic tests and static import/call-signature audit
python tools/run_full_tests.py

# Compile paths and check the organized source layout
python tools/validate_environment.py --mode static --show-ok
```

When the research environment is installed (COCO, SCOPE checkpoints, detector weights and real codec executables), run **strict** checking:

```bash
python tools/run_full_tests.py \
  --imports --entrypoints --checkpoints --environment \
  --check-images -1 --strict-external
```

Strict checks validate module imports, direct CLI paths, each selected model's checkpoint conditioning family, COCO image paths, binary availability and packages. They intentionally fail if required assets/dependencies are missing. Passing these checks is not a substitute for running the full 5,000-image evaluation and comparing the numerical outputs.

## 📦 1. Prepare your environment

```bash
git clone https://github.com/SilviaDani/SCOPE.git
cd SCOPE
pip install -r requirements.txt
python training/train_scope.py --help
python evaluation/evaluate_yolo26x_codecs.py --help
python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py --help
```

For the original training setup, install a Detectron2 version compatible with your installed PyTorch/CUDA and the differentiable JPEG backend. Check external codec tools when using BPG and JPEG2000:

```bash
command -v bpgenc && command -v bpgdec
command -v opj_compress && command -v opj_decompress
```

The repo includes `environment.yml` as an alternative environment specification.

## 🗂️ 2. Set up COCO and the detectors

Get [COCO 2017](https://cocodataset.org/#download) images and instance annotations. The default launchers/configs refer to:

```text
datasets/coco-2017/train/data/
datasets/coco-2017/train/labels.json
datasets/coco-2017/validation/data/
datasets/coco-2017/validation/labels.json
models/yolo26x.pt
```

If your data uses standard COCO directories such as `train2017` and `val2017`, point the training environment variables and YAML `dataset` fields to those locations. The label files must be **COCO-format instance annotation JSONs**.

For the detector-transfer experiments, use the model weights corresponding to the intended Faster R-CNN, RT-DETR-l, and YOLO26x architectures.

## 🧠 3. Obtain or train SCOPE weights

**Option A — released full weights (default):** Download the **full SCOPE** checkpoint from [Google Drive](https://drive.google.com/drive/folders/1LFbt_ZrvPJgfbSr174lVPzxeQEkMO0Uz?usp=sharing). All supplied YAML files already point to:

```text
runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt
```

Place `best_val.pt` there to run without editing the checkpoint paths, or set `scope_common.scope_ckpt` in each YAML to its actual location. The Drive folder is also intended to contain **ablated models**; when selecting an ablated checkpoint, use checkpoint-compatible conditioning/inference and architecture settings, and give its evaluation a new `output_dir` (see [Pipeline documentation](PIPELINE.md#choosing-weights-and-configurations)).

**Option B — train from scratch:**

```bash
TRAIN_IMAGES=/data/coco/train2017 \
TRAIN_JSON=/data/coco/annotations/instances_train2017.json \
VAL_IMAGES=/data/coco/val2017 \
VAL_JSON=/data/coco/annotations/instances_val2017.json \
OUT_DIR=runs/scope_from_scratch \
bash training/train_scope.sh
```

When training completes, use the checkpoint stored in the output directory, typically `runs/scope_from_scratch/best_val.pt`.

The released model is conditioned on **three** training codec families: `diff_jpeg`, `bmshj`, and `bpg_like`.

## 🎛️ 4. Configure your evaluation

Edit these YAML files to set the correct paths:

| Config | Experiments |
|:--|:--|
| `configs/eval.yaml` | Direct vs SCOPE for JPEG, BPG, BMSHJ and Cheng; JPEG2000 evaluated in `ablation.yaml`. |
| `configs/ablation.yaml` | JPEG2000 transfer under the three trained conditions. |
| `configs/timing.yaml` | Inference timing for family/severity settings. |

The supplied YAMLs **already select the full model**. Their output directories are `outputs_scope_full_main`, `outputs_scope_full_jpeg2000`, and `outputs_scope_full_timing` respectively. Adapt their settings to your machine or a different checkpoint. The essential YAML fields include:

```yaml
# Example values — adapt to your setup
output_dir: outputs_scope_full_main
dataset:
  images_dir: datasets/coco-2017/validation/data
  annotations_json: datasets/coco-2017/validation/labels.json
models:
  server_model: models/yolo26x.pt
scope_common: &scope_common
  scope_ckpt: runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt
  # Keep the existing architecture settings from your configuration.
```

This fragment illustrates **the default full-model paths**, not a replacement for the complete config; keep the original experiment definitions, codec templates, and model architecture parameters. For training- or conditioning-ablated weights, change `scope_ckpt` and `output_dir`, and ensure the architecture and actual model inputs match the checkpoint. The supplied inference path always passes family **and** severity; changing `condition_mode` in YAML alone does not reproduce family-only or severity-only ablations.

JPEG2000 is an **unseen wavelet codec** and is **not present in `configs/eval.yaml`**. Use `configs/ablation.yaml`; SCOPE+JPEG2000 experiments condition on one of the three trained families, not on `jp2_like`. The wavelet label in the **unmodified JPEG2000 baseline** is codec metadata, not a SCOPE conditioning embedding.

## 🎯 5. Generate codec–detector curves

The main evaluation processes COCO images using the actual codec, runs YOLO on decoded images, and saves one result folder per experiment:

```bash
python evaluation/evaluate_yolo26x_codecs.py --config configs/eval.yaml
python evaluation/evaluate_yolo26x_codecs.py --config configs/ablation.yaml
```

Inspect the configured `output_dir` for:

```text
<output_dir>/
├── <experiment-name>/
│   ├── reconstructed/
│   ├── predictions_coco.json
│   ├── bandwidth.csv
│   └── summary.json
└── all_results.csv
```

`summary.json` reports detection metrics and bitrate; `bandwidth.csv` records encoded-size measurements. The `reconstructed` images are the decoded inputs consumed by the detectors.

## 🧪 6. Evaluate the same reconstructions with more detectors

```bash
# RT-DETR-l
python evaluation/evaluate_rtdetr_l_saved.py \
  --source-eval-dir PATH_TO_YOLO_RESULTS \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --detector-weights PATH_TO_RTDETR_WEIGHTS \
  --category-map coco91 \
  --output-dir PATH_TO_RTDETR_RESULTS
```

```bash
# Faster R-CNN ResNeXt-101-32x8d-FPN (Detectron2)
python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py \
  --source-eval-dir PATH_TO_YOLO_RESULTS \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --output-dir PATH_TO_FASTERRCNN_RESULTS
```

All detector comparisons should use identical reconstructed images, COCO annotations, and consistent COCO category-ID mapping.

## 📈 7. Generate BDBR and LaTeX tables

```bash
python analysis/compute_bdbr.py \
  --detector-dir \
    full:yolo=outputs_scope_full_main \
    full:rtdetr=PATH_TO_RTDETR_RESULTS \
    full:fastrcnn=PATH_TO_FASTERRCNN_RESULTS \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --table-methods full \
  --out-dir outputs_scope_bdbr
```

The output contains `bdbr_summary.csv`, `pairwise_equal_rate_and_bdbr.csv`, and a `latex/` folder. The analysis computes BDBR over the common detection-AP interval. **Negative BDBR represents fewer bits for the same AP.**

For the training-objective ablations, evaluate the respective **ablated checkpoint** into an independent directory first; then pass these directories as `no_task:...` and `no_bg:...` and set `--table-methods full,no_task,no_bg`. Do not relabel full-model outputs as ablated runs.

## ⏱️ 8. Benchmark pre-editing

```bash
python evaluation/benchmark_scope.py \
  --config configs/timing.yaml \
  --warmup 20 --repeat 5 --limit 100
```

Timing reports the model's image pre-editing path. Full compression and detector latency are separate operations.

## 🔎 9. Check your experiment outputs

For each reported detector/codec combination, use the same COCO validation split and detector settings. In the analysis outputs, `collected_results.csv` links the measured AP and bitrate values to their experiment summaries. `interpolated_equal_rate_points.csv` provides numerical rate-matched comparisons; these are interpolated from the measured curves.

The paper's main results use the full **5,000-image COCO 2017 validation set**, not a reduced debugging subset. For JPEG2000, compare the three valid pretrained conditioning choices using the **same actual codec settings**. Cheng belongs to the SCOPE-versus-direct-compression comparison rather than the training-loss ablation set.

### 📚 More details

- [🏠 README](../README.md)
- [📖 Architecture, pipeline and module reference](PIPELINE.md)
- 📄 **Paper:** Proceedings link will be added once available.

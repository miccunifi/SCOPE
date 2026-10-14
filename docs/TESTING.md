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

## 📦 1. Install Python dependencies

Use Python 3.10 on Linux with a compatible NVIDIA driver. **`requirements.txt` is the only Python dependency manifest**. It includes PyTorch, torchvision, Ultralytics, CompressAI, Diff-JPEG and the scientific packages needed by the pipeline.

```bash
git clone https://github.com/SilviaDani/SCOPE.git
cd SCOPE
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Build Detectron2 against the installed PyTorch/CUDA stack
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

You can instead create and activate an empty Python 3.10 Conda environment (`conda create -n d2_scope python=3.10 -y`; `conda activate d2_scope`) and use the same pip installation commands. There is no Conda dependency YAML. PyTorch 2.4.1 / TorchVision 0.19.1 are pinned, and the requirements use the **CUDA 12.1 wheel index**. Detectron2 may also require a system CUDA toolkit (`nvcc`) and a suitable C++ compiler; see its [build instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

Check imports before continuing:

```bash
python -c "import torch, torchvision, yaml, ultralytics, compressai, pycocotools; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import detectron2; print('Detectron2 import: OK')"
python training/train_scope.py --help
python evaluation/evaluate_yolo26x_codecs.py --help
python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py --help
```

The **real-codec executables** are separate from the Python environment:

```bash
# Required by the BPG experiments in configs/eval.yaml
command -v bpgenc && command -v bpgdec
# Required only by the JPEG2000 transfer in configs/ablation.yaml
command -v opj_compress && command -v opj_decompress
```

Install BPG from [Fabrice Bellard's distribution](https://www.bellard.org/bpg/) and OpenJPEG from [the OpenJPEG project](https://github.com/uclouvain/openjpeg). Training uses differentiable surrogates and does **not** require the BPG or JPEG2000 command-line tools.

## 🗂️ 2. Download COCO 2017 and detectors

Download **three official COCO files**: [`train2017.zip`](https://images.cocodataset.org/zips/train2017.zip), [`val2017.zip`](https://images.cocodataset.org/zips/val2017.zip), and [`annotations_trainval2017.zip`](https://images.cocodataset.org/annotations/annotations_trainval2017.zip). Extract the images and the two **instance annotation** files. The default repository layout is:

```text
datasets/coco-2017/
├── train/
│   ├── data/                # images extracted from train2017.zip
│   └── labels.json          # annotations/instances_train2017.json
└── validation/
    ├── data/                # images extracted from val2017.zip
    └── labels.json          # annotations/instances_val2017.json
models/
├── yolo26x.pt               # COCO-pretrained detection model
└── rtdetr-l.pt              # COCO-pretrained RT-DETR-l, for saved-image scoring
```

To reuse COCO already extracted in its original directory layout without copying thousands of images, create local symlinks, substituting your real COCO directory for the placeholder:

```bash
COCO_ROOT="$(realpath /path/to/extracted/COCO)"  # contains train2017/, val2017/, annotations/
mkdir -p datasets/coco-2017/{train,validation}
ln -s "$COCO_ROOT/train2017" datasets/coco-2017/train/data
ln -s "$COCO_ROOT/val2017" datasets/coco-2017/validation/data
ln -s "$COCO_ROOT/annotations/instances_train2017.json" datasets/coco-2017/train/labels.json
ln -s "$COCO_ROOT/annotations/instances_val2017.json" datasets/coco-2017/validation/labels.json
```

Alternatively, use your COCO directories directly by updating `TRAIN_IMAGES` / `TRAIN_JSON` / `VAL_IMAGES` / `VAL_JSON` when training and `dataset.images_dir` / `dataset.annotations_json` in each YAML when evaluating.

### Detector weights

- **[YOLO26x](https://docs.ultralytics.com/models/yolo26/):** use the COCO detection checkpoint `yolo26x.pt` at `models/yolo26x.pt` for the main evaluator.
- **[RT-DETR-l](https://docs.ultralytics.com/models/rtdetr/):** use `rtdetr-l.pt` (not RT-DETR-x) at `models/rtdetr-l.pt` when running the RT-DETR-l saved-image evaluator, passing it via `--detector-weights`.
- **[Detectron2 Faster R-CNN ResNeXt-101-32x8d-FPN](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md):** the training shell and Faster R-CNN evaluator use `COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml`. They automatically fetch the model-zoo COCO checkpoint when not given a local override; for an offline machine, set `D2_WEIGHTS` for training or `--detectron2-weights` for evaluation.

You can let Ultralytics retrieve its public weights in an internet-connected environment and then place them at the script-specific paths:

```bash
mkdir -p models
python - <<'PY'
from ultralytics import YOLO, RTDETR
YOLO('yolo26x.pt')
RTDETR('rtdetr-l.pt')
PY
mv -n yolo26x.pt rtdetr-l.pt models/
```

The `mv` command assumes the two files were downloaded into the repository root and are not already present in `models/`. The source models download automatically when initialized by their standard names. For evaluation, use the *same detector architectures and checkpoints* throughout your comparisons.

## 🧠 3. Obtain full/ablated SCOPE weights or train from scratch

**Option A — released weights (evaluation without training):** Get the **full checkpoint and all ablation checkpoints** from the [SCOPE Google Drive folder](https://drive.google.com/drive/folders/1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES?usp=sharing). The three shipped YAML files default to the **full SCOPE** checkpoint:

```text
runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt
```

Place `best_val.pt` at the default location or change `scope_common.scope_ckpt` in each YAML. To use an ablation, **copy** the config, set its own checkpoint path and `output_dir`, and match its architecture and conditioning inputs (see [Pipeline](PIPELINE.md#choosing-weights-and-configurations)). Merely changing `condition_mode` does **not** disable a conditioning branch in the current runner.

**Option B — training from scratch:** No SCOPE checkpoint is required. The following are required before launching:

1. **Train and validation COCO images** and instance JSONs (see Section 2).
2. **Detectron2 and the COCO-pretrained frozen Faster R-CNN X101-FPN teacher** (automatically downloaded from the model zoo if `D2_WEIGHTS` is unset).
3. **Diff-JPEG** (installed from `requirements.txt`) and **pretrained CompressAI BMSHJ models** (downloaded and cached on first use).
4. A compatible CUDA device and writable `runs/` directory. **No BPG/OpenJPEG executable or YOLO26x/RT-DETR weight is needed to train SCOPE.**

```bash
TRAIN_IMAGES=/data/coco/train2017 \
TRAIN_JSON=/data/coco/annotations/instances_train2017.json \
VAL_IMAGES=/data/coco/val2017 \
VAL_JSON=/data/coco/annotations/instances_val2017.json \
OUT_DIR=runs/scope_from_scratch \
bash training/train_scope.sh
```

The training script saves its own SCOPE checkpoint under the chosen run directory (for example `runs/scope_from_scratch/best_val.pt`). The full released model conditions on `diff_jpeg`, `bmshj`, and `bpg_like`.

**Downloads required for evaluation:** main codec curves additionally need **YOLO26x**, **BPG binaries**, the chosen **SCOPE checkpoint**, and **CompressAI pretrained weights** for BMSHJ/Cheng (downloaded on demand). `configs/ablation.yaml` additionally needs **OpenJPEG binaries**. RT-DETR-l and Faster R-CNN saved-image evaluations additionally need **their own detector checkpoints** and the completed reconstructions; no codec rerun is required. BDBR only needs completed experiment summaries.

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
  --source-eval-dir outputs_scope_full_main \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --detector-weights models/rtdetr-l.pt \
  --category-map coco91 \
  --output-dir outputs_scope_full_rtdetr_l
```

```bash
# Faster R-CNN ResNeXt-101-32x8d-FPN (Detectron2)
python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py \
  --source-eval-dir outputs_scope_full_main \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --output-dir outputs_scope_full_fasterrcnn_x101_fpn
```

All detector comparisons should use identical reconstructed images, COCO annotations, and consistent COCO category-ID mapping.

## 📈 7. Generate BDBR and LaTeX tables

```bash
python analysis/compute_bdbr.py \
  --detector-dir \
    full:yolo=outputs_scope_full_main \
    full:rtdetr=outputs_scope_full_rtdetr_l \
    full:fastrcnn=outputs_scope_full_fasterrcnn_x101_fpn \
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

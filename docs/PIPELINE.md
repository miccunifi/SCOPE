# 📖 SCOPE — Pipeline and Code Reference

This guide describes the **SCOPE implementation** from input images to training, inference, detector evaluation, and BDBR tables. All commands are run from the repository root.

> 📄 **Paper:** *SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression* — Silvia Dani, Leonardo Galteri, Marco Bertini. Proceedings link will be added when available.

## 🧠 1. Architecture

![SCOPE FiLM-conditioned U-Net architecture with codec-family embedding, severity basis, attention, and two residual heads](images/scope_architecture.png)

*Figure 1. Codec-family and severity conditioning modulates the image encoder–decoder. The network predicts an attention map and object/background residuals.*

SCOPE takes an RGB image `I`, a codec-family identifier `f`, and normalized severity `s ∈ [0, 1]`. Its conditioning branch combines a **learned family embedding** with the severity basis, then produces FiLM parameters for the convolutional image branch.

The pre-editor predicts three outputs:

| Output | Meaning |
|:--|:--|
| `A` | Attention map that controls where object preservation is emphasized. |
| `Δ_obj` | Bounded residual applied to the original image in the object branch. |
| `Δ_bg` | Bounded residual applied to the blurred-background branch. |

The final edit blends both branches using `A`. It is then passed to an **ordinary image codec**. The model is a pre-encoder transform: the codec bitstream and downstream detector remain unchanged.

**Training families** are `diff_jpeg` (Diff-JPEG proxy), `bmshj` (learned hyperprior proxy), and `bpg_like` (BPG-like surrogate). The trained family mapping and severity normalization are reused during inference.

## 🔥 2. Training pipeline

![SCOPE training with COCO boxes, training masks, three loss branches, codec proxy, and frozen detector](images/scope_training_pipeline.png)

*Figure 2. The image and condition enter SCOPE; differentiable codec proxies and a frozen detector provide training signals. Only the pre-editor weights are updated.*

1. **Load COCO:** sample training images with annotated bounding boxes; create object-centered crops.
2. **Build masks:** compute attention, detector-context, and background masks from the boxes.
3. **Sample a condition:** select a training codec family and compression severity.
4. **Pre-edit:** run the FiLM-conditioned image network and combine its attention/residual outputs.
5. **Proxy-compress:** pass the edited image through the selected differentiable codec proxy.
6. **Compute losses:** encourage attention in relevant regions, background simplification, residual regularity, and consistency with features from a frozen Faster R-CNN teacher.
7. **Optimize SCOPE:** backpropagate the losses to the pre-editor; validate and write checkpoints.

**At inference**, the masks, boxes, teacher, and differentiable compression proxies are not needed. SCOPE produces an image that can be encoded by a real codec.

### 🗂️ Training files

| File | Responsibility |
|:--|:--|
| `scope/model.py` | Image encoder–decoder, FiLM conditioning, attention/object/background heads. |
| `scope/conditioning.py` | Codec-family probabilities, severity sampling and mapping. |
| `scope/data.py` | COCO datasets, crop sampling, bounding boxes and training masks. |
| `scope/codecs.py` | Differentiable Diff-JPEG, BMSHJ and BPG-like proxies. |
| `scope/losses.py` | Background, rate, attention, detector and residual objectives. |
| `scope/teachers.py` | Frozen detector feature extraction/supervision. |
| `scope/training.py` | Epoch loop, optimizer, validation, logging and checkpoints. |
| `scope/train_scope.py` | Internal CLI parser and its connection to the trainer. |
| `training/train_scope.py` | Public direct training command that delegates to the CLI parser. |
| `training/train_scope.sh` | Paper-style training configuration and convenient launcher. |

### 🧰 Command-line scripts

These small, named entry points invoke the corresponding implementation modules. They do not contain new model, codec, detector, or BDBR algorithms.

| Command | Implementation | Purpose |
|:--|:--|:--|
| `python training/train_scope.py` | `scope/train_scope.py` | Training CLI. |
| `python evaluation/evaluate_yolo26x_codecs.py` | `scope/evaluation/evaluate.py` | Real codecs and YOLO26x evaluation. |
| `python evaluation/evaluate_rtdetr_l_saved.py` | `scope/evaluation/saved_rtdetr.py` | RT-DETR-l transfer. |
| `python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py` | `scope/evaluation/saved_multi_detector.py` | Faster R-CNN ResNeXt-101-32x8d-FPN, using Detectron2. |
| `python analysis/compute_bdbr.py` | `scope/evaluation/ablation_bdbr.py` | BDBR and LaTeX. |
| `python evaluation/benchmark_scope.py` | `scope/evaluation/timing.py` | Pre-editor timing. |
| `python evaluation/evaluate_categories.py` | `scope/evaluation/rate_distortion.py` | Per-category analysis. |
| `python evaluation/visualize_results.py` | `scope/evaluation/visualize.py` | Qualitative detection figures. |

Run each with `--help` for supported options. Run commands from the **repository root**.

### 🚀 Run training

```bash
# Use the defaults from the training launcher
bash training/train_scope.sh
```

For another COCO location, set the corresponding environment variables:

```bash
TRAIN_IMAGES=/data/coco/train2017 \
TRAIN_JSON=/data/coco/annotations/instances_train2017.json \
VAL_IMAGES=/data/coco/val2017 \
VAL_JSON=/data/coco/annotations/instances_val2017.json \
OUT_DIR=runs/scope_from_scratch \
CUDA_VISIBLE_DEVICES=0 \
BATCH=8 \
LR=1e-4 \
bash training/train_scope.sh
```

The launcher uses a 512×512 crop, a 48-channel base network, 40 epochs, the three training proxy families, and Detectron2 Faster R-CNN supervision. The run saves its checkpoint files under `OUT_DIR` (including `best_val.pt` after applicable validation).

For all training arguments:

```bash
python training/train_scope.py --help
```

### ✅ Quality checks

The `tests/` directory covers configuration validity, codec subprocess interfaces, forward/backward passes, checkpoint loading, detector evaluation output contracts, timing, and BDBR. The `tools/` directory contains static import/call audits and optional real-environment preflight checks. Run from the repository root:

```bash
python tools/run_full_tests.py
python tools/validate_environment.py --mode static
```

Full model/data checks are described in [✅ Testing](TESTING.md).

## 📦 3. Data, dependencies and model weights

### Installation

```bash
git clone https://github.com/SilviaDani/SCOPE.git
cd SCOPE
pip install -r requirements.txt
```

`environment.yml` offers an alternative CUDA-oriented Conda environment. Training with the paper's frozen teacher additionally needs **Detectron2** with a suitable CUDA/PyTorch build and the differentiable JPEG implementation. BPG/JPEG2000 require command-line codec tools, which are external to the Python package:

```bash
command -v bpgenc
command -v bpgdec
command -v opj_compress
command -v opj_decompress
```

### Data layout

Download COCO 2017 from the [official website](https://cocodataset.org/#download). You may use the default layout below or point the YAML/launcher to your own folders:

```text
datasets/coco-2017/
├── train/
│   ├── data/                  # COCO training images
│   └── labels.json           # COCO instance annotations for train
└── validation/
    ├── data/                  # COCO validation images
    └── labels.json           # COCO instance annotations for validation
```

For YOLO26x, the default configuration refers to `models/yolo26x.pt`. The three evaluation YAMLs default to the same **full SCOPE** checkpoint (`runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt`). You may download its weights from [Google Drive](https://drive.google.com/drive/folders/1LFbt_ZrvPJgfbSr174lVPzxeQEkMO0Uz?usp=sharing), place them at the configured path, or edit `scope_common.scope_ckpt` to their actual location. Training from scratch is also supported.

### Choosing weights and configurations

All three shipped configurations evaluate the **full model**, not an ablated variant:

| YAML | Default output directory | Default checkpoint |
|:--|:--|:--|
| `configs/eval.yaml` | `outputs_scope_full_main` | Full SCOPE |
| `configs/ablation.yaml` | `outputs_scope_full_jpeg2000` | Full SCOPE |
| `configs/timing.yaml` | `outputs_scope_full_timing` | Full SCOPE |

Each YAML defines a `scope_common` anchor containing `scope_ckpt`, network architecture parameters, and `condition_mode`, which are inherited by the SCOPE experiments. The default checkpoint path is:

```text
runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt
```

The linked Drive folder is also intended to distribute **training/conditioning ablation checkpoints**. The YAMLs are *examples of the full-model setup*, not universal settings for every checkpoint. To run an ablation:

1. Copy the corresponding YAML to a new file (for example, `configs/eval_family_only.yaml`), keeping the original full-model config unchanged.
2. Change `scope_common.scope_ckpt` to the downloaded ablated `best_val.pt` file, and change `output_dir` to a **new, variant-specific directory** (for example, `outputs_scope_family_only_main`).
3. Match architecture parameters (`base_ch`, attention/background settings, etc.) and the **actual conditioning inputs** to the chosen checkpoint's training setup. The current SCOPE inference path always passes **both** codec-family and severity to the model; merely changing the YAML `condition_mode` field does **not** switch off a conditioning input. Reproducing a family-only or severity-only ablation may therefore require a matching inference implementation/adapter, not just selecting different weights. Inspect the checkpoint's saved training arguments and use the evaluation procedure corresponding to that variant.
4. Use the resulting directory when calling detector-transfer or BDBR scripts, with the correct method label (`full`, `no_task`, `no_bg`, etc.). Never combine partial outputs from different checkpoints in one output directory.

`configs/ablation.yaml` covers **JPEG2000 codec-family transfer using full SCOPE**, which is distinct from the **training-objective/conditioning ablations** that require separately trained weights.

These paths are relative to the repository root; checkpoint files and datasets are **not** included in the ZIP.

## 🎛️ 4. Codec families, severity and experiment configs

Larger normalized severity means **stronger compression**. Codec parameters such as JPEG quality, BPG QP and learned-codec quality are converted to severity using codec-specific ranges.

| Real codec | In training? | SCOPE family condition |
|:--|:--|:--|
| JPEG | ✅ Diff-JPEG proxy | `block_dct` → `diff_jpeg` |
| BPG | ✅ BPG-like proxy | `intra_smooth` → `bpg_like` |
| BMSHJ | ✅ BMSHJ proxy | `learned` → `bmshj` |
| Cheng | 🌍 Unseen learned codec | `learned` → `bmshj` |
| JPEG2000 | 🌍 Unseen wavelet codec | Choose **one of the three trained families** |

The pretrained checkpoint has **no dedicated `jp2_like`/wavelet family embedding**. In the JPEG2000 transfer experiment, JPEG2000 is the actual codec, while the conditioning input is one of `block_dct`, `learned`, or `intra_smooth`.

| YAML | Usage |
|:--|:--|
| `configs/eval.yaml` | JPEG/BPG/BMSHJ/Cheng direct-vs-SCOPE evaluation (46 active experiments). |
| `configs/ablation.yaml` | JPEG2000 cross-family SCOPE experiments using the *trained* family embeddings. |
| `configs/timing.yaml` | Pre-editor execution time across families and severities. |

An active experiment is an element of a YAML `experiments:` list. Its `name` names the output folder, its `mode` selects the codec path, and SCOPE experiments inherit network/checkpoint settings from `scope_common`. **Main `eval.yaml` has no JPEG2000/wavelet-family templates**; all JPEG2000 tests are isolated in `ablation.yaml`. For new datasets, detectors or model variants, edit the relevant paths and settings as described above.

## 🎯 5. Real-codec evaluation

Evaluation uses **real encoded bitstreams** for bitrate measurement, not the differentiable training proxies.

```bash
python evaluation/evaluate_yolo26x_codecs.py --config configs/eval.yaml
python evaluation/evaluate_yolo26x_codecs.py --config configs/ablation.yaml
```

The main evaluator (`scope/evaluation/evaluate.py`) loads the COCO validation split, processes each active experiment, runs YOLO on decoded images, and computes COCO AP/AR. Each experiment has either:

```text
Original image ────────────► Real codec ─► Reconstructed image ─► Detector
Original image ─► SCOPE ──► Real codec ─► Reconstructed image ─► Detector
```

The YAML's `output_dir` is populated with:

```text
<output_dir>/
├── <experiment>/
│   ├── reconstructed/           # Decoded images for detection
│   ├── attention_debug/         # Optional model intermediates
│   ├── predictions_coco.json    # Detector output in COCO format
│   ├── bandwidth.csv            # Per-image encoded bytes / bitrate
│   └── summary.json             # AP, AR, dataset_bpp, image-size statistics
└── all_results.csv              # Summary for all experiments
```

`dataset_bpp` represents actual encoded bits divided by the evaluated image pixels. The reconstructed PNGs are for **detector evaluation**, not rate calculation.

### 🔧 Evaluation implementation files

| File | Role |
|:--|:--|
| `scope/evaluation/evaluate.py` | Main experiment loop and CSV aggregation. |
| `scope/evaluation/dataset.py` | COCO evaluation image records. |
| `scope/evaluation/conditioning.py` | Family aliases and severity conversion for test-time codecs. |
| `scope/evaluation/codec_transmissions.py` | Real BPG/JPEG2000 and SCOPE transmission paths. |
| `scope/evaluation/transmission.py` | JPEG and learned-codec transmissions, optional ROI paths. |
| `scope/evaluation/yolo_utils.py` | Ultralytics detector interface. |
| `scope/evaluation/coco_metrics.py` | COCOeval metric evaluation. |
| `scope/evaluation/categories.py` | Mapping COCO80 detector labels to official COCO category IDs. |
| `scope/evaluation/utils.py` | Image loading/writing, shared geometry and file functions. |
| `scope/inference.py` | Load a trained SCOPE model and edit image tensors. |

## 🧪 6. Transfer across object detectors

Re-use the saved reconstructed images, so each detector sees **the same compressed pixels**. These commands evaluate results without recompressing the images.

### RT-DETR

```bash
python evaluation/evaluate_rtdetr_l_saved.py \
  --source-eval-dir PATH_TO_YOLO_RESULTS \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --detector-weights PATH_TO_RTDETR_WEIGHTS \
  --category-map coco91 \
  --output-dir PATH_TO_RTDETR_RESULTS
```

### Faster R-CNN (paper-aligned detector)

```bash
python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py \
  --source-eval-dir PATH_TO_YOLO_RESULTS \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --output-dir PATH_TO_FASTERRCNN_RESULTS
```

The paper uses **Faster R-CNN ResNeXt-101-32x8d-FPN**. The `evaluate_fasterrcnn_x101_fpn_saved.py` script selects Detectron2 and `fasterrcnn_x101_fpn` automatically. It uses the Detectron2 model-zoo checkpoint by default; use `--detectron2-weights` for an explicit weight file. A TorchVision ResNet-50 checkpoint is a different detector.

`scope/evaluation/saved_rtdetr.py`, `scope/evaluation/rtdetr_utils.py`, and `scope/evaluation/saved_multi_detector.py` implement these inference paths. COCO category IDs must be mapped consistently to the official annotation IDs.

## 📈 7. Rate–accuracy curves and BDBR

The BDBR code consumes the measured `summary.json` and `bandwidth.csv` files produced by evaluation; the main implementation resides in `scope/evaluation/ablation/`:

| File | Role |
|:--|:--|
| `scope/evaluation/ablation_bdbr.py` | Public BDBR entry point. |
| `scope/evaluation/ablation/cli.py` | CLI settings and analysis execution. |
| `scope/evaluation/ablation/common.py` | Codec, detector, method and metric conventions. |
| `scope/evaluation/ablation/results.py` | Read measured AP, bpp and experiment metadata. |
| `scope/evaluation/ablation/curves.py` | Curve preparation, Pareto filtering and BDBR interpolation. |
| `scope/evaluation/ablation/comparisons.py` | Pairwise equal-rate comparisons. |
| `scope/evaluation/ablation/tables.py` | CSV/LaTeX tables and summary statistics. |

### SCOPE versus direct compression

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

For training-objective ablations, pass corresponding source directories labeled `no_task:...` and `no_bg:...`, and choose `--table-methods full,no_task,no_bg`. The shared-aperture comparisons use measured detector **AP** and actual **bitrate**, with negative BDBR indicating rate savings at matched AP.

Output tables are saved in the selected directory:

```text
outputs_scope_bdbr/
├── collected_results.csv
├── interpolated_equal_rate_points.csv
├── interpolation_status.csv
├── pairwise_equal_rate_and_bdbr.csv
├── bdbr_summary.csv
├── diagnostics.txt
└── latex/
```

**Cheng** is included only for SCOPE against direct compression in the integrated analysis. The **JPEG2000** cross-family conditioning study is handled separately. The paper's external TSW-pre comparison comes from published data rather than a local SCOPE training run.

## 🗂️ 8. Other analyses and figures

| File | Purpose |
|:--|:--|
| `scope/evaluation/rate_distortion.py` | COCO metrics broken down by codec, category/supercategory and severity. |
| `scope/evaluation/visualize.py` | Side-by-side qualitative compressed-image detection panels. |
| `scope/evaluation/timing.py` | Image pre-editing timing across family/severity conditions. |
| `scope/evaluation/roi_codec_model.py` | Optional ROI codec components. |
| `scope/evaluation/roi_utils.py` | Optional ROI geometry and masks. |
| `scope/evaluation/tile_utils.py` | Optional tile-level utilities. |

```bash
python evaluation/evaluate_categories.py --help
python evaluation/visualize_results.py --help
python evaluation/benchmark_scope.py --config configs/timing.yaml
```

Runtime measurements characterize the **pre-editor**, not complete end-to-end codec encoding, transmission, and detector execution.

## 📚 9. Further reading

- [🏠 README](../README.md) — overview, results and quick start.
- [✅ Reproduce and verify](TESTING.md) — the recommended experimental sequence and output checks.
- 📄 **Paper:** official proceedings link coming soon.

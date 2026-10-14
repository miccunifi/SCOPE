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
6. **Compute losses:** supervise object attention; encourage background simplification and compressibility; regularize residual magnitude; and preserve object-related features through the frozen Faster R-CNN teacher.
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

## 📦 3. Environment, datasets and pretrained weights

### Install Python dependencies from `requirements.txt`

Use **Python 3.10** with a compatible NVIDIA driver. All regular Python dependencies (including PyTorch, Ultralytics, CompressAI and Diff-JPEG) are specified in the single `requirements.txt` file:

```bash
git clone https://github.com/SilviaDani/SCOPE.git
cd SCOPE
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

To use Conda only as the environment manager, replace the `venv` creation/activation with `conda create -n d2_scope python=3.10 -y` and `conda activate d2_scope`; still install Python dependencies with `pip install -r requirements.txt`. The requirements pin PyTorch 2.4.1 / TorchVision 0.19.1 and point to the **CUDA 12.1 wheel index** used for the original experiments. Adapt that index to your local CUDA/driver setup if necessary.

**Detectron2 is the separate build step**, required for frozen Faster R-CNN training supervision and saved-image Faster R-CNN scoring:

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import detectron2; print('Detectron2 import: OK')"
```

Detectron2 uses native/CUDA extensions, so it may require a matching **system CUDA toolkit (`nvcc`)** and C++ compiler in addition to the pip dependencies. Follow the [official build requirements](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). GPU reproduction is not guaranteed on an unsupported PyTorch/CUDA combination.

**External real-codec programs** (not Python dependencies installed by pip):

| Evaluation | Required executable(s) | Source |
|:--|:--|:--|
| Real JPEG | None; uses image codec libraries | Built-in evaluation path |
| Real BPG | `bpgenc`, `bpgdec` | [BPG project](https://www.bellard.org/bpg/) |
| JPEG2000 transfer in `configs/ablation.yaml` | `opj_compress`, `opj_decompress` | [OpenJPEG](https://github.com/uclouvain/openjpeg) |
| BMSHJ and Cheng | No external executable; pretrained CompressAI models | [CompressAI](https://github.com/InterDigitalInc/CompressAI) |

```bash
command -v bpgenc && command -v bpgdec
# Required only for the JPEG2000 transfer study:
command -v opj_compress && command -v opj_decompress
```

### Downloads before training

- **COCO 2017 `train2017.zip` + `val2017.zip` + `annotations_trainval2017.zip`** from the [official COCO downloads](https://cocodataset.org/#download), including the **instance** annotation JSON files. Training needs the training split and, with the supplied shell launcher, validation too.
- **Detectron2 Faster R-CNN X101-32x8d-FPN COCO pretrained weights:** the supplied shell uses `COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml` and downloads the model-zoo checkpoint when `D2_WEIGHTS` is empty. Set `D2_WEIGHTS=/path/to/model.pkl` to avoid a model-zoo download. This is a frozen teacher, not an SCOPE checkpoint.
- **Differentiable JPEG** is installed through `requirements.txt`, and **CompressAI BMSHJ pretrained weights** are fetched/cached automatically at first use. The BPG-like training proxy needs **no** `bpgenc` binary.
- **No existing SCOPE checkpoint is needed** to train from scratch.

### Downloads before evaluation

| Evaluation step | Required inputs / downloads |
|:--|:--|
| **Main YOLO26x codec evaluation** | COCO 2017 **validation** images and instance annotations; `models/yolo26x.pt` [YOLO26x weights](https://docs.ultralytics.com/models/yolo26/); [full SCOPE or selected ablation checkpoint](https://drive.google.com/drive/folders/1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES?usp=sharing); `bpgenc`/`bpgdec` for BPG; CompressAI's pretrained BMSHJ and Cheng weights are fetched automatically if missing. |
| **JPEG2000 transfer** | The same COCO validation data, YOLO26x and SCOPE weights, plus `opj_compress`/`opj_decompress` from OpenJPEG; uses `configs/ablation.yaml`. |
| **RT-DETR-l on saved reconstructions** | COCO validation annotations, the **completed reconstruction folders** from the main YOLO evaluation, and [`rtdetr-l.pt`](https://docs.ultralytics.com/models/rtdetr/). Compression does **not** need to be rerun. |
| **Faster R-CNN on saved reconstructions** | COCO validation annotations, the same **saved reconstructions**, and Detectron2's Faster R-CNN X101-FPN model-zoo weights (or a local checkpoint). Compression does **not** need to be rerun. |
| **BDBR analysis** | Completed detector evaluation result folders with `summary.json` / `bandwidth.csv`; no new model weights required. |

### COCO and detector paths

Extract the standard COCO archives and either use your existing data locations in `configs/*.yaml` and `TRAIN_IMAGES`/`TRAIN_JSON`/`VAL_IMAGES`/`VAL_JSON`, or arrange the files in the repository's default layout:

```text
datasets/coco-2017/
├── train/
│   ├── data/                  # train2017 images
│   └── labels.json            # instances_train2017.json
└── validation/
    ├── data/                  # val2017 images
    └── labels.json            # instances_val2017.json
models/
├── yolo26x.pt                 # for main COCO evaluations
└── rtdetr-l.pt                # for RT-DETR-l saved-image evaluation
```

The files inside the dataset directory may be symlinks to an existing COCO installation. The config's `models.server_model` defaults to `models/yolo26x.pt`. Obtain the YOLO26x and RT-DETR-l COCO detection weights from [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) and [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/), respectively. Both detectors support automated weight retrieval through their Python library when instantiated by model name, but save the files at the explicit paths above for these scripts.

### Choosing weights and configurations

All three shipped configurations use the **full SCOPE** checkpoint as their default. The **[SCOPE Google Drive folder](https://drive.google.com/drive/folders/1YbB1M0Bdp1QZl-aoGNWCCsJRMgzROGES?usp=sharing)** contains the **full model and ablated variants**; the weight files are not stored in this Git repository.

| YAML | Default output directory | Default checkpoint |
|:--|:--|:--|
| `configs/eval.yaml` | `outputs_scope_full_main` | Full SCOPE |
| `configs/ablation.yaml` | `outputs_scope_full_jpeg2000` | Full SCOPE |
| `configs/timing.yaml` | `outputs_scope_full_timing` | Full SCOPE |

By default, `scope_common.scope_ckpt` points to:

```text
runs/preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_FULL_20260701_140557/best_val.pt
```

To evaluate a different checkpoint, **copy** the relevant YAML (for example, `configs/eval_family_only.yaml`), update `scope_common.scope_ckpt`, and use a distinct `output_dir` (for example, `outputs_scope_family_only_main`). Confirm that architecture parameters and actual conditioning inputs are compatible with that checkpoint. The current inference path passes both family and severity; setting `condition_mode` alone does not recreate the family-only or severity-only ablations. Read the checkpoint's saved hyperparameters and match its evaluation procedure. Keep results from different checkpoints in separate directories for fair BDBR comparisons.

`configs/ablation.yaml` is for **JPEG2000 transfer using full SCOPE**, not a general configuration for all released ablated checkpoints. All paths shown are relative to the repository root, so invoke the commands from there.

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
  --source-eval-dir outputs_scope_full_main \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --detector-weights models/rtdetr-l.pt \
  --category-map coco91 \
  --output-dir outputs_scope_full_rtdetr_l
```

### Faster R-CNN (paper-aligned detector)

```bash
python evaluation/evaluate_fasterrcnn_x101_fpn_saved.py \
  --source-eval-dir outputs_scope_full_main \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --output-dir outputs_scope_full_fasterrcnn_x101_fpn
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
    full:rtdetr=outputs_scope_full_rtdetr_l \
    full:fastrcnn=outputs_scope_full_fasterrcnn_x101_fpn \
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

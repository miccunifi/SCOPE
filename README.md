# SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression

Official implementation of:

**SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression**  
Silvia Dani, Leonardo Galteri, Marco Bertini

**SCOPE** is a conditional image pre-editor that improves the **bitrate–object-detection accuracy trade-off** by modifying images *before* compression. One network adapts to different **codec families** and **compression severities**, without retraining a codec-specific pre-editor or modifying the downstream object detector.

SCOPE learns where to **preserve object-relevant information** and where to **simplify background content**, enabling more efficient compression while maintaining detection accuracy.

> 📄 **Paper:** Proceedings link will be added once available.  
> 🧠 **Pretrained checkpoint:** [Google Drive](https://drive.google.com/drive/folders/1LFbt_ZrvPJgfbSr174lVPzxeQEkMO0Uz?usp=sharing)  
> 💻 **Code:** [miccunifi/SCOPE](https://github.com/miccunifi/SCOPE)

## ✨ Highlights

- 🎛️ **Codec-aware conditioning:** family embeddings and normalized compression severity control a single pre-editor.
- 🎯 **Object-preserving edits:** an attention map blends object residuals with a simplified background branch.
- 🧪 **Differentiable training:** codec proxies and a frozen detector provide compression-aware, task-driven supervision.
- 🔌 **Standard-codec compatibility:** real codecs and downstream detectors are not modified.
- 🌍 **Zero-shot transfer:** the same trained model is evaluated with unseen **Cheng** and **JPEG2000** codecs.
- 📉 **Bitrate savings:** **−15.98% mean BDBR** across nine JPEG/BPG/BMSHJ–detector comparisons, as reported in the paper.

## 🔍 Method overview

```text
           Image I + codec family f + compression severity s
                               │
                               ▼
                     SCOPE conditional U-Net
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
              Attention    Object edit   Background edit
                 └─────────────┼─────────────┘
                               ▼
                         Edited image
                               │
                               ▼
                      Unmodified codec
                               │
                               ▼
                      Compressed image
                               │
                               ▼
                      Object detector
```

**Training:** the network sees differentiable **Diff-JPEG**, **BMSHJ**, and **BPG-like** proxy compressions. A frozen Faster R-CNN provides feature-consistency supervision; ground-truth boxes are used to construct training masks.

**Inference:** only the image and its codec-family/severity condition are needed. No detector, bounding boxes, or supervision masks are required to run the pre-editor itself.

**Unseen JPEG2000:** the released checkpoint does **not** contain a `jp2_like`/wavelet condition. JPEG2000 is evaluated using each of the *three trained conditioning families* (`diff_jpeg`, `bmshj`, `bpg_like`), while the real compression codec remains JPEG2000.

## 📊 Results

SCOPE is evaluated on **COCO 2017 validation** using **Faster R-CNN (ResNeXt-101-FPN)**, **RT-DETR-l**, and **YOLO26x**. BDBR is computed with object-detection AP as the quality axis; **negative values represent bitrate savings at equal detection accuracy**.

### 🎛️ Conditioning ablation

| Configuration | Mean BDBR vs. direct compression (%) |
|:--|--:|
| **SCOPE — family + severity** | **−15.98** |
| Family only | −15.42 |
| Severity only | −13.28 |

*Mean across the nine JPEG, BPG, and BMSHJ codec–detector pairs.* Codec-specific specialists, per-codec results, and pairwise comparisons with full SCOPE are reported in **Table 3 of the paper**. Pairwise BDBRs are computed independently and **must not** be obtained by subtracting two baseline-relative BDBR values.

### 🌍 Transfer to unseen codecs

On Cheng, SCOPE achieves **−20.66% BDBR (unseen)** for Faster R-CNN. The paper reports **−31.32%** for Cheng-trained **TSW-pre (seen)**; these settings differ and should not be treated as a strictly controlled head-to-head comparison.

For JPEG2000, the paper reports mean BDBRs of **−1.35%** (Diff-JPEG condition), **−9.97%** (BMSHJ), and **−11.92%** (BPG-like) across the three detectors. See the paper for the individual detector results and evaluation protocol.

## 📁 Repository structure

```text
SCOPE/
├── README.md
├── docs/
│   ├── PIPELINE.md              # Full pipeline and module reference
│   ├── TESTING.md               # Manual validation and reproducibility checks
├── configs/
│   ├── eval.yaml                # Main codec / detector experiments
│   ├── ablation.yaml            # JPEG2000 cross-family conditioning
│   └── timing.yaml              # Runtime benchmarks
├── scope/
│   ├── __init__.py
│   ├── model.py                 # Conditional editing network
│   ├── conditioning.py          # Training family/severity sampling
│   ├── data.py                  # COCO crops, boxes, training masks
│   ├── codecs.py                # Differentiable codec proxies
│   ├── losses.py                # Training objectives
│   ├── teachers.py              # Frozen detector supervision
│   ├── training.py              # Training loop and checkpoints
│   ├── train_scope.py           # Training CLI
│   ├── train_scope.sh           # Training launcher
│   ├── inference.py             # Checkpoint loading / inference
│   └── evaluation/
│       ├── evaluate.py          # Compression → detection → COCO AP
│       ├── timing.py            # SCOPE runtime
│       ├── saved_rtdetr.py      # RT-DETR on saved reconstructions
│       ├── saved_multi_detector.py  # Other detectors on saved images
│       ├── rate_distortion.py   # Category/severity analysis
│       ├── ablation_bdbr.py     # BDBR entry point
│       └── ablation/            # Curve/metric/table implementation
├── requirements.txt
└── environment.yml
```

The `evaluation/` package also contains codec-specific transmissions, COCO metrics, detector wrappers, and optional ROI utilities. See [📖 PIPELINE.md](docs/PIPELINE.md) for the **complete file-by-file reference**.

This repository layout **does not include** the previously developed `tests/` or `tools/` directories. Accordingly, the documentation below does not rely on those commands.

## ⚙️ Installation

```bash
# From the SCOPE repository root
pip install -r requirements.txt
```

Alternatively, use the CUDA-oriented `environment.yml`. Training with Detectron2 supervision additionally requires a **compatible Detectron2 installation** and the configured differentiable-JPEG backend. BPG and JPEG2000 evaluations require `bpgenc`/`bpgdec` and `opj_compress`/`opj_decompress` respectively.

External data and model files are **not shipped** with the code. Typical locations are:

```text
datasets/coco-2017/train/data/
datasets/coco-2017/train/labels.json
datasets/coco-2017/validation/data/
datasets/coco-2017/validation/labels.json
models/yolo26x.pt
runs/<training-run>/best_val.pt
```

The evaluation configurations declare their own dataset paths, detector weights, output directories, and `scope_ckpt`. Check those values before running experiments. Checkpoint folders still named `preedit_*` are legacy training artifacts and **do not need renaming**.

## 🚀 Training

```bash
bash scope/train_scope.sh
```

The shell launcher sits beside `scope/train_scope.py` and defines the actual training options. Override paths or key settings with environment variables, for example:

```bash
TRAIN_IMAGES=/data/coco/train2017 \
TRAIN_JSON=/data/coco/annotations/instances_train2017.json \
VAL_IMAGES=/data/coco/val2017 \
VAL_JSON=/data/coco/annotations/instances_val2017.json \
OUT_DIR=runs/scope_new_run \
CUDA_VISIBLE_DEVICES=0 \
bash scope/train_scope.sh
```

To view all Python training options:

```bash
python -m scope.train_scope --help
```

## 📦 Evaluation

```bash
# JPEG, BPG, BMSHJ, Cheng and supported direct-compression baselines
python -m scope.evaluation.evaluate --config configs/eval.yaml

# JPEG2000 transfer: three trained family conditions, not wavelet conditioning
python -m scope.evaluation.evaluate --config configs/ablation.yaml
```

For each evaluated experiment, SCOPE writes reconstructed images, predictions, measured bitrate statistics and a COCO summary. To reuse the reconstructions with additional detectors:

```bash
python -m scope.evaluation.saved_rtdetr --help
python -m scope.evaluation.saved_multi_detector --help
```

See [📖 PIPELINE.md](docs/PIPELINE.md) for full commands and outputs.

## 📈 BDBR and rate–accuracy analysis

```bash
python -m scope.evaluation.ablation_bdbr \
  --detector-dir \
    full:yolo=PATH_TO_FULL_YOLO_RESULTS \
    full:rtdetr=PATH_TO_FULL_RTDETR_RESULTS \
    full:fastrcnn=PATH_TO_FULL_FASTERRCNN_RESULTS \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --table-methods full \
  --out-dir outputs_scope_bdbr
```

To compare the full model with the task/background ablations, add corresponding sources such as `no_task:yolo=...` and `no_bg:yolo=...`, then use `--table-methods full,no_task,no_bg`.

The BDBR module computes comparisons from **saved measured AP and bitrate**. It includes **Cheng only for SCOPE versus direct compression**; JPEG2000's cross-family analysis is separate. The output includes `collected_results.csv` for auditing every input point, pairwise BDBR CSVs, and LaTeX tables.

**⚠️ Curve provenance:** the analysis recursively reads `summary.json` in each supplied directory, including old runs that may no longer be enabled in YAML. Review the `summary_path` column in `collected_results.csv`. Interpolated points are *derived samples*, not independent measured evaluations.

## ⏱️ Runtime

```bash
python -m scope.evaluation.timing --config configs/timing.yaml
```

Timing reports pre-editor execution; it does not measure complete codec encoding or detector runtime.

## ✅ Reproducibility and documentation

- [📖 Pipeline and file reference](docs/PIPELINE.md)
- [🔎 Manual validation and reproducibility](docs/TESTING.md)
- 📄 **Paper:** Proceedings link will be added once available.

Because `tests/` and `tools/` were removed, **there is no bundled automated test runner or path-preflight CLI**. The manual checks in `docs/TESTING.md` cover syntax, configurations, input paths, outputs, and result provenance; full compatibility still requires real evaluation on the research machine.

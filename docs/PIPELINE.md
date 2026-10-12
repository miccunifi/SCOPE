# 📖 SCOPE: Implementation and pipeline guide

This document describes the pipeline implemented in **SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression**, from data loading to training, real-codec evaluation, detector re-evaluation, and BDBR analysis.

**Run commands from the repository root.** Paths in the current YAML files are interpreted relative to the process working directory; they are not automatically relative to `configs/`.

> 📄 **Paper:** Proceedings link coming soon · 🏠 [README](../README.md) · 🔎 [Validation guide](TESTING.md)

## 🔄 1. End-to-end workflow

### 🏋️ Training

```text
COCO images + bounding boxes
            │
            ▼
  Crop sampling and GT masks                 Codec family f + severity s
            │                                            │
            └────────────────┬───────────────────────────┘
                             ▼
                   SCOPE conditional U-Net
                             │
             Attention + object/background edits
                             │
                             ▼
                  Differentiable codec proxy
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       Rate/region/attention       Frozen detector feature
          regularization               consistency
                └────────────┬────────────┘
                             ▼
                 Loss → gradient → optimizer
                             │
                             ▼
                      best_val.pt
```

Only the **pre-editor** is trained. Codec proxies and the detector used for supervision are fixed. Ground-truth boxes are used to build training masks; they are **not** required to pre-edit an image at inference time.

### 📊 Evaluation and analysis

```text
COCO validation image
   ├──► Baseline: directly compress input ──────────────────┐
   └──► SCOPE: pre-edit → real codec ───────────────────────┤
                                                           ▼
                                                 Reconstructed image
                                                           │
                                      ┌────────────────────┴──────────────────┐
                                      ▼                                       ▼
                               YOLO + COCOeval                      Saved-image detectors
                                      │                               (RT-DETR, Faster R-CNN)
                                      └──────────────────┬────────────────────┘
                                                         ▼
                                            summary.json + bandwidth.csv
                                                         │
                                                         ▼
                                           Rate–accuracy / BDBR tables
```

**Important:** The training compression proxies are not used to measure final bitrate. The reported evaluation bitrate is calculated from **actual encoded data sizes**, not from the file sizes of saved reconstructed PNGs.

## 📁 2. Directory structure and file responsibilities

```text
SCOPE/
├── README.md                         GitHub introduction and quick start
├── docs/
│   ├── PIPELINE.md                   This document
│   ├── TESTING.md                    Manual validation and reproduction checks
├── configs/
│   ├── eval.yaml                     Main evaluated codecs and detectors
│   ├── ablation.yaml                 JPEG2000 cross-family conditioning
│   └── timing.yaml                   Family/severity timing runs
├── scope/
│   ├── __init__.py                   Package exports
│   ├── model.py                      Network architecture and editing heads
│   ├── conditioning.py               Training family/severity choices
│   ├── data.py                       COCO dataset, crops, training masks
│   ├── codecs.py                     Differentiable compression proxies
│   ├── losses.py                     Training objective functions
│   ├── teachers.py                   Frozen detector supervision
│   ├── training.py                   Training loop, validation, checkpoints
│   ├── train_scope.py                Argument parser, training entry point
│   ├── train_scope.sh                Training launcher with defaults
│   ├── inference.py                  Checkpoint loader, image pre-editing
│   └── evaluation/
│       ├── __init__.py               Evaluation package
│       ├── evaluate.py               Real codecs + YOLO + COCO AP/AR
│       ├── dataset.py                Evaluation COCO records
│       ├── conditioning.py           Codec-family aliases, severity at test time
│       ├── codec_transmissions.py    SCOPE + real codec pathways
│       ├── transmission.py           JPEG/learned/ROI transmissions
│       ├── coco_metrics.py           COCOeval metrics
│       ├── categories.py             COCO class-ID conversions
│       ├── yolo_utils.py             YOLO predictor adapter
│       ├── rtdetr_utils.py           RT-DETR predictor adapter
│       ├── saved_rtdetr.py           RT-DETR on reconstructed images
│       ├── saved_multi_detector.py   TorchVision/Detectron2 on saved images
│       ├── rate_distortion.py        Per-category and severity analysis
│       ├── timing.py                 Timing and aggregation
│       ├── visualize.py              Qualitative image/box comparisons
│       ├── roi_codec_model.py        Optional ROI compression network
│       ├── roi_utils.py              ROI geometry and utilities
│       ├── tile_utils.py             Tile partitioning/priority utilities
│       ├── utils.py                  Shared image/file/geometry operations
│       ├── ablation_bdbr.py          CLI forwarding to BDBR analysis
│       └── ablation/
│           ├── __init__.py           Analysis package
│           ├── __main__.py           Alternative module entry point
│           ├── cli.py                CLI arguments and aggregation
│           ├── common.py             Codec, detector and method naming
│           ├── results.py            Read and normalize evaluation outputs
│           ├── curves.py             Pareto filtering and BDBR integration
│           ├── comparisons.py        Pairwise and equal-rate comparisons
│           └── tables.py             CSV/LaTeX tables, diagnostics
├── requirements.txt                 Python dependencies
├── environment.yml                  Conda environment configuration
└── .gitignore                        Ignore external assets and outputs
```

The public-facing repository no longer includes `tests/` or `tools/`. No documentation command depends on those directories.

### 🎛️ Main modules

| Module | Role |
|---|---|
| `scope/model.py` | U-Net-like pre-editor with FiLM-conditioned feature blocks, attention head, object residual and background residual. |
| `scope/conditioning.py` | Sampling family IDs and compression severity during training. |
| `scope/data.py` | COCO image crops and box-based masks for training. |
| `scope/codecs.py` | Differentiable Diff-JPEG, BMSHJ and BPG-like training proxies. |
| `scope/losses.py` | Attention, background-simplification, rate and residual terms. |
| `scope/teachers.py` | Frozen detector feature consistency between clean and edited-compressed images. |
| `scope/training.py` | Optimization, validation, model checkpointing and logging. |
| `scope/inference.py` | Load model/conditioning metadata and pre-edit images. |
| `scope/evaluation/evaluate.py` | Execute configured real-codec experiments and compute YOLO-based COCO AP. |
| `scope/evaluation/saved_rtdetr.py` | Evaluate saved reconstructions with RT-DETR. |
| `scope/evaluation/saved_multi_detector.py` | Evaluate saved reconstructions with TorchVision or Detectron2 detectors. |
| `scope/evaluation/rate_distortion.py` | Organize prediction-based AP by codec, detector, category and severity. |
| `scope/evaluation/ablation/` | Read existing summaries, form measured curves, compute BDBR, generate tables. |

The `roi_*` and `tile_utils.py` modules support optional ROI/tile transmission paths and are not necessary for the main codec lists in the supplied configs.

## ⚙️ 3. Dependencies and external assets

Install the core Python dependencies:

```bash
pip install -r requirements.txt
```

`environment.yml` contains an alternative Conda environment specification. The training launcher selects Detectron2 feature supervision; install a Detectron2 build compatible with your PyTorch/CUDA environment. The differentiable-JPEG backend used for training may also require a separate implementation. Real BPG and JPEG2000 use external codec command-line tools, typically:

```bash
command -v bpgenc
command -v bpgdec
command -v opj_compress
command -v opj_decompress
```

Missing binaries matter only for workflows that use those codecs. The repository does not distribute COCO images, frozen detector weights, or SCOPE training checkpoints.

**Expected default paths:**

```text
datasets/coco-2017/train/data/
datasets/coco-2017/train/labels.json
datasets/coco-2017/validation/data/
datasets/coco-2017/validation/labels.json
models/yolo26x.pt
runs/<training-run>/best_val.pt
```

The configured paths in `configs/*.yaml` and the environment variables in `scope/train_scope.sh` take precedence over these examples. COCO annotations must contain the image list, object annotations and category metadata needed for evaluation.

Legacy checkpoint directories named `preedit_*` are deliberately unchanged when the Python code was renamed to SCOPE. **Do not rename checkpoints solely for consistency.**

## 🎛️ 4. Conditioning, codec families and YAML configs

SCOPE is conditioned on a codec-family embedding and normalized severity `s ∈ [0, 1]` (higher values indicate stronger compression). The conditioning branch expands `s` into polynomial and high-severity basis terms, then uses an MLP to produce FiLM parameters for the image model.

| Actual codec | Training status | Conditioning at evaluation |
|---|---|---|
| JPEG | Seen proxy | `diff_jpeg` (`block_dct` in YAML) |
| BPG | Seen proxy | `bpg_like` (`intra_smooth`) |
| BMSHJ | Seen proxy | `bmshj` (`learned`) |
| Cheng | Unseen codec | Trained learned family (`bmshj`) |
| JPEG2000 | Unseen wavelet codec | **Three separate experiments** using `diff_jpeg`, `bmshj`, or `bpg_like` |

The checkpoint family mapping contains `diff_jpeg`, `bmshj`, and `bpg_like`; **`jp2_like` is not trained**. Thus, direct JPEG2000 baselines may appear in `configs/eval.yaml`, but SCOPE+JPEG2000 with `codec_family: wavelet` would request a non-existent embedding. The valid unseen-codec study instead uses `configs/ablation.yaml` with actual JPEG2000 compression and three *trained* family conditions.

### Configuration roles

| Config | Purpose |
|---|---|
| `configs/eval.yaml` | Main JPEG/BPG/BMSHJ/Cheng curves and baseline measurements, including direct JPEG2000 compression. |
| `configs/ablation.yaml` | JPEG2000 baselines plus the three trained conditioning families at several codec rates. |
| `configs/timing.yaml` | Forward-pass timing for each trained family across severity settings. |

The number of **active** experiments may change when entries are enabled or commented out. Use the YAML `experiments` list rather than a hard-coded experiment count. Preserved commented-out blocks do not execute. Shared YAML anchors such as `scope_common` and `codec_templates` are configuration conveniences; they do not add experiments by themselves.

Each experiment has a `name` (output folder) and `mode` (codec/transmission path). SCOPE experiments may also specify the checkpoint, architecture, `codec_family`, and optionally `condition.severity` (other cases derive severity from the codec parameter). An experiment using SCOPE must request a family that exists in the checkpoint mapping.

## 🏋️ 5. Training

The default launcher is co-located with its Python CLI:

```bash
bash scope/train_scope.sh
```

To supply another COCO location:

```bash
TRAIN_IMAGES=/data/coco/train2017 \
TRAIN_JSON=/data/coco/annotations/instances_train2017.json \
VAL_IMAGES=/data/coco/val2017 \
VAL_JSON=/data/coco/annotations/instances_val2017.json \
OUT_DIR=runs/scope_experiment \
CUDA_VISIBLE_DEVICES=0 BATCH=8 LR=1e-4 \
bash scope/train_scope.sh
```

`scope/train_scope.sh` is the **source of truth** for its experiment parameters: 512×512 crops, 40 epochs, scratch initialization, three codec proxies, family/severity sampling and Detectron2 Faster R-CNN feature supervision. It can be customized through environment variables or by editing the launcher.

Run `python -m scope.train_scope --help` for training flags. Trained runs save configuration, logs and checkpoints such as `best_val.pt` in the selected `OUT_DIR` (depending on completed validation and saving events).

## 📊 6. Real-codec evaluation

```bash
python -m scope.evaluation.evaluate --config configs/eval.yaml
python -m scope.evaluation.evaluate --config configs/ablation.yaml
```

For each COCO image the pipeline runs either direct compression or SCOPE pre-editing followed by a **real codec**, saves the reconstruction, performs YOLO inference, and aggregates predictions through COCOeval. The detector configuration comes from `models.server_model` in the YAML, typically `models/yolo26x.pt`.

Evaluation produces under the YAML `output_dir`:

```text
<output_dir>/
├── <experiment-name>/
│   ├── reconstructed/        # Reconstructed images used for detection
│   ├── attention_debug/      # Optional intermediate images
│   ├── predictions_coco.json # COCO-style detection predictions
│   ├── bandwidth.csv         # Per-image encoded bytes and rate
│   └── summary.json          # AP/AR, dataset_bpp, average size
└── all_results.csv           # Aggregate row per experiment
```

**Do not compute BDBR from reconstructed PNG sizes**: those PNGs are for downstream inference, not compressed codec bitstream sizes. Use the rate fields in the saved summaries or `bandwidth.csv`.

## 🎯 7. Additional detectors without recompression

Both modules below consume images already saved under `<source-eval-dir>/<experiment>/reconstructed/`, so they do not re-run SCOPE or compress the images again.

```bash
# RT-DETR
python -m scope.evaluation.saved_rtdetr \
  --source-eval-dir PATH_TO_YOLO_EVALUATION \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --detector-weights PATH_TO_RTDETR_WEIGHTS \
  --category-map coco91 \
  --output-dir PATH_TO_RTDETR_RESULTS

# Faster R-CNN (TorchVision example)
python -m scope.evaluation.saved_multi_detector \
  --source-eval-dir PATH_TO_YOLO_EVALUATION \
  --images-dir datasets/coco-2017/validation/data \
  --annotations-json datasets/coco-2017/validation/labels.json \
  --backend torchvision \
  --detector fasterrcnn_resnet50_fpn \
  --weights DEFAULT \
  --output-dir PATH_TO_FASTERRCNN_RESULTS
```

The paper uses Faster R-CNN with a **ResNeXt-101-32x8d-FPN** backbone, so the TorchVision ResNet-50 example is **not equivalent to the paper detector**. Select the appropriate Detectron2 backend/configuration for a paper-matched comparison. Check `--help` for backend choices and model arguments.

**Category IDs must be consistent.** COCO uses official category IDs with gaps; Ultralytics models use contiguous 80-class IDs. The `coco91` mapping matters when comparing AP across detectors.

## 📈 8. BDBR, interpolated points and tables

The BDBR entry point is `scope/evaluation/ablation_bdbr.py`, delegating to `scope/evaluation/ablation/cli.py`. The calculation pipeline is:

```text
Saved summary.json + bandwidth.csv
                 │
                 ▼
       results.py  — normalize measured AP and bpp
                 │
                 ▼
       curves.py   — group, deduplicate, Pareto-filter
                 │
                 ▼
      comparisons.py — equal-rate deltas and pairwise BDBR
                 │
                 ▼
       tables.py   — CSV, LaTeX, diagnostic export
```

### Example: SCOPE vs. direct compression

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

For the objective ablations, include separate source roots labeled `no_task:...` and `no_bg:...`, and select `--table-methods full,no_task,no_bg`.

The main BDBR analysis includes JPEG, BPG and BMSHJ, with **Cheng restricted to SCOPE versus baseline**. JPEG2000 cross-family results are separate. The published **TSW-pre** reference on Cheng comes from a digitized published curve and is **not automatically generated** from SCOPE experiments.

The BDBR sign convention is **negative = test curve achieves the same AP at a lower bitrate**. Curves are compared over their shared AP interval; invalid overlap and insufficient points are reported through status columns rather than fabricated values. Interpolated equal-rate samples and PCHIP integration points are numerical estimates, not newly evaluated images.

Outputs include:

```text
<analysis-out-dir>/
├── collected_results.csv
├── interpolated_equal_rate_points.csv
├── interpolation_status.csv
├── pairwise_equal_rate_and_bdbr.csv
├── bdbr_summary.csv
├── diagnostics.txt
└── latex/
    ├── bdbr_summary.tex
    └── interpolated_<codec>.tex
```

### 🔎 Data provenance warning

The result collector recursively searches **every `summary.json`** beneath the supplied input directories. This can include previous results even when their experiments have since been removed from active YAML configurations. Classification can also depend on experiment names and source labels.

Before reporting a number, examine `collected_results.csv`: verify `summary_path`, `experiment`, `codec`, `detector`, `curve_type`, `ablation_key`, `AP`, and `dataset_bpp` for every included point. Do not combine results from unrelated SCOPE checkpoints under one `full` curve. A plot labeled "SCOPE" does not itself establish that all underlying rows used the same training run.

## 🗂️ 9. Category/severity results and qualitative figures

`scope/evaluation/rate_distortion.py` works from saved `predictions_coco.json` and reports category/severity breakdowns; it is separate from the main BDBR computation.

```bash
python -m scope.evaluation.rate_distortion --help
python -m scope.evaluation.visualize --help
```

Standalone plotting scripts for paper-only overlays are **not part of the documented core package**. Where literature curves are digitized or transformed, their provenance and calibration must be disclosed rather than treating them as measured SCOPE results.

## ⏱️ 10. Runtime measurement

```bash
python -m scope.evaluation.timing \
  --config configs/timing.yaml \
  --warmup 20 --repeat 5 --limit 100
```

This benchmarks SCOPE pre-editing and associated conversion steps where requested. It does **not** benchmark the full image compression + decoder + detector pipeline. `--limit` reduces the timing workload, **not** the number of images used in a valid COCO AP evaluation.

## ✅ 11. Manual validation and reporting

With the public-facing `tests/` and `tools/` directories removed, use [🔎 TESTING.md](TESTING.md) for dependency checks, configuration validation, short smoke-run guidance, and result inspection. There is **no bundled automatic path/call auditor** in this layout.

For paper reproduction, record the exact COCO split, trained checkpoint, conditioning family, codec parameters, measured bpp, detector architecture/weights, BDBR interpolation method, and common AP interval. The presence of an output file alone does not establish metric correctness.

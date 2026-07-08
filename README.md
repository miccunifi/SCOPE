# Required Files and Directory Structure

This repository expects the following external assets to be placed in the corresponding folders before running the experiments.

## Checkpoints

Place the MMDetection Faster R-CNN checkpoint in:

```text
checkpoints/
```

Expected checkpoint:

```text
faster_rcnn_x101_32x4d_fpn_mstrain_3x_coco
```

This checkpoint corresponds to the MMDetection Faster R-CNN X101-32x4d-FPN model trained on COCO.

## Datasets

Place the COCO 2017 dataset in:

```text
datasets/
```

Expected dataset folder:

```text
datasets/coco-2017/
```

## Detectors

Place the detector weights in:

```text
detectors/
```

Expected detector weights:

```text
detectors/yolo26x
detectors/rtdetr-l
```

## Runs

The `runs/` folder contains the pretrained SCOPE weights used for the experiments.

These pretrained weights can be downloaded from Google Drive:

```text
https://drive.google.com/drive/folders/1LFbt_ZrvPJgfbSr174lVPzxeQEkMO0Uz?usp=sharing
```

After downloading, place the pretrained run folders inside:

```text
runs/
```

## Expected Structure

A minimal expected layout is:

```text
.
├── checkpoints/
│   └── faster_rcnn_x101_32x4d_fpn_mstrain_3x_coco
├── datasets/
│   └── coco-2017/
├── detectors/
│   ├── yolo26x
│   └── rtdetr-l
├── runs/
│   └── <downloaded_pretrained_scope_weights>
└── ...
```

Large assets such as datasets, checkpoints, pretrained weights, generated outputs, and plots are not meant to be committed to Git.

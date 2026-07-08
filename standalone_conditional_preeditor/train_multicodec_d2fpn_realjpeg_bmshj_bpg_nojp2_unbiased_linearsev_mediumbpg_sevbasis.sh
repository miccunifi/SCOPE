#!/usr/bin/env bash
set -euo pipefail

# Scratch training WITHOUT JPEG2000/wavelet family.
# Clean/unbiased setup:
#   - families are sampled uniformly: diff_jpeg, bmshj, bpg_like
#   - severity is a linear normalized scalar in [0,1]
#   - severity is sampled uniformly from a global linear grid
#   - no severity-dependent loss reweighting
#   - BPG uses the medium HEVC-like proxy: safer than strong, better than old blur/block
#
# Cheng remains an unseen learned-codec transfer test at evaluation time.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-/home/sdani/bandwidth_roi_detectionv2}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/standalone_conditional_preeditor}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_preedit_attention_conditional_bgreplace_d2fpn_realjpeg_realbmshj_q8_nojp2_unbiased_mediumbpg_sevbasis.py}"

TRAIN_IMAGES="${TRAIN_IMAGES:-${PROJECT_ROOT}/datasets/coco-2017/train/data}"
TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/datasets/coco-2017/train/labels.json}"

VAL_IMAGES="${VAL_IMAGES:-${PROJECT_ROOT}/datasets/coco-2017/validation/data}"
VAL_JSON="${VAL_JSON:-${PROJECT_ROOT}/datasets/coco-2017/validation/labels.json}"

RUN_NAME="${RUN_NAME:-preedit_d2fpn_nojp2_unbiased_linearsev_mediumbpg_sevbasis_scratch_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

# Detectron2 paper-like detector for COCO Det.
D2_CONFIG="${D2_CONFIG:-COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml}"
D2_WEIGHTS="${D2_WEIGHTS:-}"

# Linear severity grid: every level has equal probability.
# Expected average severity is ~0.5.
SEVERITY_LEVELS="${SEVERITY_LEVELS:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
VAL_SEVERITY_LEVELS="${VAL_SEVERITY_LEVELS:-0.0,0.2,0.4,0.6,0.8,1.0}"

CODEC_FAMILY_PROBS="${CODEC_FAMILY_PROBS:-diff_jpeg:0.333333,bmshj:0.333333,bpg_like:0.333334}"
BPG_PROXY_MODE="${BPG_PROXY_MODE:-medium}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-4}"

# Keep all severity-dependent dynamic weights disabled for an unbiased severity test.
TASK_SEVERITY_GAIN="${TASK_SEVERITY_GAIN:-0.0}"
RATE_SEVERITY_DROP="${RATE_SEVERITY_DROP:-0.0}"
BG_SEVERITY_DROP="${BG_SEVERITY_DROP:-0.0}"


echo "Project root:       ${PROJECT_ROOT}"
echo "Script dir:         ${SCRIPT_DIR}"
echo "Train script:       ${TRAIN_SCRIPT}"
echo "Train images:       ${TRAIN_IMAGES}"
echo "Train JSON:         ${TRAIN_JSON}"
echo "Val images:         ${VAL_IMAGES}"
echo "Val JSON:           ${VAL_JSON}"
echo "Init checkpoint:    none / scratch"
echo "Detectron2 cfg:     ${D2_CONFIG}"
echo "Detectron2 weights: ${D2_WEIGHTS:-model-zoo}"
echo "Output dir:         ${OUT_DIR}"
echo "CUDA devices:       ${CUDA_VISIBLE_DEVICES}"
echo "Codec families:     diff_jpeg,bmshj,bpg_like"
echo "Family probs:       ${CODEC_FAMILY_PROBS}"
echo "Severity levels:    ${SEVERITY_LEVELS}"
echo "Val severities:     ${VAL_SEVERITY_LEVELS}"
echo "Severity probs:     uniform"
echo "Severity weights:   task=${TASK_SEVERITY_GAIN}, rate_drop=${RATE_SEVERITY_DROP}, bg_drop=${BG_SEVERITY_DROP}"
echo "BPG proxy mode:     ${BPG_PROXY_MODE}"
echo "Severity condition: scalar severity, internal basis [s,s^2,s^3,hi50,hi75,hi90]"
echo "Batch / LR:         ${BATCH} / ${LR}"

test -f "${TRAIN_SCRIPT}" || { echo "ERROR: missing TRAIN_SCRIPT: ${TRAIN_SCRIPT}"; exit 1; }
test -d "${TRAIN_IMAGES}" || { echo "ERROR: missing TRAIN_IMAGES: ${TRAIN_IMAGES}"; exit 1; }
test -f "${TRAIN_JSON}" || { echo "ERROR: missing TRAIN_JSON: ${TRAIN_JSON}"; exit 1; }
test -d "${VAL_IMAGES}" || { echo "ERROR: missing VAL_IMAGES: ${VAL_IMAGES}"; exit 1; }
test -f "${VAL_JSON}" || { echo "ERROR: missing VAL_JSON: ${VAL_JSON}"; exit 1; }

mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${D2_WEIGHTS}" ]]; then
  EXTRA_ARGS+=(--detectron2-weights "${D2_WEIGHTS}")
fi

python "${TRAIN_SCRIPT}" \
  --images-dir "${TRAIN_IMAGES}" \
  --annotations-json "${TRAIN_JSON}" \
  --val-images-dir "${VAL_IMAGES}" \
  --val-annotations-json "${VAL_JSON}" \
  --output-dir "${OUT_DIR}" \
  \
  --codec-families diff_jpeg,bmshj,bpg_like \
  --codec-family-probs "${CODEC_FAMILY_PROBS}" \
  \
  --use-real-diffjpeg \
  --diffjpeg-quality-min 1 \
  --diffjpeg-quality-max 99 \
  \
  --bpg-qp-min 0 \
  --bpg-qp-max 51 \
  --bpg-proxy-mode "${BPG_PROXY_MODE}" \
  \
  --use-real-bmshj \
  --bmshj-model bmshj2018-hyperprior \
  --bmshj-quality-min 1 \
  --bmshj-quality-max 8 \
  --bmshj-bpp-scale 0.03 \
  --rate-tv-scale 1.0 \
  \
  --severity-min 0.0 \
  --severity-max 1.0 \
  --severity-levels "${SEVERITY_LEVELS}" \
  --severity-probs "" \
  --val-severity-levels "${VAL_SEVERITY_LEVELS}" \
  \
  --crop-size 512 \
  --object-crop-prob 0.60 \
  --crop-margin 4.0 \
  \
  --base-ch 48 \
  --family-emb-dim 8 \
  --cond-dim 64 \
  --max-delta-obj 0.09 \
  --max-delta-bg 0.03 \
  --attention-bias -4.0 \
  --bg-mode blur_replace \
  --bg-blur-kernel 51 \
  --bg-lowres-scale 16 \
  \
  --epochs 40 \
  --batch "${BATCH}" \
  --samples-per-epoch 12000 \
  --workers 4 \
  --lr "${LR}" \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --warmup-epochs 3 \
  --grad-clip 1.0 \
  --amp \
  \
  --attn-blur 3 \
  --det-mask-blur 41 \
  --attn-area-mult 1.0 \
  \
  --lambda-obj-l1 0.0 \
  --lambda-obj-edge 0.0 \
  --lambda-bg-l1 0.0 \
  --lambda-bg-flat 0.25 \
  --bg-flat-mode blur \
  --lambda-bg-smooth 0.35 \
  --lambda-rate 0.50 \
  --lambda-attn 1.00 \
  --lambda-attn-sparse 0.15 \
  --lambda-attn-area 0.50 \
  --lambda-delta 0.02 \
  --lambda-delta-tv 0.05 \
  \
  --task-severity-gain "${TASK_SEVERITY_GAIN}" \
  --rate-severity-drop "${RATE_SEVERITY_DROP}" \
  --bg-severity-drop "${BG_SEVERITY_DROP}" \
  --min-rate-mult 0.50 \
  --min-bg-mult 0.50 \
  \
  --detector-backend detectron2 \
  --detectron2-config "${D2_CONFIG}" \
  --detectron2-feature-names p2,p3,p4,p5,p6 \
  --detectron2-input-format BGR \
  --d2-fpn-mask-weight 0.70 \
  --d2-fpn-full-weight 0.30 \
  --lambda-det 4.0 \
  --detector-feature-loss smooth_l1 \
  \
  --val-every 5 \
  --val-steps 24 \
  --log-every 25 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "Done. Checkpoints:"
echo "  ${OUT_DIR}/last.pt"
echo "  ${OUT_DIR}/best_loss.pt"
echo "  ${OUT_DIR}/best_val.pt"
echo "  ${OUT_DIR}/train.log"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/vit_restore_dtw_s16/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
REPRESENTATION="${REPRESENTATION:-primary}"
ALIGNMENT_UNIT="${ALIGNMENT_UNIT:-word}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/Results/Evaluation/Yelda/vit_restore_dtw/${EVAL_SPLIT}/${REPRESENTATION}_${IMAGE_PREPROCESSING}_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_DIR}"
export MPLBACKEND=Agg

"${PYTHON_BIN}" -m py_compile \
  Evaluation/eval_yelda.py \
  Evaluation/eval_img_align_nw_diagnostic.py \
  Evaluation/visual_word_alignment.py
echo "Evaluation syntax preflight: OK"

exec "${PYTHON_BIN}" -u -m Evaluation.eval_yelda \
  --dataset "${DATASET}" \
  --weights "${WEIGHTS}" \
  --branch restoration \
  --representation "${REPRESENTATION}" \
  --alignment-unit "${ALIGNMENT_UNIT}" \
  --word-support-floor "${WORD_SUPPORT_FLOOR:-0.0}" \
  --image-preprocessing "${IMAGE_PREPROCESSING}" \
  --split "${EVAL_SPLIT}" \
  --training-samples "${TRAINING_SAMPLES:-6000}" \
  --split-seed "${SPLIT_SEED:-42}" \
  --n-samples "${N_SAMPLES:-100}" \
  --device "${DEVICE:-cuda}" \
  --score-mode "${SCORE_MODE:-raw}" \
  --local-weight "${LOCAL_WEIGHT:-0.5}" \
  --threshold "${THRESHOLD:-0.45}" \
  --gap "${GAP:--0.30}" \
  --output-dir "${RESULTS_DIR}" \
  "$@"

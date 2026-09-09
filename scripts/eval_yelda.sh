#!/usr/bin/env bash
# Foreground evaluation: run inside an allocated GPU terminal.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/vit_vlm_cross/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
REPRESENTATION="${REPRESENTATION:-joint}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/Results/Evaluation/Yelda/vit_vlm_cross/${EVAL_SPLIT}/${REPRESENTATION}_${IMAGE_PREPROCESSING}_$(date +%Y%m%d_%H%M%S)}"
cd "${PROJECT_DIR}"
export MPLBACKEND=Agg
exec "${PYTHON_BIN}" -u -m Evaluation.eval_yelda \
  --dataset "${DATASET}" --weights "${WEIGHTS}" \
  --branch cross --representation "${REPRESENTATION}" \
  --image-preprocessing "${IMAGE_PREPROCESSING}" \
  --split "${EVAL_SPLIT}" --training-samples "${TRAINING_SAMPLES:-6000}" \
  --split-seed "${SPLIT_SEED:-42}" --n-samples "${N_SAMPLES:-100}" \
  --device "${DEVICE:-cuda}" --score-mode "${SCORE_MODE:-raw}" \
  --local-weight "${LOCAL_WEIGHT:-0.5}" --threshold "${THRESHOLD:-0.45}" --gap "${GAP:--0.30}" \
  --output-dir "${RESULTS_DIR}" "$@"

#!/usr/bin/env bash
# Foreground evaluation: run inside an allocated GPU terminal.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/vit_vlm_letters/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
REPRESENTATION="${REPRESENTATION:-primary}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/Results/Evaluation/Yelda/vit_vlm_letters/${EVAL_SPLIT}/${REPRESENTATION}_$(date +%Y%m%d_%H%M%S)}"
cd "${PROJECT_DIR}"
export MPLBACKEND=Agg
exec "${PYTHON_BIN}" -u -m Evaluation.eval_yelda \
  --dataset "${DATASET}" --weights "${WEIGHTS}" \
  --branch hierarchy --representation "${REPRESENTATION}" \
  --split "${EVAL_SPLIT}" --training-samples "${TRAINING_SAMPLES:-6000}" \
  --split-seed "${SPLIT_SEED:-42}" --n-samples "${N_SAMPLES:-100}" \
  --device "${DEVICE:-cuda}" --score-mode "${SCORE_MODE:-raw}" \
  --threshold "${THRESHOLD:-0.45}" --gap "${GAP:--0.30}" \
  --output-dir "${RESULTS_DIR}" "$@"

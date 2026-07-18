#!/usr/bin/env bash
set -euo pipefail

weights="${1:-${WEIGHTS:-}}"
data_dir="${2:-${DATA_DIR:-DataSet/ArabicDataset}}"

if [[ -z "${weights}" ]]; then
  echo "Usage: $0 WEIGHTS_PATH [REAL_DATA_DIR]" >&2
  echo "Example: $0 Weights/JOB/model_latest.pth DataSet/ArabicDataset" >&2
  exit 2
fi

DATASET_TYPE=real \
REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-1}" \
python Evaluation/evaluate_retrieval.py \
  --dataset-type real \
  --data-dir "${data_dir}" \
  --weights "${weights}" \
  --split "${SPLIT:-test}" \
  --sides "${SIDES:-first}" \
  --n-samples "${N_SAMPLES:-64}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --score-mode "${SCORE_MODE:-d3tw}" \
  --device "${DEVICE:-auto}"

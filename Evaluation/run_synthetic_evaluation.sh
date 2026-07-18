#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE

weights="${1:-${WEIGHTS:-}}"
data_dir="${2:-${DATA_DIR:-DataSet/Synthetic_Arabic}}"

if [[ -z "${weights}" ]]; then
  echo "Usage: $0 WEIGHTS_PATH [SYNTHETIC_DATA_DIR]" >&2
  echo "Example: $0 Weights/JOB/model_latest.pth DataSet/Synthetic_Arabic" >&2
  exit 2
fi

python Evaluation/evaluate_retrieval.py \
  --dataset-type synthetic \
  --data-dir "${data_dir}" \
  --weights "${weights}" \
  --split "${SPLIT:-test}" \
  --sides "${SIDES:-first}" \
  --n-samples "${N_SAMPLES:-64}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --score-mode "${SCORE_MODE:-d3tw}" \
  --device "${DEVICE:-auto}"

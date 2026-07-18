#!/usr/bin/env bash
# Evaluate the improve_neg checkpoint on the real Arabic manifest dataset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATASET_TYPE=real
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/improve_model_win32_fastpair/model_latest.pth}"
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
exec bash scripts/eval/run_all_span2_evaluations.sh

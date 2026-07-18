#!/usr/bin/env bash
# Evaluate the improve_neg checkpoint on the same synthetic dataset used in training.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATASET_TYPE=synthetic
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
export WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/improve_model_win32_fastpair/model_latest.pth}"
exec bash scripts/eval/run_all_span2_evaluations.sh

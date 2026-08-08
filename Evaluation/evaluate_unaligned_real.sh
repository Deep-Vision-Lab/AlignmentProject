#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export DATASET_TYPE=real
export N_SAMPLES="${N_SAMPLES:-20}"
export DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
export REAL_SPLIT="${REAL_SPLIT:-test}"
exec bash "${ROOT}/Evaluation/run_unaligned_evaluation_project_out.sh"

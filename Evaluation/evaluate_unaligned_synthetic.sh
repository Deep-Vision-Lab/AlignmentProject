#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export DATASET_TYPE=synthetic
export N_SAMPLES="${N_SAMPLES:-20}"
export DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/AugmentedArabicDataset63}"
exec bash "${ROOT}/Evaluation/run_unaligned_evaluation_project_out.sh"

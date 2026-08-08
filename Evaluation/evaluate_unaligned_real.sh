#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export DATASET_TYPE=real
export N_SAMPLES="${N_SAMPLES:-20}"
export DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
export REAL_SPLIT="${REAL_SPLIT:-test}"

# Do not let a stale/exported checkpoint path poison a new evaluation run.
# The shared launcher will auto-select model_best.pth, model_latest.pth, or
# checkpoint_latest.pth when WEIGHTS is unset.
if [[ -n "${WEIGHTS:-}" && ! -f "${WEIGHTS}" ]]; then
  echo "WARNING: ignoring nonexistent WEIGHTS=${WEIGHTS}" >&2
  unset WEIGHTS
fi

exec bash "${ROOT}/Evaluation/run_unaligned_evaluation_project_out.sh"

#!/usr/bin/env bash
# Evaluate the improve_neg checkpoint on the same synthetic dataset used in training.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATASET_TYPE=synthetic
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
export WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/improve_model_win32_fastpair/model_latest.pth}"

# Resolve the frozen text backbone from the current checkout, the sibling
# AlignmentProject_clone cache, or ~/.cache/huggingface. Passing the concrete
# snapshot path prevents Transformers from attempting an online lookup.
if [[ -z "${ARABIC_TEXT_MODEL_NAME:-}" ]]; then
  export ARABIC_TEXT_MODEL_NAME="$(
    python scripts/eval/resolve_cached_text_model.py \
      --weights "${WEIGHTS}" \
      --project-dir "${PROJECT_DIR}"
  )"
fi

exec bash scripts/eval/run_all_span2_evaluations.sh

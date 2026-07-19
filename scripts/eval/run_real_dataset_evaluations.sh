#!/usr/bin/env bash
# Evaluate the improve_neg checkpoint on binarized real Arabic manifest data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

# Evaluation helpers live under scripts/eval but import project modules from the
# repository root. Keeping PYTHONPATH set is harmless even though the real-image
# preprocessor is now self-contained.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export DATASET_TYPE=real
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/improve_model_win32_fastpair/model_latest.pth}"
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"

# Real evaluation preprocessing.
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
# Normalize arbitrary scan polarity before applying the final requested inversion.
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
# Final saved polarity: black background and white text.
export REAL_INVERT_COLORS="${REAL_INVERT_COLORS:-1}"
export REAL_EVAL_HEIGHT="${REAL_EVAL_HEIGHT:-128}"
export REAL_EVAL_WIDTH="${REAL_EVAL_WIDTH:-1024}"

# Resolve the frozen text backbone from local caches before offline evaluation.
if [[ -z "${ARABIC_TEXT_MODEL_NAME:-}" ]]; then
  export ARABIC_TEXT_MODEL_NAME="$(
    python scripts/eval/resolve_cached_text_model.py \
      --weights "${WEIGHTS}" \
      --project-dir "${PROJECT_DIR}"
  )"
fi

exec bash scripts/eval/run_all_span2_evaluations.sh

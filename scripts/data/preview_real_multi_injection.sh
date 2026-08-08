#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/AugmentationPreview/real_multi_injection}"
NUM_PAIRS="${NUM_PAIRS:-8}"
SEED="${SEED:-42}"
SOURCE_PAIRS="${SOURCE_PAIRS:-300}"
PYTHON_BIN="${CONDA_ENV_PYTHON:-python}"

"${PYTHON_BIN}" scripts/data/preview_real_multi_injection_visual.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-pairs "${NUM_PAIRS}" \
  --source-pairs "${SOURCE_PAIRS}" \
  --seed "${SEED}"

printf '%s\n' \
  "Real multi-injection VISUAL preview created." \
  "  previews = ${OUTPUT_DIR}/previews" \
  "  summary  = ${OUTPUT_DIR}/preview_summary.json" \
  "  rows     = original + 1/2/3 shared injected regions" \
  "  red      = injected real-handwriting regions" \
  "  bbox     = not required for this visual-only fallback" \
  "  warning  = inspection only; do not use estimated regions for training"

#!/usr/bin/env bash
# Visual real-data augmentation preview that does not require bbox annotations.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/AugmentationPreview/real_visual}"
NUM_PAIRS="${NUM_PAIRS:-12}"
SEED="${SEED:-42}"
PYTHON_BIN="${CONDA_ENV_PYTHON:-python}"

"${PYTHON_BIN}" scripts/data/preview_real_visual_augmentations.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-pairs "${NUM_PAIRS}" \
  --seed "${SEED}"

printf '%s\n' \
  "Real visual augmentation preview created." \
  "  previews = ${OUTPUT_DIR}/previews" \
  "  summary  = ${OUTPUT_DIR}/preview_summary.json" \
  "  modes    = original, photometric, geometry, mixed" \
  "  bbox     = not required"

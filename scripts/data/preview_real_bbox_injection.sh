#!/usr/bin/env bash
# Focused gallery for the bbox-aware aligned injection augmentation.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/AugmentationPreview/real_bbox_injection}"
NUM_PAIRS="${NUM_PAIRS:-12}"
SEED="${SEED:-42}"

python scripts/data/augment_real_bbox_dataset.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-pairs "${NUM_PAIRS}" \
  --seed "${SEED}" \
  --modes aligned_injection,mixed \
  --preview \
  --overwrite

printf '%s\n' \
  "Focused real aligned-injection preview created." \
  "  gallery = ${OUTPUT_DIR}/previews" \
  "  per-preview JSON = same filename with .json extension" \
  "  summary = ${OUTPUT_DIR}/augmentation_summary.json"

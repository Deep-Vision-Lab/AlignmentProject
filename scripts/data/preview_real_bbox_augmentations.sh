#!/usr/bin/env bash
# Generate a small bbox-aware real augmentation gallery for manual inspection.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/AugmentationPreview/real_bbox}"
NUM_PAIRS="${NUM_PAIRS:-24}"
SEED="${SEED:-42}"
MODES="${MODES:-original,photometric,geometry,aligned_injection,unaligned_injection,mixed}"

python scripts/data/augment_real_bbox_dataset.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-pairs "${NUM_PAIRS}" \
  --seed "${SEED}" \
  --modes "${MODES}" \
  --preview \
  --overwrite

printf '%s\n' \
  "Real bbox-aware augmentation preview created." \
  "  gallery = ${OUTPUT_DIR}/previews" \
  "  manifest = ${OUTPUT_DIR}/dataset_manifest.jsonl" \
  "  summary = ${OUTPUT_DIR}/augmentation_summary.json" \
  "Red boxes are injected real bbox crops; blue boxes are retained/geometry-transformed boxes."

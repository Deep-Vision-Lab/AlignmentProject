#!/usr/bin/env bash
# Focused gallery for bbox-exact full-height real aligned injection.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/AugmentationPreview/real_bbox_injection}"
NUM_PAIRS="${NUM_PAIRS:-12}"
SEED="${SEED:-42}"
MIN_REGIONS="${MIN_REGIONS:-1}"
MAX_REGIONS="${MAX_REGIONS:-3}"
CONDA_ENV_PYTHON="${CONDA_ENV_PYTHON:-python}"

"${CONDA_ENV_PYTHON}" scripts/data/augment_real_bbox_strip_injection.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-pairs "${NUM_PAIRS}" \
  --seed "${SEED}" \
  --height 128 \
  --min-regions "${MIN_REGIONS}" \
  --max-regions "${MAX_REGIONS}" \
  --preview \
  --overwrite

printf '%s\n' \
  "BBox-exact real aligned-injection preview created." \
  "  gallery = ${OUTPUT_DIR}/previews" \
  "  dataset = ${OUTPUT_DIR}/pairs" \
  "  manifest = ${OUTPUT_DIR}/dataset_manifest.jsonl" \
  "  summary = ${OUTPUT_DIR}/augmentation_summary.json" \
  "  rule = full-height 128px strips cut only at complete subword bbox boundaries" \
  "  preview text = each preview PNG has a matching .txt file"

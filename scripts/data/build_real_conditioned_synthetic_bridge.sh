#!/usr/bin/env bash
# Build the real-conditioned synthetic bridge corpus once, outside GPU training.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v1}"
NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-4}"
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-4}"
MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-3}"
SEED="${SEED:-42}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --negatives-per-anchor "${NEGATIVES_PER_ANCHOR}"
  --negative-ngram "${NEGATIVE_NGRAM}"
  --min-overlap-word-chars "${MIN_OVERLAP_WORD_CHARS}"
  --seed "${SEED}"
  --max-anchors "${MAX_ANCHORS}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ -n "${BRIDGE_FONTS:-}" ]]; then
  args+=(--fonts "${BRIDGE_FONTS}")
fi

python scripts/data/build_real_conditioned_synthetic_bridge.py "${args[@]}"

[[ -s "${OUTPUT_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: bridge manifest was not produced: ${OUTPUT_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}
[[ -s "${OUTPUT_DIR}/metadata.json" ]] || {
  echo "ERROR: bridge metadata was not produced: ${OUTPUT_DIR}/metadata.json" >&2
  exit 2
}

python scripts/data/smoke_test_real_synthetic_bridge.py \
  --data-dir "${OUTPUT_DIR}" \
  --expected-negatives "${NEGATIVES_PER_ANCHOR}"

echo "Bridge dataset ready: ${OUTPUT_DIR}"

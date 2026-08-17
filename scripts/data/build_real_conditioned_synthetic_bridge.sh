#!/usr/bin/env bash
# Build the real-conditioned synthetic bridge V2 corpus once, outside GPU training.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-4}"
# Content-clean negative defaults:
#   1) no complete normalized word may be shared with the real anchor; and
#   2) no normalized three-character sequence may be shared.
# Individual letters/bigrams may repeat so negatives remain realistic Arabic.
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}"
MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}"
MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}"
SEGMENT_GAP_MIN_PX="${SEGMENT_GAP_MIN_PX:-12}"
SEGMENT_GAP_MAX_PX="${SEGMENT_GAP_MAX_PX:-28}"
SEED="${SEED:-42}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --negatives-per-anchor "${NEGATIVES_PER_ANCHOR}"
  --negative-ngram "${NEGATIVE_NGRAM}"
  --min-overlap-word-chars "${MIN_OVERLAP_WORD_CHARS}"
  --max-shared-islands "${MAX_SHARED_ISLANDS}"
  --segment-gap-min-px "${SEGMENT_GAP_MIN_PX}"
  --segment-gap-max-px "${SEGMENT_GAP_MAX_PX}"
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
[[ -d "${OUTPUT_DIR}/masks" ]] || {
  echo "ERROR: bridge mask directory was not produced: ${OUTPUT_DIR}/masks" >&2
  exit 2
}

python scripts/data/smoke_test_real_synthetic_bridge.py \
  --data-dir "${OUTPUT_DIR}" \
  --expected-negatives "${NEGATIVES_PER_ANCHOR}"

echo "Bridge V2 dataset ready: ${OUTPUT_DIR}"

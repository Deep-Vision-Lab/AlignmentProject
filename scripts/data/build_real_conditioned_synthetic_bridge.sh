#!/usr/bin/env bash
# Build RealSyntheticBridge V3 once, offline on CPU.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-4}"
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}"
MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}"
MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}"
SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-5}"
SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-12}"
MAX_SENTENCE_CHARS="${MAX_SENTENCE_CHARS:-100}"
MAX_FONT_CHUNK_WORDS="${MAX_FONT_CHUNK_WORDS:-3}"
SEGMENT_GAP_MIN_PX="${SEGMENT_GAP_MIN_PX:-8}"
SEGMENT_GAP_MAX_PX="${SEGMENT_GAP_MAX_PX:-20}"
BLUR_PROB="${BLUR_PROB:-0.65}"
BLUR_MAX_RADIUS="${BLUR_MAX_RADIUS:-1.15}"
NOISE_PROB="${NOISE_PROB:-0.80}"
NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-9.0}"
CONTRAST_MIN="${CONTRAST_MIN:-0.88}"
CONTRAST_MAX="${CONTRAST_MAX:-1.14}"
BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.90}"
BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.08}"
SEED="${SEED:-42}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --negatives-per-anchor "${NEGATIVES_PER_ANCHOR}"
  --negative-ngram "${NEGATIVE_NGRAM}"
  --min-overlap-word-chars "${MIN_OVERLAP_WORD_CHARS}"
  --max-shared-islands "${MAX_SHARED_ISLANDS}"
  --sentence-min-words "${SENTENCE_MIN_WORDS}"
  --sentence-max-words "${SENTENCE_MAX_WORDS}"
  --max-sentence-chars "${MAX_SENTENCE_CHARS}"
  --max-font-chunk-words "${MAX_FONT_CHUNK_WORDS}"
  --segment-gap-min-px "${SEGMENT_GAP_MIN_PX}"
  --segment-gap-max-px "${SEGMENT_GAP_MAX_PX}"
  --blur-prob "${BLUR_PROB}"
  --blur-max-radius "${BLUR_MAX_RADIUS}"
  --noise-prob "${NOISE_PROB}"
  --noise-sigma-max "${NOISE_SIGMA_MAX}"
  --contrast-min "${CONTRAST_MIN}"
  --contrast-max "${CONTRAST_MAX}"
  --brightness-min "${BRIGHTNESS_MIN}"
  --brightness-max "${BRIGHTNESS_MAX}"
  --seed "${SEED}"
  --max-anchors "${MAX_ANCHORS}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then args+=(--overwrite); fi
if [[ -n "${BRIDGE_FONTS:-}" ]]; then args+=(--fonts "${BRIDGE_FONTS}"); fi

python scripts/data/build_real_conditioned_synthetic_bridge_v3.py "${args[@]}"

[[ -s "${OUTPUT_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing bridge manifest" >&2; exit 2; }
[[ -s "${OUTPUT_DIR}/metadata.json" ]] || { echo "ERROR: missing bridge metadata" >&2; exit 2; }
[[ -d "${OUTPUT_DIR}/masks" ]] || { echo "ERROR: missing bridge masks" >&2; exit 2; }

python scripts/data/smoke_test_real_synthetic_bridge_v3.py \
  --data-dir "${OUTPUT_DIR}" \
  --expected-negatives "${NEGATIVES_PER_ANCHOR}"

echo "Bridge V3 dataset ready: ${OUTPUT_DIR}"

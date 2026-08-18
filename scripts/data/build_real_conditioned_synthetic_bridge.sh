#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_DIR}"; export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"; OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-8}"; NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}"; MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}"; MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}"
MIN_POSITIVE_CHARS="${MIN_POSITIVE_CHARS:-4}"; MAX_PHRASE_CHARS="${MAX_PHRASE_CHARS:-6}"; MAX_PHRASE_WORDS="${MAX_PHRASE_WORDS:-2}"
SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-6}"; SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-10}"; MIN_SENTENCE_CHARS="${MIN_SENTENCE_CHARS:-28}"; MAX_SENTENCE_CHARS="${MAX_SENTENCE_CHARS:-55}"
MAX_FONT_CHUNK_WORDS="${MAX_FONT_CHUNK_WORDS:-2}"; FONT_SIZE="${FONT_SIZE:-60}"; MIN_FONT_SIZE="${MIN_FONT_SIZE:-42}"; MAX_FONT_SIZE="${MAX_FONT_SIZE:-64}"; MIN_LINE_FILL_RATIO="${MIN_LINE_FILL_RATIO:-0.90}"
PADDING="${PADDING:-8}"; SEGMENT_GAP_MIN_PX="${SEGMENT_GAP_MIN_PX:-2}"; SEGMENT_GAP_MAX_PX="${SEGMENT_GAP_MAX_PX:-6}"
BLUR_PROB="${BLUR_PROB:-0.65}"; BLUR_MAX_RADIUS="${BLUR_MAX_RADIUS:-1.15}"; NOISE_PROB="${NOISE_PROB:-0.80}"; NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-9.0}"
CONTRAST_MIN="${CONTRAST_MIN:-0.88}"; CONTRAST_MAX="${CONTRAST_MAX:-1.14}"; BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.90}"; BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.08}"; SEED="${SEED:-42}"; MAX_ANCHORS="${MAX_ANCHORS:-0}"
args=(--data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --negatives-per-anchor "${NEGATIVES_PER_ANCHOR}" --negative-ngram "${NEGATIVE_NGRAM}" --min-overlap-word-chars "${MIN_OVERLAP_WORD_CHARS}" --max-shared-islands "${MAX_SHARED_ISLANDS}" --min-positive-chars "${MIN_POSITIVE_CHARS}" --max-phrase-chars "${MAX_PHRASE_CHARS}" --max-phrase-words "${MAX_PHRASE_WORDS}" --sentence-min-words "${SENTENCE_MIN_WORDS}" --sentence-max-words "${SENTENCE_MAX_WORDS}" --min-sentence-chars "${MIN_SENTENCE_CHARS}" --max-sentence-chars "${MAX_SENTENCE_CHARS}" --max-font-chunk-words "${MAX_FONT_CHUNK_WORDS}" --font-size "${FONT_SIZE}" --min-font-size "${MIN_FONT_SIZE}" --max-font-size "${MAX_FONT_SIZE}" --min-line-fill-ratio "${MIN_LINE_FILL_RATIO}" --padding "${PADDING}" --segment-gap-min-px "${SEGMENT_GAP_MIN_PX}" --segment-gap-max-px "${SEGMENT_GAP_MAX_PX}" --blur-prob "${BLUR_PROB}" --blur-max-radius "${BLUR_MAX_RADIUS}" --noise-prob "${NOISE_PROB}" --noise-sigma-max "${NOISE_SIGMA_MAX}" --contrast-min "${CONTRAST_MIN}" --contrast-max "${CONTRAST_MAX}" --brightness-min "${BRIGHTNESS_MIN}" --brightness-max "${BRIGHTNESS_MAX}" --seed "${SEED}" --max-anchors "${MAX_ANCHORS}")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite); [[ -n "${BRIDGE_FONTS:-}" ]] && args+=(--fonts "${BRIDGE_FONTS}")
python scripts/data/build_real_conditioned_synthetic_bridge_v3_dense.py "${args[@]}"
python scripts/data/organize_real_synthetic_bridge_v3.py --data-dir "${OUTPUT_DIR}"
python scripts/data/create_bridge_v3_category_folders.py --data-dir "${OUTPUT_DIR}"
python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${OUTPUT_DIR}" --expected-negatives "${NEGATIVES_PER_ANCHOR}"
python scripts/data/validate_bridge_v3_font_size.py --data-dir "${OUTPUT_DIR}" --min-font-size "${MIN_FONT_SIZE}"
python scripts/data/validate_bridge_v3_dense_layout.py --data-dir "${OUTPUT_DIR}" --min-recorded-fill "${MIN_LINE_FILL_RATIO}" --min-pixel-span 0.84 --expected-negatives "${NEGATIVES_PER_ANCHOR}"
echo "Bridge V3 dataset ready: ${OUTPUT_DIR}"; echo "Human folders: ${OUTPUT_DIR}/{real,positive,negative}/<anchor_id>/"

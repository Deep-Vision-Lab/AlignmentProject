#!/usr/bin/env bash
set -euo pipefail

python generateDataArabicThreeFontsRandomized.py \
  --font-dir "${FONT_DIR:-Fonts}" \
  --fonts \
    "${FONT_1:-Amiri-Regular.ttf}" \
    "${FONT_2:-Arslan_Wessam_B.ttf}" \
    "${FONT_3:-DejaVuSans-Bold.ttf}" \
  --font-count "${FONT_COUNT:-3}" \
  --samples-per-font "${SAMPLES_PER_FONT:-3000}" \
  --output-dir "${OUTPUT_DIR:-DataSet/Synthetic_Arabic_Three_Font_Augmented}" \
  --original-ratio "${ORIGINAL_RATIO:-0.10}" \
  --cross-injection-ratio "${CROSS_INJECTION_RATIO:-0.25}" \
  --aligned-unaligned-ratio "${ALIGNED_UNALIGNED_RATIO:-0.20}" \
  --two-aligned-parts-ratio "${TWO_ALIGNED_PARTS_RATIO:-0.25}" \
  --three-aligned-parts-ratio "${THREE_ALIGNED_PARTS_RATIO:-0.20}" \
  --mixed-font-injection-prob "${MIXED_FONT_INJECTION_PROB:-0.50}" \
  --segment-gap-min "${SEGMENT_GAP_MIN:-2}" \
  --segment-gap-max "${SEGMENT_GAP_MAX:-6}" \
  --target-fill-ratio "${TARGET_FILL_RATIO:-0.96}" \
  "$@"

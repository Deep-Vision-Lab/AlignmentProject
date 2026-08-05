#!/usr/bin/env bash
set -euo pipefail

python generateDataArabicThreeFontsCompatible.py \
  --font-dir "${FONT_DIR:-Fonts}" \
  --font-count "${FONT_COUNT:-3}" \
  --samples-per-font "${SAMPLES_PER_FONT:-3000}" \
  --output-dir "${OUTPUT_DIR:-DataSet/Synthetic_Arabic_Three_Font_Augmented}" \
  --original-ratio "${ORIGINAL_RATIO:-0.25}" \
  --cross-injection-ratio "${CROSS_INJECTION_RATIO:-0.45}" \
  --aligned-unaligned-ratio "${ALIGNED_UNALIGNED_RATIO:-0.30}" \
  --mixed-font-injection-prob "${MIXED_FONT_INJECTION_PROB:-0.30}" \
  --segment-gap-min "${SEGMENT_GAP_MIN:-4}" \
  --segment-gap-max "${SEGMENT_GAP_MAX:-10}" \
  --target-fill-ratio "${TARGET_FILL_RATIO:-0.94}" \
  --skip-matrices \
  "$@"

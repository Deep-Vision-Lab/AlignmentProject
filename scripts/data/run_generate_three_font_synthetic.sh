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
  --font-size "${FONT_SIZE:-84}" \
  --min-text-chars "${MIN_TEXT_CHARS:-85}" \
  --max-text-chars "${MAX_TEXT_CHARS:-120}" \
  --original-ratio "${ORIGINAL_RATIO:-0.10}" \
  --cross-injection-ratio "${CROSS_INJECTION_RATIO:-0.25}" \
  --aligned-unaligned-ratio "${ALIGNED_UNALIGNED_RATIO:-0.20}" \
  --two-aligned-parts-ratio "${TWO_ALIGNED_PARTS_RATIO:-0.25}" \
  --three-aligned-parts-ratio "${THREE_ALIGNED_PARTS_RATIO:-0.20}" \
  --mixed-font-injection-prob "${MIXED_FONT_INJECTION_PROB:-0.50}" \
  --noise-prob "${NOISE_PROB:-0.75}" \
  --original-noise-scale "${ORIGINAL_NOISE_SCALE:-0.20}" \
  --gaussian-noise-prob "${GAUSSIAN_NOISE_PROB:-0.90}" \
  --gaussian-noise-std-min "${GAUSSIAN_NOISE_STD_MIN:-2.0}" \
  --gaussian-noise-std-max "${GAUSSIAN_NOISE_STD_MAX:-10.0}" \
  --salt-pepper-noise-prob "${SALT_PEPPER_NOISE_PROB:-0.55}" \
  --salt-pepper-density-min "${SALT_PEPPER_DENSITY_MIN:-0.0005}" \
  --salt-pepper-density-max "${SALT_PEPPER_DENSITY_MAX:-0.0035}" \
  --blur-prob "${BLUR_PROB:-0.25}" \
  --blur-radius-min "${BLUR_RADIUS_MIN:-0.15}" \
  --blur-radius-max "${BLUR_RADIUS_MAX:-0.75}" \
  --segment-gap-min "${SEGMENT_GAP_MIN:-2}" \
  --segment-gap-max "${SEGMENT_GAP_MAX:-6}" \
  --target-fill-ratio "${TARGET_FILL_RATIO:-0.98}" \
  "$@"

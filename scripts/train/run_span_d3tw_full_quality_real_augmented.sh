#!/usr/bin/env bash
# Submit full-quality real-data training with training-only augmentation.
#
# This launcher exists only on branch real_data_augmentation. It leaves the
# improve_neg real-data loader and evaluation-axis settings untouched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_full_quality_real_augmented.py}"
export JOB_ID="${JOB_ID:-real_arabic_fullquality_span3_augmented}"

# Master switch. Validation/test are never augmented even when this is enabled.
export REAL_AUGMENT="${REAL_AUGMENT:-1}"

# Smart paired RTL stitching. For logical text `first second`, the image is laid
# out physically as [second | gap | first], which is correct for Arabic RTL.
export REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0.25}"
export REAL_AUG_STITCH_POOL_SIZE="${REAL_AUG_STITCH_POOL_SIZE:-32}"
export REAL_AUG_STITCH_MAX_TEXT_CHARS="${REAL_AUG_STITCH_MAX_TEXT_CHARS:-120}"
export REAL_AUG_STITCH_PREFER_ADJACENT="${REAL_AUG_STITCH_PREFER_ADJACENT:-1}"
export REAL_AUG_STITCH_GAP_MIN="${REAL_AUG_STITCH_GAP_MIN:-0.08}"
export REAL_AUG_STITCH_GAP_MAX="${REAL_AUG_STITCH_GAP_MAX:-0.18}"

# Conservative pre-binarization scan variation.
export REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.85}"
export REAL_AUG_ROTATE_DEG="${REAL_AUG_ROTATE_DEG:-1.25}"
export REAL_AUG_TRANSLATE_X="${REAL_AUG_TRANSLATE_X:-0.012}"
export REAL_AUG_TRANSLATE_Y="${REAL_AUG_TRANSLATE_Y:-0.035}"
export REAL_AUG_BRIGHTNESS="${REAL_AUG_BRIGHTNESS:-0.12}"
export REAL_AUG_CONTRAST="${REAL_AUG_CONTRAST:-0.18}"
export REAL_AUG_BLUR_PROB="${REAL_AUG_BLUR_PROB:-0.15}"
export REAL_AUG_BLUR_RADIUS="${REAL_AUG_BLUR_RADIUS:-0.8}"
export REAL_AUG_NOISE_PROB="${REAL_AUG_NOISE_PROB:-0.18}"
export REAL_AUG_NOISE_STD="${REAL_AUG_NOISE_STD:-5.0}"

# Light post-binarization handwriting/scan defects.
export REAL_AUG_MORPH_PROB="${REAL_AUG_MORPH_PROB:-0.25}"
export REAL_AUG_SPECKLE_PROB="${REAL_AUG_SPECKLE_PROB:-0.12}"
export REAL_AUG_SPECKLE_FRACTION="${REAL_AUG_SPECKLE_FRACTION:-0.0006}"

exec bash "${SCRIPT_DIR}/run_span_d3tw_full_quality_real.sh"

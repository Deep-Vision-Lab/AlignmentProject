#!/usr/bin/env bash
# Phase-4 real fine-tuning recipe: use safe no_shared_content rows as explicit
# image-image negatives while retaining the canonical image-text objectives.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REAL_USE_EXTRA_NO_SHARED="${REAL_USE_EXTRA_NO_SHARED:-1}"
export REAL_EXTRA_EXCLUDE_EVAL_PAGES="${REAL_EXTRA_EXCLUDE_EVAL_PAGES:-1}"
export USE_NO_SHARED_IMAGE_NEGATIVES="${USE_NO_SHARED_IMAGE_NEGATIVES:-1}"
export NO_SHARED_IMAGE_NEGATIVE_WEIGHT="${NO_SHARED_IMAGE_NEGATIVE_WEIGHT:-0.25}"
export NO_SHARED_IMAGE_NEGATIVE_MAX_COSINE="${NO_SHARED_IMAGE_NEGATIVE_MAX_COSINE:-0.45}"
export NO_SHARED_IMAGE_NEGATIVE_TOP_K="${NO_SHARED_IMAGE_NEGATIVE_TOP_K:-8}"
export NO_SHARED_IMAGE_NEGATIVE_MIN_CHARS="${NO_SHARED_IMAGE_NEGATIVE_MIN_CHARS:-2}"
export NO_SHARED_IMAGE_NEGATIVE_MAX_SAMPLES_PER_BATCH="${NO_SHARED_IMAGE_NEGATIVE_MAX_SAMPLES_PER_BATCH:-8}"

# Keep the image-text negative recipe fixed for this ablation. Override only if
# explicitly requested so the effect of the new image-image objective is isolated.
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"

exec bash "${SCRIPT_DIR}/run_real_finetune.sh"

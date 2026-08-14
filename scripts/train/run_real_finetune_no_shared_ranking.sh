#!/usr/bin/env bash
# Corrected final real-data fine-tune: rank genuine cross-manuscript matches
# above hard no_shared_content mismatches without an absolute cosine target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Final adaptation uses the original real distribution, not online augmentation.
export AUGMENT="${AUGMENT:-0}"
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
export REAL_USE_EXTRA_NO_SHARED="${REAL_USE_EXTRA_NO_SHARED:-1}"
export REAL_EXTRA_EXCLUDE_EVAL_PAGES="${REAL_EXTRA_EXCLUDE_EVAL_PAGES:-1}"

export NO_SHARED_IMAGE_OBJECTIVE="ranking"
export USE_NO_SHARED_IMAGE_RANKING="${USE_NO_SHARED_IMAGE_RANKING:-1}"
export NO_SHARED_RANKING_WEIGHT="${NO_SHARED_RANKING_WEIGHT:-0.20}"
export NO_SHARED_RANKING_MARGIN="${NO_SHARED_RANKING_MARGIN:-0.10}"
export NO_SHARED_RANKING_TOP_K="${NO_SHARED_RANKING_TOP_K:-8}"
export NO_SHARED_RANKING_MIN_CHARS="${NO_SHARED_RANKING_MIN_CHARS:-2}"
export NO_SHARED_RANKING_MAX_POS_SAMPLES_PER_BATCH="${NO_SHARED_RANKING_MAX_POS_SAMPLES_PER_BATCH:-8}"
export NO_SHARED_RANKING_MAX_NEG_SAMPLES_PER_BATCH="${NO_SHARED_RANKING_MAX_NEG_SAMPLES_PER_BATCH:-8}"

# Isolate this change: keep the established image-text negative recipe.
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"

exec bash "${SCRIPT_DIR}/run_real_finetune.sh"

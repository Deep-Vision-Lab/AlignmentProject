#!/usr/bin/env bash
# Connected-Arabic-subword experiment launcher.
# Run from the repository root on the login node:
#   JOB_ID=<name> PRETRAINED_WEIGHTS=<checkpoint> \
#   bash scripts/train/run_connected_subword_finetune.sh
set -euo pipefail

export SPAN_TOKENIZATION_MODE="${SPAN_TOKENIZATION_MODE:-connected_subword}"
export SPAN_USE_BLANK_TRANSITIONS="${SPAN_USE_BLANK_TRANSITIONS:-1}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-16}"
export SPAN_CONNECTED_WINDOWS_PER_CHAR="${SPAN_CONNECTED_WINDOWS_PER_CHAR:-3}"
export SPAN_CONNECTED_EXTRA_WINDOWS="${SPAN_CONNECTED_EXTRA_WINDOWS:-1}"
export SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS="${SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS:-2}"
export SPAN_SPACE_MAX_WINDOWS="${SPAN_SPACE_MAX_WINDOWS:-3}"

# Connected units are much shorter than character-level text sequences.
export SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-32}"
export SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-128}"

# The old optimized safety rule capped character spans at three windows. This
# experiment uses a different unit: a connected Arabic run can legitimately
# occupy many windows. Per-unit caps are enforced by connected_subword_mode.py.
export ALLOW_UNSAFE_SPAN_CONFIG=1

# Filter real positives using the active tokenizer's exact state count. The
# connected-aware implementation counts subwords, explicit boundaries, and
# spaces, preventing native long lines from reaching Span-DTW with more text
# states than available image windows.
export REAL_FILTER_INFEASIBLE_SPAN_DTW="${REAL_FILTER_INFEASIBLE_SPAN_DTW:-1}"

# Semantic image-image matching compares connected runs, not structural tokens.
export PAIR_COMPOSITION_MAX_REGIONS="${PAIR_COMPOSITION_MAX_REGIONS:-1}"
export PAIR_COMPOSITION_MAX_CHARS="${PAIR_COMPOSITION_MAX_CHARS:-24}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-connected-subword}"

exec bash scripts/train/run_real_finetune.sh

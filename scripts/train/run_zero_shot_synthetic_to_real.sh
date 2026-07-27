#!/usr/bin/env bash
# Strict zero-shot launcher: train only on synthetic data, evaluate on real data.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)}"

export PROJECT_DIR
export DATASET_TYPE="synthetic"
export ZERO_SHOT_PROFILE="${ZERO_SHOT_PROFILE:-1}"

# Shared synthetic/real geometry. The runtime patch keeps actual ink at a fixed
# height and compresses only the horizontal axis when a line is wider than 1024.
# This avoids the tiny-line failure caused by fitting both dimensions jointly.
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"

# sitecustomize is imported by every torchrun Python rank before train_optimized,
# ensuring all GPUs use exactly the same corrected preprocessing implementation.
ZERO_SHOT_RUNTIME_DIR="${PROJECT_DIR}/zero_shot_runtime"
export PYTHONPATH="${ZERO_SHOT_RUNTIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Synthetic manuscript domain randomization.
export SYNTHETIC_MANUSCRIPT_AUGMENT="${SYNTHETIC_MANUSCRIPT_AUGMENT:-1}"
export SYNTHETIC_AUGMENT_PROBABILITY="${SYNTHETIC_AUGMENT_PROBABILITY:-0.85}"
export SYNTHETIC_CLEAN_PROBABILITY="${SYNTHETIC_CLEAN_PROBABILITY:-0.20}"
export SYNTHETIC_BINARIZE="${SYNTHETIC_BINARIZE:-1}"
export SYNTHETIC_BINARIZE_METHOD="${SYNTHETIC_BINARIZE_METHOD:-random}"
export SYNTHETIC_BINARIZE_THRESHOLD="${SYNTHETIC_BINARIZE_THRESHOLD:-180}"
export SYNTHETIC_THRESHOLD_JITTER="${SYNTHETIC_THRESHOLD_JITTER:-24}"

# Reuse both local and three-window grouped features in local supervision.
export ZERO_SHOT_GROUPED_BLEND="${ZERO_SHOT_GROUPED_BLEND:-0.50}"

# Keep ImageNet BatchNorm statistics instead of learning synthetic-only moments.
export ZERO_SHOT_NORM_MODE="${ZERO_SHOT_NORM_MODE:-frozen-bn}"

# Zero-shot training batch size per GPU.
export BATCH_SIZE="${BATCH_SIZE:-64}"

# Local discrimination is especially important for transfer.
export USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.35}"
export LOCAL_HARD_NEGATIVE_MIN_INK="${LOCAL_HARD_NEGATIVE_MIN_INK:-0.02}"

export JOB_ID="${JOB_ID:-synthetic_arabic_zero_shot_${NUM_SAMPLES:-8000}_gpu${NUM_GPUS:-2}}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-align_zero_shot}"

exec bash "${PROJECT_DIR}/scripts/train/run_span_d3tw_optimized.sh"

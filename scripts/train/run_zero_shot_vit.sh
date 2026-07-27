#!/usr/bin/env bash
# Strict zero-shot ViT experiment: synthetic training, real evaluation.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_zero_shot_vit.sh" >&2
  echo "Override settings through environment variables." >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)}"

export PROJECT_DIR
export VISUAL_ENCODER_TYPE="vit"

# Pure patch Transformer ablation: no ResNet, no BiLSTM, no Conv1d grouping.
export USE_BILSTM="0"
export USE_LOCAL_WINDOW_GROUPING="0"
export ZERO_SHOT_GROUPED_BLEND="0.0"

# One token per existing full-height sliding window.
export VIT_INPUT_HEIGHT="${VIT_INPUT_HEIGHT:-128}"
export VIT_LAYERS="${VIT_LAYERS:-4}"
export VIT_HEADS="${VIT_HEADS:-4}"
export VIT_MLP_DIM="${VIT_MLP_DIM:-512}"
export VIT_DROPOUT="${VIT_DROPOUT:-0.10}"
export VIT_MAX_TOKENS="${VIT_MAX_TOKENS:-256}"

# ViT uses LayerNorm throughout, so there are no BatchNorm statistics to freeze.
export ZERO_SHOT_NORM_MODE="${ZERO_SHOT_NORM_MODE:-train-bn}"
export USE_CHANNELS_LAST="0"
export TORCH_COMPILE_VISUAL="${TORCH_COMPILE_VISUAL:-0}"

# Keep the requested 64 samples per GPU and an effective global batch of 128
# on two GPUs unless explicitly overridden.
export BATCH_SIZE="${BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

export JOB_ID="${JOB_ID:-synthetic_arabic_zero_shot_vit_${NUM_SAMPLES:-8000}_gpu${NUM_GPUS:-2}}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-align_zero_vit}"

exec bash "${PROJECT_DIR}/scripts/train/run_zero_shot_synthetic_to_real.sh"

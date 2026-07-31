#!/usr/bin/env bash
# Submit real-dataset fine-tuning through the canonical multi-GPU launcher.
# Run this file with bash from the login node; do not submit it with sbatch.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: configure with environment variables, then run:" >&2
  echo "  bash scripts/train/run_real_finetune.sh" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: do not run 'sbatch scripts/train/run_real_finetune.sh'." >&2
  echo "Run it with bash from the login node; it submits the real Slurm job itself." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID to the output weights-folder name.}"
: "${PRETRAINED_WEIGHTS:?Set PRETRAINED_WEIGHTS to model_latest.pth.}"

[[ -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2
  exit 2
}

AUGMENT="${AUGMENT:-1}"
case "${AUGMENT}" in
  0|1) ;;
  *) echo "ERROR: AUGMENT must be 0 or 1, got ${AUGMENT}." >&2; exit 2 ;;
esac

NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_GPUS must be a positive integer, got ${NUM_GPUS}." >&2
  exit 2
fi
if ! [[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GLOBAL_BATCH_SIZE must be a positive integer, got ${GLOBAL_BATCH_SIZE}." >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi
BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS))

# The optimized Arabic span encoder is designed for truthful visible cores of at
# most two characters. Longer cores can make one small image region represent an
# entire phrase and are deliberately rejected by training_optimizations.py.
REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-2}"
if ! [[ "${REAL_MAX_TEXT_SPAN_CHARS}" =~ ^[0-9]+$ ]] \
  || (( REAL_MAX_TEXT_SPAN_CHARS < 1 || REAL_MAX_TEXT_SPAN_CHARS > 2 )); then
  echo "ERROR: REAL_MAX_TEXT_SPAN_CHARS must be 1 or 2 for truthful alignment." >&2
  echo "Do not enlarge spans to repair long stitched transcripts." >&2
  exit 2
fi

# The default 1024-pixel line with window size 32 and stride 16 yields 63 image
# windows. Positive transcripts that need more transitions are filtered after
# the deterministic group split, so they cannot crash training or validation.
REAL_FILTER_INFEASIBLE_SPAN_DTW="${REAL_FILTER_INFEASIBLE_SPAN_DTW:-1}"
REAL_MAX_ALIGNMENT_WINDOWS="${REAL_MAX_ALIGNMENT_WINDOWS:-63}"

# RTL line stitching can create infeasible positives. Keep scan/appearance/ink
# augmentation enabled, but disable stitching by default.
REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0}"

# Confirm that the checkpoint belongs to the active branch model before GPUs are requested.
python - "${PRETRAINED_WEIGHTS}" <<'PY'
import sys
import torch
import model_backend

path = sys.argv[1]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
actual = str(config.get("model_backend", config.get("visual_encoder_type", ""))).lower()
expected = str(model_backend.MODEL_NAME).lower()
print(f"branch_backend={expected} checkpoint_backend={actual or '<missing>'}")
if actual and actual != expected:
    raise SystemExit(
        f"Checkpoint/backend mismatch: branch={expected!r}, checkpoint={actual!r}."
    )
PY

export PROJECT_DIR
export DATASET_TYPE=real
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
export REAL_AUGMENT="${AUGMENT}"
export REAL_AUG_STITCH_PROB
export REAL_FILTER_INFEASIBLE_SPAN_DTW
export REAL_MAX_ALIGNMENT_WINDOWS
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-568}"
export REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"

export NUM_GPUS
export BATCH_SIZE
export GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
export PARTITION="${PARTITION:-rtx4090}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export MEMORY="${MEMORY:-96G}"
export TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"

export JOB_ID
export PRETRAINED_WEIGHTS
export EPOCHS="${EPOCHS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-finetuning}"

export MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS}"
export SPAN_MAX_CORE_CHARS_CAP="${REAL_MAX_TEXT_SPAN_CHARS}"
export MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-2}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"
export SPAN_INCLUDE_SPACE_CONTEXT=0
export SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
export ALLOW_UNSAFE_SPAN_CONFIG=0

export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
export USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"

printf '%s\n' \
  "Submitting real fine-tuning through the canonical launcher" \
  "  branch              = $(git branch --show-current)" \
  "  commit              = $(git rev-parse HEAD)" \
  "  checkpoint          = ${PRETRAINED_WEIGHTS}" \
  "  job id              = ${JOB_ID}" \
  "  GPUs                = ${GPU_RESOURCE}:${NUM_GPUS}" \
  "  per-GPU batch       = ${BATCH_SIZE}" \
  "  global batch        = ${GLOBAL_BATCH_SIZE}" \
  "  epochs              = ${EPOCHS}" \
  "  real augmentation   = ${REAL_AUGMENT}" \
  "  stitch probability  = ${REAL_AUG_STITCH_PROB}" \
  "  feasibility filter  = ${REAL_FILTER_INFEASIBLE_SPAN_DTW}" \
  "  alignment windows   = ${REAL_MAX_ALIGNMENT_WINDOWS}" \
  "  train samples/epoch = ${REAL_TRAIN_SAMPLES_PER_EPOCH}" \
  "  max text span chars = ${MAX_TEXT_SPAN_CHARS}"

exec bash "${PROJECT_DIR}/scripts/train/run_model_full_quality.sh"

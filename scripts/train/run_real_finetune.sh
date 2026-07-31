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

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

NUM_GPUS="${NUM_GPUS:-2}"
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_GPUS must be a positive integer, got ${NUM_GPUS}." >&2
  exit 2
fi

# Dense overlap approximately doubles the visual sequence length. Keep an
# effective batch of 64 while lowering the per-GPU micro-batch to protect memory.
case "${MODEL_BACKEND}" in
  cnn_bilstm) DEFAULT_ACCUMULATION_STEPS=2 ;;
  vit) DEFAULT_ACCUMULATION_STEPS=4 ;;
  *)
    echo "ERROR: unsupported model backend '${MODEL_BACKEND}'." >&2
    exit 2
    ;;
esac
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUMULATION_STEPS}}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-64}}"

if ! [[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GRADIENT_ACCUMULATION_STEPS must be positive." >&2
  exit 2
fi
if ! [[ "${EFFECTIVE_GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE must be positive." >&2
  exit 2
fi
BATCH_DENOMINATOR=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
if (( EFFECTIVE_GLOBAL_BATCH_SIZE % BATCH_DENOMINATOR != 0 )); then
  echo "ERROR: effective batch ${EFFECTIVE_GLOBAL_BATCH_SIZE} must be divisible by" >&2
  echo "       NUM_GPUS * GRADIENT_ACCUMULATION_STEPS = ${BATCH_DENOMINATOR}." >&2
  exit 2
fi
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / BATCH_DENOMINATOR))
MICRO_GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Keep the pretrained 32-pixel receptive field. Dense stride-8 overlap improves
# localization and increases the sequence from 63 to 125 windows without changing
# any checkpoint parameter shape.
LINE_HEIGHT="${LINE_HEIGHT:-128}"
LINE_WIDTH="${LINE_WIDTH:-1024}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"

read -r STRIDE_PIXELS COMPUTED_ALIGNMENT_WINDOWS < <(
  python - "${LINE_WIDTH}" "${WINDOW_SIZE}" "${STRIDE_RATIO}" "${WINDOW_OVERLAP_MODE}" <<'PY'
import sys

line_width = int(sys.argv[1])
window_size = int(sys.argv[2])
stride_ratio = float(sys.argv[3])
mode = sys.argv[4].strip().lower()

if line_width <= 0 or window_size <= 0 or window_size > line_width:
    raise SystemExit(
        f"Invalid line/window geometry: line_width={line_width}, window_size={window_size}."
    )
if mode == "no_overlap":
    stride = window_size
elif mode == "light_overlap":
    stride = max(1, window_size // 2)
elif mode == "dense_overlap":
    stride = max(1, window_size // 4)
elif mode == "custom":
    stride = max(1, int(window_size * stride_ratio))
else:
    raise SystemExit(f"Unknown WINDOW_OVERLAP_MODE={mode!r}.")

windows = ((line_width - window_size) // stride) + 1
print(stride, windows)
PY
)

# The optimized Arabic span encoder is designed for truthful visible cores of at
# most two characters. Longer cores can make one small image region represent an
# entire phrase and are deliberately rejected by training_optimizations.py.
REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-2}"
if ! [[ "${REAL_MAX_TEXT_SPAN_CHARS}" =~ ^[0-9]+$ ]] \
  || (( REAL_MAX_TEXT_SPAN_CHARS < 1 || REAL_MAX_TEXT_SPAN_CHARS > 2 )); then
  echo "ERROR: REAL_MAX_TEXT_SPAN_CHARS must be 1 or 2 for truthful alignment." >&2
  echo "Do not enlarge spans to repair long transcripts." >&2
  exit 2
fi

# Derive the feasibility cap from the exact geometry used by the visual model.
# A smaller explicit cap is allowed as a conservative filter; a larger one would
# keep samples that the actual image sequence cannot align and is rejected.
REAL_FILTER_INFEASIBLE_SPAN_DTW="${REAL_FILTER_INFEASIBLE_SPAN_DTW:-1}"
REAL_MAX_ALIGNMENT_WINDOWS="${REAL_MAX_ALIGNMENT_WINDOWS:-${COMPUTED_ALIGNMENT_WINDOWS}}"
if ! [[ "${REAL_MAX_ALIGNMENT_WINDOWS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: REAL_MAX_ALIGNMENT_WINDOWS must be positive." >&2
  exit 2
fi
if (( REAL_MAX_ALIGNMENT_WINDOWS > COMPUTED_ALIGNMENT_WINDOWS )); then
  echo "ERROR: REAL_MAX_ALIGNMENT_WINDOWS=${REAL_MAX_ALIGNMENT_WINDOWS} exceeds" >&2
  echo "       the actual ${COMPUTED_ALIGNMENT_WINDOWS} windows from this geometry." >&2
  exit 2
fi

# Keep appearance and ink augmentation enabled but disable line stitching because
# concatenated transcripts can violate the fixed visual-sequence contract.
REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0}"
REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-7000}"

if ! [[ "${REAL_TRAIN_SAMPLES_PER_EPOCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: REAL_TRAIN_SAMPLES_PER_EPOCH must be positive." >&2
  exit 2
fi

# Confirm backend and fixed patch width before requesting GPUs. Changing stride is
# checkpoint-compatible; changing window width is not safe for the ViT projection
# and changes the CNN receptive-field contract.
python - "${PRETRAINED_WEIGHTS}" "${WINDOW_SIZE}" "${STRIDE_PIXELS}" <<'PY'
import sys
import torch
import model_backend

path = sys.argv[1]
requested_window = int(sys.argv[2])
requested_stride = int(sys.argv[3])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
actual = str(config.get("model_backend", config.get("visual_encoder_type", ""))).lower()
expected = str(model_backend.MODEL_NAME).lower()
checkpoint_window = config.get("window_size")
checkpoint_stride = config.get("stride")
print(
    f"branch_backend={expected} checkpoint_backend={actual or '<missing>'} "
    f"checkpoint_window={checkpoint_window or '<missing>'} "
    f"checkpoint_stride={checkpoint_stride or '<missing>'} "
    f"requested_window={requested_window} requested_stride={requested_stride}"
)
if actual and actual != expected:
    raise SystemExit(
        f"Checkpoint/backend mismatch: branch={expected!r}, checkpoint={actual!r}."
    )
if checkpoint_window is not None and int(checkpoint_window) != requested_window:
    raise SystemExit(
        "Checkpoint/window mismatch: keep WINDOW_SIZE equal to the pretrained "
        f"value ({checkpoint_window}), and change stride for denser coverage."
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
export REAL_TRAIN_SAMPLES_PER_EPOCH
export REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"

export NUM_GPUS
export BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export EFFECTIVE_GLOBAL_BATCH_SIZE
export GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
export PARTITION="${PARTITION:-rtx4090}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export MEMORY="${MEMORY:-96G}"
export TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"

export JOB_ID
export PRETRAINED_WEIGHTS
export EPOCHS="${EPOCHS:-15}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
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

export LINE_HEIGHT
export LINE_WIDTH
export WINDOW_SIZE
export STRIDE_RATIO
export WINDOW_OVERLAP_MODE
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
export USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"

printf '%s\n' \
  "Submitting real fine-tuning through the canonical launcher" \
  "  branch                = $(git branch --show-current)" \
  "  commit                = $(git rev-parse HEAD)" \
  "  model backend          = ${MODEL_BACKEND}" \
  "  checkpoint            = ${PRETRAINED_WEIGHTS}" \
  "  job id                = ${JOB_ID}" \
  "  GPUs                  = ${GPU_RESOURCE}:${NUM_GPUS}" \
  "  per-GPU micro-batch   = ${BATCH_SIZE}" \
  "  micro global batch    = ${MICRO_GLOBAL_BATCH_SIZE}" \
  "  accumulation steps    = ${GRADIENT_ACCUMULATION_STEPS}" \
  "  effective global batch= ${EFFECTIVE_GLOBAL_BATCH_SIZE}" \
  "  epochs                = ${EPOCHS}" \
  "  validation frequency  = ${VALID_EVERY_N_EPOCHS}" \
  "  real augmentation     = ${REAL_AUGMENT}" \
  "  stitch probability    = ${REAL_AUG_STITCH_PROB}" \
  "  train samples/epoch   = ${REAL_TRAIN_SAMPLES_PER_EPOCH}" \
  "  line geometry         = ${LINE_WIDTH}x${LINE_HEIGHT}" \
  "  window/stride         = ${WINDOW_SIZE}/${STRIDE_PIXELS}" \
  "  actual visual windows = ${COMPUTED_ALIGNMENT_WINDOWS}" \
  "  feasibility cap       = ${REAL_MAX_ALIGNMENT_WINDOWS}" \
  "  max text span chars   = ${MAX_TEXT_SPAN_CHARS}"

exec bash "${PROJECT_DIR}/scripts/train/run_model_full_quality.sh"

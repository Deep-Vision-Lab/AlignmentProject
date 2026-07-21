#!/usr/bin/env bash
# Generic full-quality training launcher for synthetic or real Arabic data.
# This is the only shell training entry point.
#
# Examples:
#   DATASET_TYPE=synthetic NUM_SAMPLES=8000 NUM_GPUS=2 \
#     bash scripts/train/run_span_d3tw_full_quality.sh
#
#   DATASET_TYPE=real REAL_AUGMENT=1 REAL_TRAIN_SAMPLES_PER_EPOCH=10000 \
#     NUM_GPUS=2 bash scripts/train/run_span_d3tw_full_quality.sh

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_span_d3tw_full_quality.sh" >&2
  echo "Override configuration through environment variables." >&2
  exit 2
fi

# sbatch copies this script into Slurm's protected spool directory. During the
# submitted run, always trust the PROJECT_DIR exported by the submitting process
# instead of deriving the repository from BASH_SOURCE[0].
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

[[ -d "${PROJECT_DIR}" ]] || {
  echo "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 1
}
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/out" "${PROJECT_DIR}/logs"

# Export every resolved setting to the Slurm job and torchrun ranks.
set -a

# ---------------------------------------------------------------------------
# Slurm and runtime
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-07-00:00:00}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-align_full}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_GPUS must be a positive integer, got ${NUM_GPUS}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
DATASET_TYPE="${DATASET_TYPE:-synthetic}"
case "${DATASET_TYPE}" in
  synthetic)
    NUM_SAMPLES="${NUM_SAMPLES:-8000}"
    REAL_AUGMENT="${REAL_AUGMENT:-0}"
    REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
    if [[ -z "${DATA_DIR:-}" ]]; then
      if [[ -d "${PROJECT_DIR}/DataSet/Synthetic_Arabic_${NUM_SAMPLES}" ]]; then
        DATA_DIR="${PROJECT_DIR}/DataSet/Synthetic_Arabic_${NUM_SAMPLES}"
      else
        DATA_DIR="${PROJECT_DIR}/DataSet/Synthetic_Arabic"
      fi
    fi
    ;;
  real)
    NUM_SAMPLES="${NUM_SAMPLES:-10000}"
    REAL_AUGMENT="${REAL_AUGMENT:-1}"
    REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-10000}"
    DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
    ;;
  *)
    echo "ERROR: DATASET_TYPE must be synthetic or real, got ${DATASET_TYPE}" >&2
    exit 2
    ;;
esac

LANGUAGE="${LANGUAGE:-Arabic}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"

# Real manifest and preprocessing. Ignored for synthetic data.
REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest.jsonl}"
REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-0}"
REAL_BINARIZE="${REAL_BINARIZE:-1}"
REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"

# Real-data training augmentation. Ignored for synthetic data.
REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0.25}"
REAL_AUG_STITCH_POOL_SIZE="${REAL_AUG_STITCH_POOL_SIZE:-32}"
REAL_AUG_STITCH_MAX_TEXT_CHARS="${REAL_AUG_STITCH_MAX_TEXT_CHARS:-120}"
REAL_AUG_STITCH_PREFER_ADJACENT="${REAL_AUG_STITCH_PREFER_ADJACENT:-1}"
REAL_AUG_STITCH_GAP_MIN="${REAL_AUG_STITCH_GAP_MIN:-0.08}"
REAL_AUG_STITCH_GAP_MAX="${REAL_AUG_STITCH_GAP_MAX:-0.18}"
REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.85}"
REAL_AUG_ROTATE_DEG="${REAL_AUG_ROTATE_DEG:-1.25}"
REAL_AUG_TRANSLATE_X="${REAL_AUG_TRANSLATE_X:-0.012}"
REAL_AUG_TRANSLATE_Y="${REAL_AUG_TRANSLATE_Y:-0.035}"
REAL_AUG_BRIGHTNESS="${REAL_AUG_BRIGHTNESS:-0.12}"
REAL_AUG_CONTRAST="${REAL_AUG_CONTRAST:-0.18}"
REAL_AUG_BLUR_PROB="${REAL_AUG_BLUR_PROB:-0.15}"
REAL_AUG_BLUR_RADIUS="${REAL_AUG_BLUR_RADIUS:-0.8}"
REAL_AUG_NOISE_PROB="${REAL_AUG_NOISE_PROB:-0.18}"
REAL_AUG_NOISE_STD="${REAL_AUG_NOISE_STD:-5.0}"
REAL_AUG_MORPH_PROB="${REAL_AUG_MORPH_PROB:-0.25}"
REAL_AUG_SPECKLE_PROB="${REAL_AUG_SPECKLE_PROB:-0.12}"
REAL_AUG_SPECKLE_FRACTION="${REAL_AUG_SPECKLE_FRACTION:-0.0006}"

# ---------------------------------------------------------------------------
# Model and full-quality objectives
# ---------------------------------------------------------------------------
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-arabic_span}"
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
USE_BILSTM="${USE_BILSTM:-1}"
BILSTM_LAYERS="${BILSTM_LAYERS:-2}"
USE_LOCAL_WINDOW_GROUPING="${USE_LOCAL_WINDOW_GROUPING:-1}"
LOCAL_GROUP_SIZE="${LOCAL_GROUP_SIZE:-3}"
MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-3}"
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
SPAN_BOUNDARY_CONTEXT_CHARS="${SPAN_BOUNDARY_CONTEXT_CHARS:-1}"
SPAN_INCLUDE_SPACE_CONTEXT="${SPAN_INCLUDE_SPACE_CONTEXT:-1}"
SPAN_SPACE_TOKEN="${SPAN_SPACE_TOKEN:-<SPACE>}"
STRIP_SPAN_TEXT_EDGES="${STRIP_SPAN_TEXT_EDGES:-1}"

SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
SPAN_DTW_BUCKET_TEXT_LENGTHS="${SPAN_DTW_BUCKET_TEXT_LENGTHS:-1}"
SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-16}"
SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-256}"
SPAN_DTW_MEM_DEBUG="${SPAN_DTW_MEM_DEBUG:-0}"
SPAN_FEATURE_CACHE_SIZE="${SPAN_FEATURE_CACHE_SIZE:-2048}"
SPAN_FEATURE_CACHE_DTYPE="${SPAN_FEATURE_CACHE_DTYPE:-float16}"
CLEAR_SPAN_CACHE_EACH_EPOCH="${CLEAR_SPAN_CACHE_EACH_EPOCH:-1}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
SPAN_NEGATIVE_GRAD_MODE="${SPAN_NEGATIVE_GRAD_MODE:-hardest}"
NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
CONTRASTIVE_SOFT_DTW_GAMMA="${CONTRASTIVE_SOFT_DTW_GAMMA:-0.1}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-10.0}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.07}"
IMAGE_TEXT_LOSS_ON_BOTH_LINES="${IMAGE_TEXT_LOSS_ON_BOTH_LINES:-1}"

INK_CONTRAST_THRESHOLD="${INK_CONTRAST_THRESHOLD:-0.15}"
USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.25}"
LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.35}"
LOCAL_HARD_NEGATIVE_TOP_K="${LOCAL_HARD_NEGATIVE_TOP_K:-12}"
LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS="${LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS:-3}"
LOCAL_HARD_NEGATIVE_MIN_INK="${LOCAL_HARD_NEGATIVE_MIN_INK:-0.01}"
LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES="${LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES:-2}"
LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH="${LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH:-8}"

USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"
IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.40}"
IMAGE_PAIR_MARGIN="${IMAGE_PAIR_MARGIN:-0.40}"
IMAGE_PAIR_TOP_K="${IMAGE_PAIR_TOP_K:-8}"
IMAGE_PAIR_EVERY_N_BATCHES="${IMAGE_PAIR_EVERY_N_BATCHES:-1}"
IMAGE_PAIR_MAX_SAMPLES_PER_BATCH="${IMAGE_PAIR_MAX_SAMPLES_PER_BATCH:-8}"
PAIR_COMPOSITION_MAX_REGIONS="${PAIR_COMPOSITION_MAX_REGIONS:-2}"
PAIR_COMPOSITION_MAX_CHARS="${PAIR_COMPOSITION_MAX_CHARS:-3}"
SEQUENCE_CONSISTENCY_LOSS_WEIGHT="${SEQUENCE_CONSISTENCY_LOSS_WEIGHT:-0.05}"
ORDER_TEMPERATURE="${ORDER_TEMPERATURE:-0.07}"
ORDER_MONOTONIC_MARGIN="${ORDER_MONOTONIC_MARGIN:-0.02}"
ORDER_POSITION_COMPONENT_WEIGHT="${ORDER_POSITION_COMPONENT_WEIGHT:-1.0}"
ORDER_MONOTONIC_COMPONENT_WEIGHT="${ORDER_MONOTONIC_COMPONENT_WEIGHT:-1.0}"
IMAGE_VARIANCE_LOSS_WEIGHT="${IMAGE_VARIANCE_LOSS_WEIGHT:-0.01}"
IMAGE_VARIANCE_TARGET_STD="${IMAGE_VARIANCE_TARGET_STD:-0.05}"

# ---------------------------------------------------------------------------
# Training runtime and output
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-35}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-2}"
VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
LOG_MEMORY_EVERY_N_BATCHES="${LOG_MEMORY_EVERY_N_BATCHES:-25}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
ALLOW_JAX_DATALOADER_WORKERS="${ALLOW_JAX_DATALOADER_WORKERS:-0}"
DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-4}"
USE_AMP="${USE_AMP:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-alignment-project}"
JOB_ID="${JOB_ID:-${DATASET_TYPE}_arabic_fullquality_${NUM_SAMPLES}_gpu${NUM_GPUS}}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
RESUME="${RESUME:-}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Some BGU RTX 4090 pairs cannot use direct CUDA peer access.
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

set +a

# ---------------------------------------------------------------------------
# Resolve a local Hugging Face cache containing the frozen Arabic model.
# ---------------------------------------------------------------------------
resolve_hf_home() {
  local model_slug="models--${ARABIC_TEXT_MODEL_NAME//\//--}"
  local candidate layout snapshots snapshot
  local candidates=(
    "${HF_HOME:-}"
    "${PROJECT_DIR}/.hf_cache"
    "${PROJECT_DIR}_clone/.hf_cache"
    "${HOME}/.cache/huggingface"
    "${TRANSFORMERS_CACHE:-}"
  )

  if [[ "${TEXT_ENCODER_TYPE}" != "arabic_span" && "${TEXT_ENCODER_TYPE}" != "arabic_token" ]]; then
    printf '%s\n' "${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
    return 0
  fi
  if [[ -d "${ARABIC_TEXT_MODEL_NAME}" && -f "${ARABIC_TEXT_MODEL_NAME}/config.json" ]]; then
    printf '%s\n' "${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
    return 0
  fi

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -d "${candidate}" ]] || continue
    for layout in "${candidate}" "${candidate}/hub"; do
      snapshots="${layout}/${model_slug}/snapshots"
      [[ -d "${snapshots}" ]] || continue
      while IFS= read -r -d '' snapshot; do
        [[ -f "${snapshot}/config.json" ]] || continue
        if compgen -G "${snapshot}/model*.safetensors" >/dev/null \
          || compgen -G "${snapshot}/pytorch_model*.bin" >/dev/null; then
          printf '%s\n' "${candidate}"
          return 0
        fi
      done < <(find "${snapshots}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    done
  done

  echo "ERROR: ${ARABIC_TEXT_MODEL_NAME} was not found in a local Hugging Face cache." >&2
  echo "Checked ${PROJECT_DIR}/.hf_cache, ${PROJECT_DIR}_clone/.hf_cache, and ~/.cache/huggingface." >&2
  return 1
}

export HF_HOME="$(resolve_hf_home)"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

# ---------------------------------------------------------------------------
# Validate the dataset before requesting GPUs.
# ---------------------------------------------------------------------------
if [[ "${DATASET_TYPE}" == "synthetic" ]]; then
  [[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
    echo "ERROR: synthetic dataset requires images/ and texts/ under ${DATA_DIR}." >&2
    exit 1
  }
  DETECTED_SAMPLES="$(find "${DATA_DIR}/images" -maxdepth 1 -type f -name 'img1_*.png' | wc -l | tr -d ' ')"
  if (( DETECTED_SAMPLES < NUM_SAMPLES )); then
    echo "ERROR: requested ${NUM_SAMPLES} synthetic samples, but found ${DETECTED_SAMPLES}." >&2
    exit 1
  fi
else
  if [[ "${DATA_DIR}" == *.jsonl ]]; then
    MANIFEST_PATH="${DATA_DIR}"
  else
    MANIFEST_PATH="${DATA_DIR}/${REAL_MANIFEST_NAME}"
  fi
  [[ -f "${MANIFEST_PATH}" ]] || {
    echo "ERROR: real dataset manifest not found: ${MANIFEST_PATH}" >&2
    exit 1
  }
fi

# ---------------------------------------------------------------------------
# Submit this same script. --chdir and an absolute --output path prevent the
# copied Slurm script from running relative to Slurm's spool directory.
# ---------------------------------------------------------------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting generic full-quality training"
  echo "  project              = ${PROJECT_DIR}"
  echo "  dataset              = ${DATASET_TYPE}"
  echo "  data directory       = ${DATA_DIR}"
  echo "  samples              = ${NUM_SAMPLES}"
  echo "  epochs               = ${EPOCHS}"
  echo "  time limit           = ${TIME_LIMIT}"
  echo "  real augmentation    = ${REAL_AUGMENT}"
  echo "  train samples/epoch  = ${REAL_TRAIN_SAMPLES_PER_EPOCH}"
  echo "  GPUs                 = ${GPU_RESOURCE}:${NUM_GPUS}"
  echo "  per-GPU batch        = ${BATCH_SIZE}"
  echo "  global batch         = $((BATCH_SIZE * NUM_GPUS))"
  echo "  job id               = ${JOB_ID}"
  echo "  Hugging Face cache   = ${HF_HOME}"

  sbatch \
    --job-name="${SLURM_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --tasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Run inside the allocated node.
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/out" "${PROJECT_DIR}/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -c "import torch, transformers, jax; print(f'torch={torch.__version__} transformers={transformers.__version__} jax={jax.__version__}')" || {
  echo "ERROR: PyTorch, Transformers, or JAX is unavailable in ${CONDA_ENV}." >&2
  exit 1
}

if [[ "${TEXT_ENCODER_TYPE}" == "arabic_span" || "${TEXT_ENCODER_TYPE}" == "arabic_token" ]]; then
  python -c "from transformers import AutoModel, AutoTokenizer; import os; m=os.environ['ARABIC_TEXT_MODEL_NAME']; c=os.environ['HF_HOME']; AutoTokenizer.from_pretrained(m, cache_dir=c, local_files_only=True); AutoModel.from_pretrained(m, cache_dir=c, local_files_only=True)" >/dev/null 2>&1 || {
    echo "ERROR: ${ARABIC_TEXT_MODEL_NAME} is not readable from ${HF_HOME}." >&2
    exit 1
  }
fi

if [[ -n "${PRETRAINED_WEIGHTS}" && -n "${RESUME}" ]]; then
  echo "ERROR: set PRETRAINED_WEIGHTS or RESUME, not both." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
  [[ -f "${PRETRAINED_WEIGHTS}" ]] || {
    echo "ERROR: PRETRAINED_WEIGHTS not found: ${PRETRAINED_WEIGHTS}" >&2
    exit 1
  }
  EXTRA_ARGS+=(--pretrained_weights "${PRETRAINED_WEIGHTS}")
elif [[ -n "${RESUME}" ]]; then
  [[ -f "${RESUME}" ]] || {
    echo "ERROR: RESUME not found: ${RESUME}" >&2
    exit 1
  }
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

AUGMENT_ARG=--no-augment
if [[ "${REAL_AUGMENT}" == "1" || "${REAL_AUGMENT}" == "true" ]]; then
  AUGMENT_ARG=--augment
fi

TRAIN_ARGS=(
  train.py
  --job_id "${JOB_ID}"
  --dataset_type "${DATASET_TYPE}"
  --data_dir "${DATA_DIR}"
  "${AUGMENT_ARG}"
  --train_samples_per_epoch "${REAL_TRAIN_SAMPLES_PER_EPOCH}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --negative_mode "${NEGATIVE_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
  "${EXTRA_ARGS[@]}"
)

echo "Starting generic full-quality training"
echo "  project              = ${PROJECT_DIR}"
echo "  working directory    = $(pwd)"
echo "  dataset              = ${DATASET_TYPE}"
echo "  data directory       = ${DATA_DIR}"
echo "  samples              = ${NUM_SAMPLES}"
echo "  epochs               = ${EPOCHS}"
echo "  real augmentation    = ${REAL_AUGMENT}"
echo "  train samples/epoch  = ${REAL_TRAIN_SAMPLES_PER_EPOCH}"
echo "  world size           = ${NUM_GPUS}"
echo "  per-GPU batch        = ${BATCH_SIZE}"
echo "  global batch         = $((BATCH_SIZE * NUM_GPUS))"
echo "  NCCL P2P disabled    = ${NCCL_P2P_DISABLE}"
echo "  NCCL SHM disabled    = ${NCCL_SHM_DISABLE}"
echo "  job id               = ${JOB_ID}"
nvidia-smi -L || true

if (( NUM_GPUS > 1 )); then
  exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --max_restarts=0 \
    "${TRAIN_ARGS[@]}"
else
  exec python "${TRAIN_ARGS[@]}"
fi

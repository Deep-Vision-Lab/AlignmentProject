#!/usr/bin/env bash
#
# Run directly with:
#   sbatch scripts/sbatch_span_d3tw_win32_offline.sbatch
#
# Or override settings at submit time, for example:
#   USE_WANDB=1 NUM_NEGATIVES=2 sbatch scripts/sbatch_span_d3tw_win32_offline.sbatch
#
# This job runs Hugging Face in offline mode. Make sure the Arabic model is
# cached before submitting:
#   bash scripts/setup_manucripts_align_span_env.sh
#
#SBATCH --job-name=span_jax_w32
#SBATCH --output=out/%x_%J.out
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=03-00:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject_clone}}"
cd "${PROJECT_DIR}"
mkdir -p out logs

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
if [[ -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

python -c "import torch, transformers, jax; from transformers import AutoModel; print(f'torch={torch.__version__} transformers={transformers.__version__} jax={jax.__version__}')" || {
  echo "ERROR: PyTorch/Transformers/JAX is not ready in conda env '${CONDA_ENV}'." >&2
  echo "Run once before submitting:" >&2
  echo "  bash scripts/setup_manucripts_align_span_env.sh" >&2
  exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

# Let PyTorch and JAX share the same GPU instead of letting JAX preallocate it.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Dataset / logging.
export LANGUAGE="${LANGUAGE:-Arabic}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-project}"

# Offline Hugging Face loading.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Dynamic span-DTW quality-focused settings.
export SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-arabic_span}"
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-3}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
export STRIP_SPAN_TEXT_EDGES="${STRIP_SPAN_TEXT_EDGES:-1}"

# Bounded frozen AraBERT span-feature cache. This prevents host RAM OOM when
# many unique hard negatives are generated in one epoch.
export SPAN_FEATURE_CACHE_SIZE="${SPAN_FEATURE_CACHE_SIZE:-2048}"
export SPAN_FEATURE_CACHE_DTYPE="${SPAN_FEATURE_CACHE_DTYPE:-float16}"
export CLEAR_SPAN_CACHE_EACH_EPOCH="${CLEAR_SPAN_CACHE_EACH_EPOCH:-1}"
export SPAN_DTW_BUCKET_TEXT_LENGTHS="${SPAN_DTW_BUCKET_TEXT_LENGTHS:-1}"
export SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-16}"
export SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-256}"
export LOG_MEMORY_EVERY_N_BATCHES="${LOG_MEMORY_EVERY_N_BATCHES:-25}"
export SPAN_DTW_MEM_DEBUG="${SPAN_DTW_MEM_DEBUG:-0}"

# Quality-focused contrastive setup. Override NUM_NEGATIVES=2 if the job is too slow.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
export NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
export SPAN_NEGATIVE_GRAD_MODE="${SPAN_NEGATIVE_GRAD_MODE:-hardest}"
export USE_BILSTM="${USE_BILSTM:-1}"

# Full validation every 5 epochs. Set VALID_MAX_BATCHES=10 for faster monitoring.
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-5}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-0}"

# JAX is multithreaded. Forking PyTorch DataLoader workers after JAX has started
# can deadlock or duplicate large memory state, so keep workers at 0 by default.
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export ALLOW_JAX_DATALOADER_WORKERS="${ALLOW_JAX_DATALOADER_WORKERS:-0}"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "DataSet/Synthetic_${LANGUAGE}_${NUM_SAMPLES}" ]]; then
    DATA_DIR="DataSet/Synthetic_${LANGUAGE}_${NUM_SAMPLES}"
  else
    DATA_DIR="DataSet/Synthetic_${LANGUAGE}"
  fi
fi
JOB_ID="${JOB_ID:-span_jax_best_quality_win32_offline}"
EPOCHS="${EPOCHS:-150}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

python -c "from transformers import AutoModel, AutoTokenizer; import os; model=os.environ['ARABIC_TEXT_MODEL_NAME']; cache=os.environ['HF_HOME']; AutoTokenizer.from_pretrained(model, cache_dir=cache, local_files_only=True); AutoModel.from_pretrained(model, cache_dir=cache, local_files_only=True)" >/dev/null 2>&1 || {
  echo "ERROR: '${ARABIC_TEXT_MODEL_NAME}' is not cached for offline Hugging Face loading." >&2
  echo "Expected cache directory: ${HF_HOME}" >&2
  echo "Run once before submitting:" >&2
  echo "  bash scripts/setup_manucripts_align_span_env.sh" >&2
  exit 1
}

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2
  echo "Set DATA_DIR=/path/to/dataset or create DataSet/Synthetic_${LANGUAGE}_${NUM_SAMPLES} or DataSet/Synthetic_${LANGUAGE}." >&2
  exit 1
fi

echo "Starting JAX span-DTW training"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "JOB_ID=${JOB_ID}"
echo "SPAN_DTW_BACKEND=${SPAN_DTW_BACKEND}"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_NEGATIVES=${NUM_NEGATIVES} SPAN_NEGATIVE_GRAD_MODE=${SPAN_NEGATIVE_GRAD_MODE}"
echo "MAX_TEXT_SPAN_CHARS=${MAX_TEXT_SPAN_CHARS} MAX_WINDOWS_PER_SPAN=${MAX_WINDOWS_PER_SPAN}"
echo "SPAN_FEATURE_CACHE_SIZE=${SPAN_FEATURE_CACHE_SIZE} SPAN_FEATURE_CACHE_DTYPE=${SPAN_FEATURE_CACHE_DTYPE} CLEAR_SPAN_CACHE_EACH_EPOCH=${CLEAR_SPAN_CACHE_EACH_EPOCH}"
echo "SPAN_DTW_BUCKET_TEXT_LENGTHS=${SPAN_DTW_BUCKET_TEXT_LENGTHS} SPAN_DTW_TEXT_BUCKET_SIZE=${SPAN_DTW_TEXT_BUCKET_SIZE} SPAN_DTW_MAX_TEXT_BUCKET=${SPAN_DTW_MAX_TEXT_BUCKET}"
echo "LOG_MEMORY_EVERY_N_BATCHES=${LOG_MEMORY_EVERY_N_BATCHES}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS} ALLOW_JAX_DATALOADER_WORKERS=${ALLOW_JAX_DATALOADER_WORKERS}"
echo "VALID_EVERY_N_EPOCHS=${VALID_EVERY_N_EPOCHS} VALID_MAX_BATCHES=${VALID_MAX_BATCHES}"

python train.py \
  --job_id "${JOB_ID}" \
  --data_dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --negative_mode "${NEGATIVE_MODE}" \
  --num_negatives "${NUM_NEGATIVES}"

#!/usr/bin/env bash
#
# Full-quality Arabic manuscript alignment training.
# Submit through:
#   bash scripts/train/run_span_d3tw_full_quality.sh
#
# This profile intentionally prioritizes alignment quality over speed:
#   - full Span-DTW image-text supervision on BOTH paired lines
#   - all generated transcript negatives scored each batch
#   - local hard-negative mining every batch on the full batch
#   - image-image compositional pair loss on the full batch
#   - differentiable contextual position/order consistency
#   - 1..3 character text spans and up to 4 windows per span
#
#SBATCH --job-name=align_full_q
#SBATCH --output=out/%x_%J.out
#SBATCH --partition=rtx6000
#SBATCH --gpus=rtx_6000:1
#SBATCH --time=03-00:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=jelsana

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject_clone}}"
cd "${PROJECT_DIR}"
mkdir -p out logs

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

python -c "import torch, transformers, jax; print(f'torch={torch.__version__} transformers={transformers.__version__} jax={jax.__version__}')" || {
  echo "ERROR: PyTorch, Transformers, or JAX is unavailable in ${CONDA_ENV}." >&2
  exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export USE_AMP="${USE_AMP:-1}"

# Dataset and logging.
export LANGUAGE="${LANGUAGE:-Arabic}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-project}"

# Offline Arabic text encoder.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-arabic_span}"
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"

# Image windows: preserve the trained/evaluated 32-pixel, 50%-overlap geometry.
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export USE_BILSTM="${USE_BILSTM:-1}"
export BILSTM_LAYERS="${BILSTM_LAYERS:-2}"

# Restore the stronger grouping capacity from the previous high-quality run.
# The encoder still includes 1- and 2-character candidates; 3 is only the maximum.
export MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-3}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
export STRIP_SPAN_TEXT_EDGES="${STRIP_SPAN_TEXT_EDGES:-1}"

# JAX Span-DTW with shape bucketing for compilation reuse.
export SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
export SPAN_DTW_BUCKET_TEXT_LENGTHS="${SPAN_DTW_BUCKET_TEXT_LENGTHS:-1}"
export SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-16}"
export SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-256}"
export SPAN_DTW_MEM_DEBUG="${SPAN_DTW_MEM_DEBUG:-0}"

# Bounded frozen AraBERT feature cache.
export SPAN_FEATURE_CACHE_SIZE="${SPAN_FEATURE_CACHE_SIZE:-2048}"
export SPAN_FEATURE_CACHE_DTYPE="${SPAN_FEATURE_CACHE_DTYPE:-float16}"
export CLEAR_SPAN_CACHE_EACH_EPOCH="${CLEAR_SPAN_CACHE_EACH_EPOCH:-1}"

# Main image-text contrastive Span-DTW.
# Every negative is scored, while only the hardest negative retains a gradient graph.
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export SPAN_NEGATIVE_GRAD_MODE="${SPAN_NEGATIVE_GRAD_MODE:-hardest}"
export NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
export CONTRASTIVE_SOFT_DTW_GAMMA="${CONTRASTIVE_SOFT_DTW_GAMMA:-0.1}"
export CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-10.0}"
export CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.07}"

# Full supervision on both images in each pair.
export IMAGE_TEXT_LOSS_ON_BOTH_LINES="${IMAGE_TEXT_LOSS_ON_BOTH_LINES:-1}"

# Local CNN discrimination. Run every batch on every sample, but use a balanced
# weight so it does not dominate multi-character grouping.
export USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.25}"
export LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.35}"
export LOCAL_HARD_NEGATIVE_TOP_K="${LOCAL_HARD_NEGATIVE_TOP_K:-12}"
export LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS="${LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS:-3}"
export LOCAL_HARD_NEGATIVE_MIN_INK="${LOCAL_HARD_NEGATIVE_MIN_INK:-0.05}"
export LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES="${LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES:-1}"
export LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH="${LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH:-16}"

# Full compositional image-image supervision.
# PAIR_COMPOSITION_MAX_REGIONS=2 enables AB <-> A+B in either direction.
export USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.40}"
export IMAGE_PAIR_MARGIN="${IMAGE_PAIR_MARGIN:-0.40}"
export IMAGE_PAIR_TOP_K="${IMAGE_PAIR_TOP_K:-8}"
export IMAGE_PAIR_EVERY_N_BATCHES="${IMAGE_PAIR_EVERY_N_BATCHES:-1}"
export IMAGE_PAIR_MAX_SAMPLES_PER_BATCH="${IMAGE_PAIR_MAX_SAMPLES_PER_BATCH:-16}"
export PAIR_COMPOSITION_MAX_REGIONS="${PAIR_COMPOSITION_MAX_REGIONS:-2}"
export PAIR_COMPOSITION_MAX_CHARS="${PAIR_COMPOSITION_MAX_CHARS:-3}"

# Differentiable contextual order/relative-position supervision.
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT="${SEQUENCE_CONSISTENCY_LOSS_WEIGHT:-0.05}"
export ORDER_TEMPERATURE="${ORDER_TEMPERATURE:-0.07}"
export ORDER_MONOTONIC_MARGIN="${ORDER_MONOTONIC_MARGIN:-0.02}"
export ORDER_POSITION_COMPONENT_WEIGHT="${ORDER_POSITION_COMPONENT_WEIGHT:-1.0}"
export ORDER_MONOTONIC_COMPONENT_WEIGHT="${ORDER_MONOTONIC_COMPONENT_WEIGHT:-1.0}"

# Anti-collapse regularization.
export IMAGE_VARIANCE_LOSS_WEIGHT="${IMAGE_VARIANCE_LOSS_WEIGHT:-0.01}"
export IMAGE_VARIANCE_TARGET_STD="${IMAGE_VARIANCE_TARGET_STD:-0.05}"

# Full-quality validation and stable data loading with JAX.
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-5}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-0}"
export LOG_MEMORY_EVERY_N_BATCHES="${LOG_MEMORY_EVERY_N_BATCHES:-25}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export ALLOW_JAX_DATALOADER_WORKERS="${ALLOW_JAX_DATALOADER_WORKERS:-0}"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "DataSet/Synthetic_${LANGUAGE}_${NUM_SAMPLES}" ]]; then
    DATA_DIR="DataSet/Synthetic_${LANGUAGE}_${NUM_SAMPLES}"
  else
    DATA_DIR="DataSet/Synthetic_${LANGUAGE}"
  fi
fi

JOB_ID="${JOB_ID:-improve_model_fullquality_span3_comp}"
EPOCHS="${EPOCHS:-180}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_full_quality.py}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"

python -c "from transformers import AutoModel, AutoTokenizer; import os; m=os.environ['ARABIC_TEXT_MODEL_NAME']; c=os.environ['HF_HOME']; AutoTokenizer.from_pretrained(m, cache_dir=c, local_files_only=True); AutoModel.from_pretrained(m, cache_dir=c, local_files_only=True)" >/dev/null 2>&1 || {
  echo "ERROR: ${ARABIC_TEXT_MODEL_NAME} is not cached under ${HF_HOME}." >&2
  echo "Run: bash scripts/train/setup_manucripts_align_span_env.sh" >&2
  exit 1
}

[[ -d "${DATA_DIR}" ]] || { echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2; exit 1; }
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: TRAIN_SCRIPT does not exist: ${TRAIN_SCRIPT}" >&2; exit 1; }

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
  [[ -f "${PRETRAINED_WEIGHTS}" ]] || { echo "ERROR: PRETRAINED_WEIGHTS not found: ${PRETRAINED_WEIGHTS}" >&2; exit 1; }
  EXTRA_ARGS+=(--pretrained_weights "${PRETRAINED_WEIGHTS}")
fi

echo "Starting full-quality compositional alignment training"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "JOB_ID=${JOB_ID} TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} LEARNING_RATE=${LEARNING_RATE}"
echo "MAX_TEXT_SPAN_CHARS=${MAX_TEXT_SPAN_CHARS} MAX_WINDOWS_PER_SPAN=${MAX_WINDOWS_PER_SPAN}"
echo "NUM_NEGATIVES=${NUM_NEGATIVES} ACTIVE_NEGATIVES=${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE}"
echo "IMAGE_TEXT_LOSS_ON_BOTH_LINES=${IMAGE_TEXT_LOSS_ON_BOTH_LINES}"
echo "LOCAL_WEIGHT=${LOCAL_HARD_NEGATIVE_WEIGHT} LOCAL_SAMPLES=${LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH}"
echo "PAIR_WEIGHT=${IMAGE_PAIR_LOSS_WEIGHT} PAIR_SAMPLES=${IMAGE_PAIR_MAX_SAMPLES_PER_BATCH} COMPOSITION_REGIONS=${PAIR_COMPOSITION_MAX_REGIONS}"
echo "ORDER_WEIGHT=${SEQUENCE_CONSISTENCY_LOSS_WEIGHT} ORDER_TEMPERATURE=${ORDER_TEMPERATURE}"
echo "PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-none}"

python "${TRAIN_SCRIPT}" \
  --job_id "${JOB_ID}" \
  --data_dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --negative_mode "${NEGATIVE_MODE}" \
  --num_negatives "${NUM_NEGATIVES}" \
  "${EXTRA_ARGS[@]}"

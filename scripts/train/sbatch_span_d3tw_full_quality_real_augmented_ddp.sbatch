#!/usr/bin/env bash
# Full-quality augmented real-data training with one torchrun process per GPU.
# Submit through:
#   bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
#
# BATCH_SIZE is per GPU. The default 8 x 2 GPUs preserves the previous global
# batch size of 16 while splitting Span-DTW samples across both devices.
#SBATCH --job-name=align_ddp2
#SBATCH --output=out/%x_%J.out
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --time=03-00:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
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

export NUM_GPUS="${NUM_GPUS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export USE_AMP="${USE_AMP:-1}"

# Dataset and W&B.
export LANGUAGE="${LANGUAGE:-Arabic}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-project}"
export TRAIN_SEED="${TRAIN_SEED:-42}"

# Real manifest dataset.
export DATASET_TYPE="${DATASET_TYPE:-real}"
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest.jsonl}"
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
export DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
export REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-0}"

# Real-image preprocessing.
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"

# Training-only augmentation.
export REAL_AUGMENT="${REAL_AUGMENT:-1}"
export REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0.25}"
export REAL_AUG_STITCH_POOL_SIZE="${REAL_AUG_STITCH_POOL_SIZE:-32}"
export REAL_AUG_STITCH_MAX_TEXT_CHARS="${REAL_AUG_STITCH_MAX_TEXT_CHARS:-120}"
export REAL_AUG_STITCH_PREFER_ADJACENT="${REAL_AUG_STITCH_PREFER_ADJACENT:-1}"
export REAL_AUG_STITCH_GAP_MIN="${REAL_AUG_STITCH_GAP_MIN:-0.08}"
export REAL_AUG_STITCH_GAP_MAX="${REAL_AUG_STITCH_GAP_MAX:-0.18}"
export REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.85}"
export REAL_AUG_ROTATE_DEG="${REAL_AUG_ROTATE_DEG:-1.25}"
export REAL_AUG_TRANSLATE_X="${REAL_AUG_TRANSLATE_X:-0.012}"
export REAL_AUG_TRANSLATE_Y="${REAL_AUG_TRANSLATE_Y:-0.035}"
export REAL_AUG_BRIGHTNESS="${REAL_AUG_BRIGHTNESS:-0.12}"
export REAL_AUG_CONTRAST="${REAL_AUG_CONTRAST:-0.18}"
export REAL_AUG_BLUR_PROB="${REAL_AUG_BLUR_PROB:-0.15}"
export REAL_AUG_BLUR_RADIUS="${REAL_AUG_BLUR_RADIUS:-0.8}"
export REAL_AUG_NOISE_PROB="${REAL_AUG_NOISE_PROB:-0.18}"
export REAL_AUG_NOISE_STD="${REAL_AUG_NOISE_STD:-5.0}"
export REAL_AUG_MORPH_PROB="${REAL_AUG_MORPH_PROB:-0.25}"
export REAL_AUG_SPECKLE_PROB="${REAL_AUG_SPECKLE_PROB:-0.12}"
export REAL_AUG_SPECKLE_FRACTION="${REAL_AUG_SPECKLE_FRACTION:-0.0006}"

# Frozen Arabic text backbone and image window geometry.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-arabic_span}"
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export USE_BILSTM="${USE_BILSTM:-1}"
export BILSTM_LAYERS="${BILSTM_LAYERS:-2}"

# Full-quality span capacity.
export MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-3}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
export STRIP_SPAN_TEXT_EDGES="${STRIP_SPAN_TEXT_EDGES:-1}"

# JAX Span-DTW and frozen-feature cache.
export SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
export SPAN_DTW_BUCKET_TEXT_LENGTHS="${SPAN_DTW_BUCKET_TEXT_LENGTHS:-1}"
export SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-16}"
export SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-256}"
export SPAN_DTW_MEM_DEBUG="${SPAN_DTW_MEM_DEBUG:-0}"
export SPAN_FEATURE_CACHE_SIZE="${SPAN_FEATURE_CACHE_SIZE:-2048}"
export SPAN_FEATURE_CACHE_DTYPE="${SPAN_FEATURE_CACHE_DTYPE:-float16}"
export CLEAR_SPAN_CACHE_EACH_EPOCH="${CLEAR_SPAN_CACHE_EACH_EPOCH:-1}"

# Main loss. BATCH_SIZE is per GPU: 8 x 2 = global batch 16.
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export SPAN_NEGATIVE_GRAD_MODE="${SPAN_NEGATIVE_GRAD_MODE:-hardest}"
export NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
export CONTRASTIVE_SOFT_DTW_GAMMA="${CONTRASTIVE_SOFT_DTW_GAMMA:-0.1}"
export CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-10.0}"
export CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.07}"
export IMAGE_TEXT_LOSS_ON_BOTH_LINES="${IMAGE_TEXT_LOSS_ON_BOTH_LINES:-1}"

# Every phase runs on every local sample, on every rank.
export USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.25}"
export LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.35}"
export LOCAL_HARD_NEGATIVE_TOP_K="${LOCAL_HARD_NEGATIVE_TOP_K:-12}"
export LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS="${LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS:-3}"
export LOCAL_HARD_NEGATIVE_MIN_INK="${LOCAL_HARD_NEGATIVE_MIN_INK:-0.05}"
export LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES="${LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES:-1}"
export LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH="${LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH:-8}"
export USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.40}"
export IMAGE_PAIR_MARGIN="${IMAGE_PAIR_MARGIN:-0.40}"
export IMAGE_PAIR_TOP_K="${IMAGE_PAIR_TOP_K:-8}"
export IMAGE_PAIR_EVERY_N_BATCHES="${IMAGE_PAIR_EVERY_N_BATCHES:-1}"
export IMAGE_PAIR_MAX_SAMPLES_PER_BATCH="${IMAGE_PAIR_MAX_SAMPLES_PER_BATCH:-8}"
export PAIR_COMPOSITION_MAX_REGIONS="${PAIR_COMPOSITION_MAX_REGIONS:-2}"
export PAIR_COMPOSITION_MAX_CHARS="${PAIR_COMPOSITION_MAX_CHARS:-3}"
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT="${SEQUENCE_CONSISTENCY_LOSS_WEIGHT:-0.05}"
export ORDER_TEMPERATURE="${ORDER_TEMPERATURE:-0.07}"
export ORDER_MONOTONIC_MARGIN="${ORDER_MONOTONIC_MARGIN:-0.02}"
export ORDER_POSITION_COMPONENT_WEIGHT="${ORDER_POSITION_COMPONENT_WEIGHT:-1.0}"
export ORDER_MONOTONIC_COMPONENT_WEIGHT="${ORDER_MONOTONIC_COMPONENT_WEIGHT:-1.0}"
export IMAGE_VARIANCE_LOSS_WEIGHT="${IMAGE_VARIANCE_LOSS_WEIGHT:-0.01}"
export IMAGE_VARIANCE_TARGET_STD="${IMAGE_VARIANCE_TARGET_STD:-0.05}"

# Validation and loader stability with one JAX runtime per rank.
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-5}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-0}"
export LOG_MEMORY_EVERY_N_BATCHES="${LOG_MEMORY_EVERY_N_BATCHES:-25}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export ALLOW_JAX_DATALOADER_WORKERS="${ALLOW_JAX_DATALOADER_WORKERS:-0}"

JOB_ID="${JOB_ID:-real_arabic_fullquality_span3_augmented_ddp2}"
EPOCHS="${EPOCHS:-180}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_full_quality_real_augmented_ddp.py}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
RESUME="${RESUME:-}"

[[ -d "${DATA_DIR}" ]] || { echo "ERROR: DATA_DIR does not exist: ${DATA_DIR}" >&2; exit 1; }
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: TRAIN_SCRIPT does not exist: ${TRAIN_SCRIPT}" >&2; exit 1; }
if [[ -n "${PRETRAINED_WEIGHTS}" && -n "${RESUME}" ]]; then
  echo "ERROR: set PRETRAINED_WEIGHTS or RESUME, not both." >&2
  exit 1
fi

python -c "from transformers import AutoModel, AutoTokenizer; import os; m=os.environ['ARABIC_TEXT_MODEL_NAME']; c=os.environ['HF_HOME']; AutoTokenizer.from_pretrained(m, cache_dir=c, local_files_only=True); AutoModel.from_pretrained(m, cache_dir=c, local_files_only=True)" >/dev/null 2>&1 || {
  echo "ERROR: ${ARABIC_TEXT_MODEL_NAME} is not cached under ${HF_HOME}." >&2
  exit 1
}

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_WEIGHTS}" ]]; then
  [[ -f "${PRETRAINED_WEIGHTS}" ]] || { echo "ERROR: PRETRAINED_WEIGHTS not found: ${PRETRAINED_WEIGHTS}" >&2; exit 1; }
  EXTRA_ARGS+=(--pretrained_weights "${PRETRAINED_WEIGHTS}")
elif [[ -n "${RESUME}" ]]; then
  [[ -f "${RESUME}" ]] || { echo "ERROR: RESUME not found: ${RESUME}" >&2; exit 1; }
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

echo "Starting ${NUM_GPUS}-GPU DDP full-quality training"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR=${DATA_DIR} JOB_ID=${JOB_ID}"
echo "per_gpu_batch=${BATCH_SIZE} global_batch=$((BATCH_SIZE * NUM_GPUS))"
echo "all phases enabled: both_lines=${IMAGE_TEXT_LOSS_ON_BOTH_LINES} active_negatives=${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE}"
nvidia-smi -L || true

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --max_restarts=0 \
  "${TRAIN_SCRIPT}" \
  --job_id "${JOB_ID}" \
  --data_dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --negative_mode "${NEGATIVE_MODE}" \
  --num_negatives "${NUM_NEGATIVES}" \
  "${EXTRA_ARGS[@]}"

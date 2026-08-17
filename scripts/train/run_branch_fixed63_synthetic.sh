#!/usr/bin/env bash
# Branch-aware fixed-63 synthetic training. Always enters training_runtime/entrypoint.py
# so model_backend.py, rather than a legacy direct train.py import, selects CNN/ViT/DINO.
set -euo pipefail
set -a

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/AugmentedArabicDataset63}"
JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-fixed63}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
EPOCHS="${EPOCHS:-35}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

case "${MODEL_BACKEND}" in
  cnn_bilstm) DEFAULT_ACCUM=1 ;;
  vit) DEFAULT_ACCUM=4 ;;
  dinov3_convnext) DEFAULT_ACCUM=4 ;;
  *) echo "ERROR: unsupported model backend ${MODEL_BACKEND}" >&2; exit 2 ;;
esac
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUM}}"
DENOM=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE must be divisible by GPUs*accum=${DENOM}" >&2
  exit 2
}
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / DENOM))

[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: fixed63 dataset missing images/ or texts/: ${DATA_DIR}" >&2
  exit 2
}

export DATA_DIR JOB_ID NUM_SAMPLES EPOCHS LEARNING_RATE NUM_GPUS BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export DATASET_TYPE=synthetic
export SYNTHETIC_MANUSCRIPT_AUGMENT=0
export REAL_AUGMENT=0
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
export SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
export SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${MAX_TEXT_SPAN_CHARS}}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-2}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting fixed63 branch-aware training: backend=${MODEL_BACKEND} job=${JOB_ID} batch=${BATCH_SIZE}x${NUM_GPUS} accum=${GRADIENT_ACCUMULATION_STEPS}"
  sbatch \
    --partition="${PARTITION}" \
    --job-name="${JOB_ID}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL \
    "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING=1

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
export TRAIN_EXPECTED_BRANCH="${BRANCH}"
export TRAIN_EXPECTED_COMMIT="${COMMIT}"
export TRAIN_EXPECTED_BACKEND="${MODEL_BACKEND}"

echo "=== FIXED63 SYNTHETIC TRAINING ==="
echo "branch=${BRANCH} commit=${COMMIT}"
echo "backend=${MODEL_BACKEND}"
echo "job=${JOB_ID} epochs=${EPOCHS} lr=${LEARNING_RATE}"
echo "batch_per_gpu=${BATCH_SIZE} gpus=${NUM_GPUS} accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "DINOV3_REPO_DIR=${DINOV3_REPO_DIR:-<unset>} DINOV3_WEIGHTS=${DINOV3_WEIGHTS:-<unset>}"
echo "DINOV3_FREEZE_BACKBONE=${DINOV3_FREEZE_BACKBONE:-1} USE_BILSTM=${USE_BILSTM:-1}"

ARGS=(
  training_runtime/entrypoint.py
  --job_id "${JOB_ID}"
  --dataset_type synthetic
  --data_dir "${DATA_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --window_size "${WINDOW_SIZE}"
  --stride_ratio "${STRIDE_RATIO}"
  --window_overlap_mode "${WINDOW_OVERLAP_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
  --no-augment
)

if (( NUM_GPUS > 1 )); then
  exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 "${ARGS[@]}"
fi
exec python "${ARGS[@]}"

#!/usr/bin/env bash
# Architecture-neutral canonical real fine-tuning stage used by the research pipeline.
set -euo pipefail
set -a

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

: "${PRETRAINED_WEIGHTS:?Set PRETRAINED_WEIGHTS to the synthetic checkpoint.}"
[[ -f "${PRETRAINED_WEIGHTS}" ]] || { echo "ERROR: missing checkpoint ${PRETRAINED_WEIGHTS}" >&2; exit 2; }

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-real-stage}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing real manifest under ${DATA_DIR}" >&2; exit 2; }

NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
case "${MODEL_BACKEND}" in
  cnn_bilstm) DEFAULT_ACCUM=1 ;;
  vit) DEFAULT_ACCUM=4 ; export USE_BILSTM=0 USE_LOCAL_WINDOW_GROUPING=0 ;;
  dinov3_convnext)
    DEFAULT_ACCUM=4
    : "${DINOV3_REPO_DIR:?DINOv3 branch requires DINOV3_REPO_DIR.}"
    export DINOV3_ALLOW_RANDOM_INIT="${DINOV3_ALLOW_RANDOM_INIT:-1}"
    ;;
  *) echo "ERROR: unsupported backend ${MODEL_BACKEND}" >&2; exit 2 ;;
esac
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUM}}"
DENOM=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE} not divisible by ${DENOM}" >&2
  exit 2
}
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / DENOM))

export JOB_ID DATA_DIR PRETRAINED_WEIGHTS NUM_GPUS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export DATASET_TYPE=real
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_SPLIT_BY_PAIR_ID=1
export REAL_AUGMENT=0 AUGMENT=0 SYNTHETIC_MANUSCRIPT_AUGMENT=0
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
export EPOCHS="${EPOCHS:-5}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-arabic_span}"
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
export REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-${MAX_TEXT_SPAN_CHARS}}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export SPAN_NEGATIVE_GRAD_MODE="${SPAN_NEGATIVE_GRAD_MODE:-hardest}"
export USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
export USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"
export IMAGE_TEXT_LOSS_ON_BOTH_LINES=1
export REAL_USE_EXTRA_NO_SHARED=0
export REAL_UNIQUE_LINE_ADAPTATION=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING=1

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting canonical real stage: backend=${MODEL_BACKEND} job=${JOB_ID} epochs=${EPOCHS} lr=${LEARNING_RATE}"
  DEP_ARGS=()
  [[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS+=(--dependency="${DEPENDENCY}")
  sbatch --partition="${PARTITION}" --job-name="${JOB_ID}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" --time="${TIME_LIMIT}" --mail-type=ALL --mail-user="${MAIL_USER}" \
    "${DEP_ARGS[@]}" --export=ALL "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
export TRAIN_EXPECTED_BRANCH="${BRANCH}" TRAIN_EXPECTED_COMMIT="${COMMIT}" TRAIN_EXPECTED_BACKEND="${MODEL_BACKEND}"

echo "=== S4 CANONICAL REAL FINE-TUNING ==="
echo "branch=${BRANCH} commit=${COMMIT} backend=${MODEL_BACKEND}"
echo "checkpoint=${PRETRAINED_WEIGHTS} job=${JOB_ID}"
echo "epochs=${EPOCHS} lr=${LEARNING_RATE} augment=0 negatives=${NUM_NEGATIVES}"

ARGS=(
  training_runtime/entrypoint.py
  --job_id "${JOB_ID}"
  --dataset_type real
  --data_dir "${DATA_DIR}"
  --no-augment
  --train_samples_per_epoch "${REAL_TRAIN_SAMPLES_PER_EPOCH}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --window_size "${WINDOW_SIZE}"
  --stride_ratio "${STRIDE_RATIO}"
  --window_overlap_mode "${WINDOW_OVERLAP_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
  --pretrained_weights "${PRETRAINED_WEIGHTS}"
)

if (( NUM_GPUS > 1 )); then
  exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 "${ARGS[@]}"
fi
exec python "${ARGS[@]}"

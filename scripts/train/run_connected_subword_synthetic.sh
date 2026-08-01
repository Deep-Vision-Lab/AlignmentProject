#!/usr/bin/env bash
# Stage 1: connected-subword training from random initialization on synthetic Arabic data.
# Run from the repository root on the login node. This script submits its own Slurm job.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: JOB_ID=<name> bash scripts/train/run_connected_subword_synthetic.sh" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p out logs

: "${JOB_ID:?Set JOB_ID to the synthetic output weights-folder name.}"
if [[ -n "${PRETRAINED_WEIGHTS:-}" || -n "${SYNTHETIC_WEIGHTS:-}" ]]; then
  echo "ERROR: Stage 1 must start from random initialization; do not set PRETRAINED_WEIGHTS or SYNTHETIC_WEIGHTS." >&2
  exit 2
fi

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-128}"
case "${MODEL_BACKEND}" in
  cnn_bilstm) DEFAULT_ACCUMULATION_STEPS=2 ;;
  vit) DEFAULT_ACCUMULATION_STEPS=4 ;;
  *) echo "ERROR: unsupported model backend '${MODEL_BACKEND}'." >&2; exit 2 ;;
esac
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUMULATION_STEPS}}"
for name in NUM_GPUS EFFECTIVE_GLOBAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
  value="${!name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: ${name} must be a positive integer." >&2; exit 2; }
done
DENOMINATOR=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOMINATOR == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * GRADIENT_ACCUMULATION_STEPS." >&2
  exit 2
}
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / DENOMINATOR))

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"

DATASET_TYPE=synthetic
LANGUAGE=Arabic
TEXT_ENCODER_TYPE=arabic_span
SPAN_TOKENIZATION_MODE=connected_subword
SPAN_USE_BLANK_TRANSITIONS=1
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-16}"
SPAN_CONNECTED_WINDOWS_PER_CHAR="${SPAN_CONNECTED_WINDOWS_PER_CHAR:-3}"
SPAN_CONNECTED_EXTRA_WINDOWS="${SPAN_CONNECTED_EXTRA_WINDOWS:-1}"
SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS="${SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS:-2}"
SPAN_SPACE_MAX_WINDOWS="${SPAN_SPACE_MAX_WINDOWS:-3}"
SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-32}"
SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-128}"
SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
ALLOW_UNSAFE_SPAN_CONFIG=1
PAIR_COMPOSITION_MAX_REGIONS=1
PAIR_COMPOSITION_MAX_CHARS=24
USE_IMAGE_PAIR_CONTRASTIVE=0
IMAGE_TEXT_LOSS_ON_BOTH_LINES=0

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
NUM_SAMPLES="${NUM_SAMPLES:-8000}"
EPOCHS="${EPOCHS:-35}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-2}"
VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
USE_AMP="${USE_AMP:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-alignment-connected-subword-synthetic}"
ZERO_SHOT_PROFILE="${ZERO_SHOT_PROFILE:-1}"

JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_DIR}/.jax_cache/connected_subword}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-7200}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
TOKENIZERS_PARALLELISM=false
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
set +a
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

resolve_hf_home() {
  local slug="models--${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
  slug="models--${slug#models--}"
  slug="${slug//\//--}"
  local candidate layout snapshots snapshot
  local candidates=("${HF_HOME:-}" "${PROJECT_DIR}/.hf_cache" "${PROJECT_DIR}_clone/.hf_cache" "${HOME}/.cache/huggingface")
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -d "${candidate}" ]] || continue
    for layout in "${candidate}" "${candidate}/hub"; do
      snapshots="${layout}/${slug}/snapshots"
      [[ -d "${snapshots}" ]] || continue
      while IFS= read -r -d '' snapshot; do
        [[ -f "${snapshot}/config.json" ]] || continue
        if compgen -G "${snapshot}/model*.safetensors" >/dev/null || compgen -G "${snapshot}/pytorch_model*.bin" >/dev/null; then
          printf '%s\n' "${candidate}"
          return 0
        fi
      done < <(find "${snapshots}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    done
  done
  return 1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf '%s\n' \
    "Connected-subword synthetic scratch training" \
    "  branch                 = $(git branch --show-current)" \
    "  backend                = ${MODEL_BACKEND}" \
    "  job id                 = ${JOB_ID}" \
    "  data                   = ${DATA_DIR}" \
    "  epochs                 = ${EPOCHS}" \
    "  synthetic samples      = ${NUM_SAMPLES}" \
    "  effective global batch = ${EFFECTIVE_GLOBAL_BATCH_SIZE}"
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
HF_HOME="$(resolve_hf_home)" || {
  echo "ERROR: local AraBERT cache was not found." >&2
  exit 1
}
export HF_HOME
unset TRANSFORMERS_CACHE

TRAIN_ARGS=(
  training_runtime/entrypoint.py
  --job_id "${JOB_ID}"
  --dataset_type synthetic
  --data_dir "${DATA_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --negative_mode "${NEGATIVE_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
)

RANK_WRAPPER="${PROJECT_DIR}/training_runtime/run_rank_isolated.sh"
[[ -f "${RANK_WRAPPER}" ]] || { echo "ERROR: missing ${RANK_WRAPPER}" >&2; exit 1; }
if (( NUM_GPUS > 1 )); then
  exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 --no_python \
    bash "${RANK_WRAPPER}" "${TRAIN_ARGS[@]}"
fi
exec bash "${RANK_WRAPPER}" "${TRAIN_ARGS[@]}"

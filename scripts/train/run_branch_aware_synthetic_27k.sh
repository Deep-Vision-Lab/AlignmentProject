#!/usr/bin/env bash
# Branch-aware synthetic trainer. Unlike the legacy synthetic fallback, this
# always enters through training_runtime/entrypoint.py so model_backend.py is
# installed before the visual model is constructed.
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset63}"
DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
EXPECTED_MODEL_BACKEND="${EXPECTED_MODEL_BACKEND:-${MODEL_BACKEND}}"
if [[ "${MODEL_BACKEND}" != "${EXPECTED_MODEL_BACKEND}" ]]; then
  echo "ERROR: branch backend changed before submission: expected=${EXPECTED_MODEL_BACKEND} active=${MODEL_BACKEND}." >&2
  exit 2
fi

case "${MODEL_BACKEND}" in
  vit) DEFAULT_JOB_ID="vit_augmented_fixed63_27k_v2" ;;
  cnn_bilstm) DEFAULT_JOB_ID="cnn_bilstm_augmented_fixed63_27k_v2" ;;
  *) echo "ERROR: unsupported model backend ${MODEL_BACKEND}." >&2; exit 2 ;;
esac

JOB_ID="${JOB_ID:-${DEFAULT_JOB_ID}}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
EPOCHS="${EPOCHS:-35}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"

# The optimized truthful-visible-core runtime only supports 1-2 character
# text cores. Keep the synthetic pretraining semantics identical to the real
# fine-tuning path and reject unsafe overrides before requesting GPUs.
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-2}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"
SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${MAX_TEXT_SPAN_CHARS}}"
SPAN_INCLUDE_SPACE_CONTEXT=0
SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
ALLOW_UNSAFE_SPAN_CONFIG=0

[[ "${MAX_TEXT_SPAN_CHARS}" =~ ^[12]$ ]] || {
  echo "ERROR: MAX_TEXT_SPAN_CHARS must be 1 or 2 for the optimized truthful-visible-core runtime; got ${MAX_TEXT_SPAN_CHARS}." >&2
  exit 2
}
[[ "${MAX_TEXT_TOKEN_CHARS}" =~ ^[12]$ ]] || {
  echo "ERROR: MAX_TEXT_TOKEN_CHARS must be 1 or 2; got ${MAX_TEXT_TOKEN_CHARS}." >&2
  exit 2
}
[[ "${MAX_WINDOWS_PER_SPAN}" =~ ^[1-3]$ ]] || {
  echo "ERROR: MAX_WINDOWS_PER_SPAN must be between 1 and 3; got ${MAX_WINDOWS_PER_SPAN}." >&2
  exit 2
}

DATASET_TYPE=synthetic
SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0
AUGMENT=0

NUM_GPUS="${NUM_GPUS:-2}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_DIR}/.jax_cache/span_dtw}"
PYTHONUNBUFFERED=1
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

ENTRYPOINT="${PROJECT_DIR}/training_runtime/entrypoint.py"
RANK_WRAPPER="${PROJECT_DIR}/training_runtime/run_rank_isolated.sh"
[[ -f "${ENTRYPOINT}" ]] || { echo "ERROR: missing ${ENTRYPOINT}" >&2; exit 2; }
[[ -f "${RANK_WRAPPER}" ]] || { echo "ERROR: missing ${RANK_WRAPPER}" >&2; exit 2; }

resolve_env_prefix() {
  local candidate
  for candidate in \
    "${CONDA_PREFIX:-}" \
    "${HOME}/.conda/envs/${CONDA_ENV}" \
    "${HOME}/miniconda3/envs/${CONDA_ENV}" \
    "${HOME}/anaconda3/envs/${CONDA_ENV}"; do
    [[ -n "${candidate}" ]] || continue
    [[ -x "${candidate}/bin/python" && -x "${candidate}/bin/torchrun" ]] || continue
    if "${candidate}/bin/python" - <<'PY' >/dev/null 2>&1
import torch, transformers, jax
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_hf_home() {
  local model="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
  local slug="models--${model//\//--}"
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
HF_HOME="$(resolve_hf_home)" || {
  echo "ERROR: local Hugging Face cache for ${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02} was not found." >&2
  exit 2
}

export PROJECT_DIR DATA_DIR MODEL_BACKEND EXPECTED_MODEL_BACKEND JOB_ID
export NUM_SAMPLES EPOCHS WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE
export MAX_TEXT_SPAN_CHARS MAX_TEXT_TOKEN_CHARS MAX_WINDOWS_PER_SPAN
export SPAN_MAX_CORE_CHARS_CAP SPAN_CONNECTED_MAX_UNITS_PER_SPAN
export SPAN_INCLUDE_SPACE_CONTEXT SPAN_ALLOW_CHARACTER_SPACE_SURFACES ALLOW_UNSAFE_SPAN_CONFIG
export DATASET_TYPE SYNTHETIC_MANUSCRIPT_AUGMENT REAL_AUGMENT AUGMENT
export NUM_GPUS CONDA_ENV GPU_RESOURCE PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT
export SLURM_JOB_NAME MAIL_USER NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG TOKENIZERS_PARALLELISM
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE XLA_PYTHON_CLIENT_PREALLOCATE
export JAX_COMPILATION_CACHE_DIR PYTHONUNBUFFERED OMP_NUM_THREADS MKL_NUM_THREADS HF_HOME
set +a
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

print_config() {
  printf '%s\n' \
    "Branch-aware Fixed63 synthetic training" \
    "  branch=$(git branch --show-current)" \
    "  model backend=${MODEL_BACKEND}" \
    "  dataset=${DATA_DIR}" \
    "  samples=${NUM_SAMPLES}" \
    "  epochs=${EPOCHS}" \
    "  window=${WINDOW_SIZE}" \
    "  stride ratio=${STRIDE_RATIO}" \
    "  max text span chars=${MAX_TEXT_SPAN_CHARS}" \
    "  max text token chars=${MAX_TEXT_TOKEN_CHARS}" \
    "  max windows/span=${MAX_WINDOWS_PER_SPAN}" \
    "  online augmentation=disabled" \
    "  GPUs=${GPU_RESOURCE}:${NUM_GPUS}" \
    "  time limit=${TIME_LIMIT}" \
    "  job id=${JOB_ID}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  print_config
  sbatch \
    --job-name="${SLURM_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",EXPECTED_MODEL_BACKEND="${EXPECTED_MODEL_BACKEND}" \
    "${SCRIPT_PATH}"
  exit 0
fi

ENV_PREFIX="$(resolve_env_prefix)" || {
  echo "ERROR: could not find a usable conda environment '${CONDA_ENV}' with torch, transformers, and jax." >&2
  echo "Checked: ${CONDA_PREFIX:-<unset>}, ${HOME}/.conda/envs/${CONDA_ENV}, ${HOME}/miniconda3/envs/${CONDA_ENV}, ${HOME}/anaconda3/envs/${CONDA_ENV}" >&2
  exit 2
}
TRAIN_PYTHON="${ENV_PREFIX}/bin/python"
TORCHRUN_BIN="${ENV_PREFIX}/bin/torchrun"
export ENV_PREFIX TRAIN_PYTHON TORCHRUN_BIN
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

ACTIVE_MODEL_BACKEND="$(${TRAIN_PYTHON} - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
if [[ "${ACTIVE_MODEL_BACKEND}" != "${EXPECTED_MODEL_BACKEND}" ]]; then
  echo "ERROR: repository branch/backend changed while this Slurm job was queued: expected=${EXPECTED_MODEL_BACKEND} active=${ACTIVE_MODEL_BACKEND}." >&2
  echo "Do not switch the shared working directory to another model branch until this job has started/finished." >&2
  exit 2
fi
MODEL_BACKEND="${ACTIVE_MODEL_BACKEND}"

TRAIN_ARGS=(
  training_runtime/entrypoint.py
  --job_id "${JOB_ID}"
  --dataset_type synthetic
  --data_dir "${DATA_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --window_size "${WINDOW_SIZE}"
  --stride_ratio "${STRIDE_RATIO}"
  --window_overlap_mode "${WINDOW_OVERLAP_MODE}"
  --no-augment
)

print_config
printf '%s\n' "  env prefix=${ENV_PREFIX}" "  python=${TRAIN_PYTHON}" "  torchrun=${TORCHRUN_BIN}"
"${TRAIN_PYTHON}" -c "import torch, transformers, jax, model_backend; print(f'torch={torch.__version__} transformers={transformers.__version__} jax={jax.__version__} backend={model_backend.MODEL_NAME}')"
nvidia-smi -L || true

if (( NUM_GPUS > 1 )); then
  exec "${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --max_restarts=0 \
    --no_python \
    bash "${RANK_WRAPPER}" "${TRAIN_ARGS[@]}"
fi
exec bash "${RANK_WRAPPER}" "${TRAIN_ARGS[@]}"

#!/usr/bin/env bash
# Unified launcher for the pre-augmented 27,000-pair Arabic synthetic dataset.
# Run from the repository root on the login node:
#   JOB_ID=<name> bash scripts/train/run_augmented_synthetic_27k.sh
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p out logs

BRANCH_NAME="$(git branch --show-current 2>/dev/null || true)"
[[ -n "${BRANCH_NAME}" ]] || BRANCH_NAME="$(basename "${PROJECT_DIR}")"
SAFE_BRANCH_NAME="${BRANCH_NAME//\//-}"

JOB_ID="${JOB_ID:-${SAFE_BRANCH_NAME}-augmented-arabic-27k}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
EPOCHS="${EPOCHS:-35}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"

if [[ -z "${DATA_DIR:-}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/DataSet/AugmentedArabicDataset" \
    "${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset"; do
    if [[ -d "${candidate}/images" && -d "${candidate}/texts" ]]; then
      DATA_DIR="${candidate}"
      break
    fi
  done
fi

: "${DATA_DIR:?Could not locate DataSet/AugmentedArabicDataset. Set DATA_DIR explicitly.}"
DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}

PAIR_COUNT="$(find "${DATA_DIR}/images" -maxdepth 1 -type f -name 'img1_*.png' | wc -l | tr -d ' ')"
[[ "${PAIR_COUNT}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: failed to count img1_*.png files under ${DATA_DIR}/images" >&2
  exit 2
}
if (( PAIR_COUNT < NUM_SAMPLES )); then
  echo "ERROR: requested ${NUM_SAMPLES} pairs, but found only ${PAIR_COUNT}." >&2
  exit 2
fi

DATASET_TYPE=synthetic
SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ -f "${PROJECT_DIR}/scripts/train/run_connected_subword_synthetic.sh" ]]; then
  NUM_GPUS="${NUM_GPUS:-2}"
  printf '%s\n' \
    "Augmented Arabic synthetic training" \
    "  branch=${BRANCH_NAME}" \
    "  dataset=${DATA_DIR}" \
    "  pairs=${PAIR_COUNT}" \
    "  samples=${NUM_SAMPLES}" \
    "  epochs=${EPOCHS}" \
    "  online augmentation=disabled" \
    "  NCCL P2P=disabled" \
    "  GPUs=${NUM_GPUS}" \
    "  Slurm tasks=1" \
    "  job id=${JOB_ID}"
  exec bash "${PROJECT_DIR}/scripts/train/run_connected_subword_synthetic.sh"
fi

if [[ -f "${PROJECT_DIR}/train.py" ]] \
  && grep -q -- '--dataset_type' "${PROJECT_DIR}/train.py"; then
  TRAINING_MODE=generic
  NUM_GPUS="${NUM_GPUS:-2}"
elif [[ -f "${PROJECT_DIR}/run_train.sh" ]]; then
  TRAINING_MODE=legacy
  NUM_GPUS="${NUM_GPUS:-1}"
else
  echo "ERROR: no supported training entry point on ${BRANCH_NAME}." >&2
  exit 2
fi

GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
ACCOUNT="${ACCOUNT:-jelsana}"
QOS="${QOS:-normal}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"

printf '%s\n' \
  "Augmented Arabic synthetic training" \
  "  branch=${BRANCH_NAME}" \
  "  dataset=${DATA_DIR}" \
  "  pairs=${PAIR_COUNT}" \
  "  samples=${NUM_SAMPLES}" \
  "  epochs=${EPOCHS}" \
  "  online augmentation=disabled" \
  "  NCCL P2P=disabled" \
  "  Slurm request=${NUM_GPUS} ${GPU_RESOURCE} GPU(s), 1 task, ${CPUS_PER_TASK} CPUs, ${MEMORY}" \
  "  job id=${JOB_ID}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  sbatch \
    --job-name="${SLURM_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --error="${PROJECT_DIR}/out/%x_%J.err" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --qos="${QOS}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",DATA_DIR="${DATA_DIR}",JOB_ID="${JOB_ID}",NUM_SAMPLES="${NUM_SAMPLES}",EPOCHS="${EPOCHS}",NUM_GPUS="${NUM_GPUS}",TRAINING_MODE="${TRAINING_MODE}",SYNTHETIC_MANUSCRIPT_AUGMENT=0,REAL_AUGMENT=0,DATASET_TYPE=synthetic,NCCL_P2P_DISABLE=1,NCCL_SHM_DISABLE=0,NCCL_ASYNC_ERROR_HANDLING=1 \
    "${SCRIPT_PATH}"
  exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load anaconda || true
fi
command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is unavailable." >&2
  exit 2
}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG
export CUDA_DEVICE_ORDER

if [[ "${TRAINING_MODE}" == generic ]]; then
  TRAIN_ARGS=(
    train.py
    --job_id "${JOB_ID}"
    --dataset_type synthetic
    --data_dir "${DATA_DIR}"
    --num_samples "${NUM_SAMPLES}"
    --epochs "${EPOCHS}"
    --no-augment
  )
  if (( NUM_GPUS > 1 )); then
    exec torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${NUM_GPUS}" \
      --max_restarts=0 \
      "${TRAIN_ARGS[@]}"
  fi
  exec python "${TRAIN_ARGS[@]}"
fi

exec bash run_train.sh "${JOB_ID}" "${DATA_DIR}"

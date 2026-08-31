#!/usr/bin/env bash
#SBATCH --job-name=vit_planC
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${HOME}/.conda/envs/manucripts_align/bin/torchrun}"

cd "${PROJECT_DIR}"
mkdir -p out

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "ERROR: torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
  exit 2
fi
if [[ ! -d "${DATASET}/images" || ! -d "${DATASET}/texts" ]]; then
  echo "ERROR: Synthetic63 dataset not found at: ${DATASET}" >&2
  exit 2
fi

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse --short HEAD)"
export JOB_NAME="${JOB_NAME:-${SLURM_JOB_NAME:-}}"

if [[ -z "${JOB_NAME}" ]]; then
  echo "ERROR: JOB_NAME is empty and SLURM_JOB_NAME is unavailable." >&2
  exit 2
fi

echo "========================================"
echo "AlignmentProject Synthetic63 training"
echo "branch       = ${BRANCH}"
echo "commit       = ${COMMIT}"
echo "node         = $(hostname)"
echo "job_name     = ${JOB_NAME}"
echo "slurm_job_id = ${SLURM_JOB_ID:-unknown}"
echo "weights_dir  = Weights/${JOB_NAME}"
echo "dataset      = ${DATASET}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --dataset "${DATASET}"

#!/usr/bin/env bash
#SBATCH --job-name=vit_cfm_spatial
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
PYTHON_BIN="${PYTHON_BIN:-$(dirname "${TORCHRUN_BIN}")/python}"

cd "${PROJECT_DIR}"
mkdir -p out

[[ -x "${TORCHRUN_BIN}" ]] || { echo "ERROR: torchrun not executable: ${TORCHRUN_BIN}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: python not executable: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATASET}/images" && -d "${DATASET}/texts" ]] || {
  echo "ERROR: Synthetic63 dataset not found at ${DATASET}" >&2
  exit 2
}

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse --short HEAD)"
export JOB_NAME="${JOB_NAME:-${SLURM_JOB_NAME:-vit_cfm_spatial}}"
export TEXT_ANCHOR_WEIGHTS="${TEXT_ANCHOR_WEIGHTS:-}"
export CFM_STRIDE_RATIO="${CFM_STRIDE_RATIO:-0.50}"

export SPATIAL_AFFINITY_RADIUS="${SPATIAL_AFFINITY_RADIUS:-3}"
export SPATIAL_AFFINITY_TEMPERATURE="${SPATIAL_AFFINITY_TEMPERATURE:-0.15}"
export SPATIAL_AFFINITY_DISTANCE_PENALTY="${SPATIAL_AFFINITY_DISTANCE_PENALTY:-0.12}"
export SPATIAL_AFFINITY_INITIAL_GATE="${SPATIAL_AFFINITY_INITIAL_GATE:-0.15}"
export CONTEXT_RESIDUAL_INITIAL_GATE="${CONTEXT_RESIDUAL_INITIAL_GATE:-0.25}"
export SPATIAL_LETTER_WEIGHT="${SPATIAL_LETTER_WEIGHT:-0.40}"
export DENSE_LETTER_WEIGHT="${DENSE_LETTER_WEIGHT:-0.15}"
export DENSE_LETTER_TEMPERATURE="${DENSE_LETTER_TEMPERATURE:-0.07}"
export LOCAL_CONTEXT_CONSISTENCY_WEIGHT="${LOCAL_CONTEXT_CONSISTENCY_WEIGHT:-0.05}"

echo "========================================"
echo "AlignmentProject CFM-inspired spatial-language training"
echo "branch       = ${BRANCH}"
echo "commit       = ${COMMIT}"
echo "job          = ${JOB_NAME}"
echo "dataset      = ${DATASET}"
echo "text_anchor  = ${TEXT_ANCHOR_WEIGHTS:-<deterministic fresh frozen anchor>}"
echo "stride_ratio = ${CFM_STRIDE_RATIO} (0.50=stride16, 0.25=stride8)"
echo "architecture = window -> local vote -> language adapter -> gated context"
echo "pair policy  = independent encoders; NO cross-line attention"
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

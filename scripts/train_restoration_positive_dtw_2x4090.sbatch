#!/usr/bin/env bash
#SBATCH --job-name=vit_restore_dtw
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
[[ -d "${DATASET}" ]] || { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse --short HEAD)"

export JOB_NAME="${JOB_NAME:-vit_restore_dtw_s16}"
export RESTORATION_DTW_STRIDE_RATIO="${RESTORATION_DTW_STRIDE_RATIO:-0.50}"
export POSITIVE_LETTER_DTW_WEIGHT="${POSITIVE_LETTER_DTW_WEIGHT:-1.0}"
export POSITIVE_LETTER_DTW_GAMMA="${POSITIVE_LETTER_DTW_GAMMA:-0.05}"
export POSITIVE_LETTER_DTW_STEP_PENALTY="${POSITIVE_LETTER_DTW_STEP_PENALTY:-0.02}"
export POSITIVE_LETTER_DTW_MIN_INK="${POSITIVE_LETTER_DTW_MIN_INK:-0.01}"
export RESTORATION_WEIGHT="${RESTORATION_WEIGHT:-0.10}"
export RESTORATION_PIXEL_WEIGHT="${RESTORATION_PIXEL_WEIGHT:-1.0}"
export RESTORATION_EDGE_WEIGHT="${RESTORATION_EDGE_WEIGHT:-0.50}"
export RESTORATION_FOREGROUND_WEIGHT="${RESTORATION_FOREGROUND_WEIGHT:-2.0}"
export RESTORATION_CONTRAST_SCALE="${RESTORATION_CONTRAST_SCALE:-0.15}"
export RESTORATION_DECODER_CHANNELS="${RESTORATION_DECODER_CHANNELS:-64}"
export LETTER_CODEBOOK_SEED="${LETTER_CODEBOOK_SEED:-1234}"

echo "============================================================"
echo "AlignmentProject: restoration + positive letter-DTW"
echo "branch       = ${BRANCH}"
echo "commit       = ${COMMIT}"
echo "job          = ${JOB_NAME}"
echo "dataset      = ${DATASET}"
echo "stride_ratio = ${RESTORATION_DTW_STRIDE_RATIO} (0.50 => stride16)"
echo "loss         = ${POSITIVE_LETTER_DTW_WEIGHT}*positive_DTW + ${RESTORATION_WEIGHT}*restoration"
echo "restoration  = ${RESTORATION_PIXEL_WEIGHT}*pixel + ${RESTORATION_EDGE_WEIGHT}*edge"
echo "negatives    = 0"
echo "pair loss    = 0"
echo "context loss = 0"
echo "text target  = frozen orthogonal Arabic character identities"
echo "============================================================"

# Cheap syntax preflight before torchrun starts two worker processes. This catches
# accidental source-editing errors before the actual training entrypoint runs.
"${PYTHON_BIN}" -m py_compile \
  train.py \
  runtime_device_setup.py \
  model_backend.py \
  vlm_restoration_positive_dtw.py \
  DataLoader.py \
  DataSet.py \
  AugmentedRealDataLoader.py
echo "Python syntax preflight: OK"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_COMPILE_VISUAL="${TORCH_COMPILE_VISUAL:-0}"

"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --dataset "${DATASET}" \
  "$@"

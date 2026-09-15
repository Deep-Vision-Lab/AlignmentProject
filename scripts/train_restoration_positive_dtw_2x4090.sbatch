#!/usr/bin/env bash
#SBATCH --job-name=resnet18_tinyvit_dtw
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

export JOB_NAME="${JOB_NAME:-resnet18_tinyvit_dtw_s16}"
export RESNET_TINYVIT_EPOCHS="${RESNET_TINYVIT_EPOCHS:-20}"
export RESNET_TINYVIT_NUM_SAMPLES="${RESNET_TINYVIT_NUM_SAMPLES:-6000}"
export RESNET_TINYVIT_LEARNING_RATE="${RESNET_TINYVIT_LEARNING_RATE:-1e-4}"
export RESNET_TINYVIT_STRIDE_RATIO="${RESNET_TINYVIT_STRIDE_RATIO:-0.50}"
export RESNET_TINYVIT_NUM_NEGATIVES="${RESNET_TINYVIT_NUM_NEGATIVES:-10}"
export RESNET_TINYVIT_CONTRASTIVE_WEIGHT="${RESNET_TINYVIT_CONTRASTIVE_WEIGHT:-0.50}"
export RESNET_TINYVIT_CONTRASTIVE_MARGIN="${RESNET_TINYVIT_CONTRASTIVE_MARGIN:-0.20}"

# Standard ViT-Tiny sequence configuration.
export TINY_VIT_EMBED_DIM="${TINY_VIT_EMBED_DIM:-192}"
export TINY_VIT_LAYERS="${TINY_VIT_LAYERS:-12}"
export TINY_VIT_HEADS="${TINY_VIT_HEADS:-3}"
export TINY_VIT_MLP_DIM="${TINY_VIT_MLP_DIM:-768}"
export TINY_VIT_DROPOUT="${TINY_VIT_DROPOUT:-0.10}"

# Off by default so jobs do not need network access to download ImageNet weights.
export RESNET18_IMAGENET_PRETRAINED="${RESNET18_IMAGENET_PRETRAINED:-0}"

export POSITIVE_LETTER_DTW_WEIGHT="${POSITIVE_LETTER_DTW_WEIGHT:-1.0}"
export POSITIVE_LETTER_DTW_GAMMA_START="${POSITIVE_LETTER_DTW_GAMMA_START:-0.50}"
export POSITIVE_LETTER_DTW_GAMMA_END="${POSITIVE_LETTER_DTW_GAMMA_END:-0.05}"
export POSITIVE_LETTER_DTW_ANNEAL_EPOCHS="${POSITIVE_LETTER_DTW_ANNEAL_EPOCHS:-10}"
export POSITIVE_LETTER_DTW_VERTICAL_PENALTY="${POSITIVE_LETTER_DTW_VERTICAL_PENALTY:-0.05}"
export POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY="${POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY:-0.30}"
export POSITIVE_LETTER_DTW_POSITION_PRIOR="${POSITIVE_LETTER_DTW_POSITION_PRIOR:-0.15}"
export POSITIVE_LETTER_DTW_DISABLE_HORIZONTAL_WHEN_FEASIBLE="${POSITIVE_LETTER_DTW_DISABLE_HORIZONTAL_WHEN_FEASIBLE:-1}"
export POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE="${POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE:-0.10}"
export POSITIVE_LETTER_DTW_COST_MODE="${POSITIVE_LETTER_DTW_COST_MODE:-full_alphabet_nll}"
export LETTER_CODEBOOK_SEED="${LETTER_CODEBOOK_SEED:-1234}"

echo "============================================================"
echo "AlignmentProject: ResNet18 windows + ViT-Tiny context + letter-DTW"
echo "branch       = ${BRANCH}"
echo "commit       = ${COMMIT}"
echo "job          = ${JOB_NAME}"
echo "dataset      = ${DATASET}"
echo "epochs       = ${RESNET_TINYVIT_EPOCHS}"
echo "samples      = ${RESNET_TINYVIT_NUM_SAMPLES}"
echo "learning_rate= ${RESNET_TINYVIT_LEARNING_RATE}"
echo "window       = 128x32; stride_ratio=${RESNET_TINYVIT_STRIDE_RATIO} (0.50 => stride16)"
echo "local encoder= ResNet-18; imagenet_pretrained=${RESNET18_IMAGENET_PRETRAINED}"
echo "Tiny ViT     = dim=${TINY_VIT_EMBED_DIM}, layers=${TINY_VIT_LAYERS}, heads=${TINY_VIT_HEADS}, mlp=${TINY_VIT_MLP_DIM}"
echo "fusion       = concat(local, contextual) -> projection -> L2 normalization"
echo "decoder      = NONE"
echo "loss         = ${POSITIVE_LETTER_DTW_WEIGHT}*positive_letter_DTW + ${RESNET_TINYVIT_CONTRASTIVE_WEIGHT}*negative_DTW_margin"
echo "dtw gamma    = ${POSITIVE_LETTER_DTW_GAMMA_START} -> ${POSITIVE_LETTER_DTW_GAMMA_END} over ${POSITIVE_LETTER_DTW_ANNEAL_EPOCHS} epochs"
echo "dtw moves    = vertical=${POSITIVE_LETTER_DTW_VERTICAL_PENALTY} horizontal=${POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY} disable_horizontal_if_feasible=${POSITIVE_LETTER_DTW_DISABLE_HORIZONTAL_WHEN_FEASIBLE}"
echo "dtw prior    = position=${POSITIVE_LETTER_DTW_POSITION_PRIOR} cost=${POSITIVE_LETTER_DTW_COST_MODE}"
echo "negatives    = ${RESNET_TINYVIT_NUM_NEGATIVES}; margin=${RESNET_TINYVIT_CONTRASTIVE_MARGIN}"
echo "text target  = frozen orthogonal Arabic character identities"
echo "evaluation   = image-only fused vectors"
echo "============================================================"

"${PYTHON_BIN}" -m py_compile \
  train.py \
  runtime_device_setup.py \
  model_backend.py \
  vlm_resnet18_tinyvit_positive_dtw.py \
  vlm_restoration_positive_dtw.py \
  restoration_recommended_components.py \
  zero_shot_preprocessing.py \
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

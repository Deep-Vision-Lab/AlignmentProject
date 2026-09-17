#!/usr/bin/env bash
#SBATCH --job-name=res18_tinyvit_dtw
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

export JOB_NAME="${JOB_NAME:-res18_tinyvit_dtw_s16}"
export RESTORATION_EPOCHS="${RESTORATION_EPOCHS:-20}"
export RESTORATION_NUM_SAMPLES="${RESTORATION_NUM_SAMPLES:-6000}"
export RESTORATION_LEARNING_RATE="${RESTORATION_LEARNING_RATE:-1e-4}"
export RESTORATION_TRAINING_STAGE="align"
export RESTORATION_NUM_NEGATIVES="${RESTORATION_NUM_NEGATIVES:-0}"
export RESTORATION_CONTRASTIVE_WEIGHT="${RESTORATION_CONTRASTIVE_WEIGHT:-0.0}"
export RESTORATION_CONTRASTIVE_MARGIN="${RESTORATION_CONTRASTIVE_MARGIN:-0.20}"
export RESTORATION_DTW_STRIDE_RATIO="${RESTORATION_DTW_STRIDE_RATIO:-0.50}"
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
export PRETRAINED_CACHE_DIR="${PRETRAINED_CACHE_DIR:-${PROJECT_DIR}/Pretrained}"
export TORCH_HOME="${TORCH_HOME:-${PRETRAINED_CACHE_DIR}/torch}"
export HF_HOME="${HF_HOME:-${PRETRAINED_CACHE_DIR}/huggingface}"
export RESNET18_PRETRAINED=1
export TINY_VIT_PRETRAINED=1
export TINY_VIT_PRETRAINED_MODEL="${TINY_VIT_PRETRAINED_MODEL:-facebook/deit-tiny-patch16-224}"
export PRETRAINED_LOCAL_ONLY=1
export DDP_STATIC_GRAPH=1
export FORCE_DDP_STATIC_GRAPH=0
export SPAN_DTW_BACKEND=torch
export USE_LOCAL_HARD_NEGATIVES=0
export USE_IMAGE_PAIR_CONTRASTIVE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TINY_VIT_DROPOUT="${TINY_VIT_DROPOUT:-0.0}"

"${PYTHON_BIN}" -m py_compile   train.py   runtime_device_setup.py   model_backend.py   vlm_restoration_positive_dtw.py   resnet18_window_encoder.py   pretrained_tiny_vit.py   scripts/cache_pretrained_models.py   restoration_recommended_components.py   zero_shot_preprocessing.py   DataLoader.py   DataSet.py   AugmentedRealDataLoader.py   training_stability.py   training_optimizations.py   epoch_subset_sampling.py   distributed_runtime_guard.py   vit_checkpoint_migration.py

"${PYTHON_BIN}" scripts/cache_pretrained_models.py --check-only >/dev/null
"${PYTHON_BIN}" -m pytest -q tests/test_restoration_positive_dtw.py >/dev/null

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_COMPILE_VISUAL="${TORCH_COMPILE_VISUAL:-0}"
export GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_DROP_LAST="${TRAIN_DROP_LAST:-0}"
export POINT2_DIAGNOSTICS="${POINT2_DIAGNOSTICS:-1}"
export POINT2_DIAGNOSTIC_EVERY="${POINT2_DIAGNOSTIC_EVERY:-1}"

"${TORCHRUN_BIN}"   --standalone   --nproc_per_node=2   train.py   --dataset "${DATASET}"   "$@"

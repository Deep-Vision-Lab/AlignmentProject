#!/usr/bin/env bash
#
# Run directly with:
#   sbatch scripts/sbatch_span_d3tw_win32_offline.sbatch
#
# Or use the no-argument wrapper:
#   bash scripts/run_span_d3tw_offline.sh
#
# Assumes the Arabic Hugging Face model is already cached because this job
# runs with TRANSFORMERS_OFFLINE=1.
#
#SBATCH --job-name=span_d3tw_w32
#SBATCH --output=out/%x_%J.out
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=03-00:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --mail-type=All
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject_clone}}"
cd "${PROJECT_DIR}"
mkdir -p out logs
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
if [[ -n "${CONDA_ENV}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python -c "import torch, transformers; from transformers import AutoModel; print(f'torch={torch.__version__} transformers={transformers.__version__}')" || {
  echo "ERROR: PyTorch-compatible transformers is not ready in conda env '${CONDA_ENV}'." >&2
  echo "Run once before submitting:" >&2
  echo "  bash scripts/setup_manucripts_align_span_env.sh" >&2
  exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

export NUM_SAMPLES=10000
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE=online
export WANDB_PROJECT=AlignmentProject
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TEXT_ENCODER_TYPE=arabic_span
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
export WINDOW_SIZE=32
export STRIDE_RATIO=0.5
export WINDOW_OVERLAP_MODE=custom
export MAX_TEXT_TOKEN_CHARS=2
export MAX_TEXT_SPAN_CHARS=2
export MAX_WINDOWS_PER_SPAN=4
export SPAN_NEGATIVE_GRAD_MODE=hardest
export USE_BILSTM=1

python -c "from transformers import AutoModel, AutoTokenizer; import os; model=os.environ['ARABIC_TEXT_MODEL_NAME']; cache=os.environ['HF_HOME']; AutoTokenizer.from_pretrained(model, cache_dir=cache, local_files_only=True); AutoModel.from_pretrained(model, cache_dir=cache, local_files_only=True)" >/dev/null 2>&1 || {
  echo "ERROR: '${ARABIC_TEXT_MODEL_NAME}' is not cached for offline Hugging Face loading." >&2
  echo "Expected cache directory: ${HF_HOME}" >&2
  echo "Run once before submitting:" >&2
  echo "  bash scripts/setup_manucripts_align_span_env.sh" >&2
  exit 1
}

python train.py \
  --job_id span_d3tw_win32_offline \
  --data_dir DataSet/Synthetic_Arabic \
  --epochs 60 \
  --learning_rate 1e-4 \
  --negative_mode length_controlled \
  --num_negatives 10

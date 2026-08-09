#!/bin/bash
#SBATCH --job-name=real_aug_10k
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT="${PROJECT_DIR:-$HOME/BGU-Lab/AlignmentProject}"
cd "${ROOT}"
mkdir -p out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-manucripts_align}"

export CONDA_ENV_PYTHON="$(command -v python)"
export DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/DataSet/ArabicDatasetRealAug10K}"
export TARGET_TRAIN_PAIRS="${TARGET_TRAIN_PAIRS:-10000}"
export SEED="${SEED:-42}"
export HEIGHT="${HEIGHT:-128}"
export MIN_REGIONS="${MIN_REGIONS:-1}"
export MAX_REGIONS="${MAX_REGIONS:-3}"
export OVERWRITE="${OVERWRITE:-0}"

bash scripts/data/build_real_augmented_10k.sh

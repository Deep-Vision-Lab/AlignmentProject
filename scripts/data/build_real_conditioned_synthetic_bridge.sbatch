#!/bin/bash
#SBATCH --job-name=build_real_synth_bridge
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --mail-type=ALL

set -euo pipefail

ROOT="${PROJECT_DIR:-$HOME/BGU-Lab/AlignmentProject}"
cd "${ROOT}"
mkdir -p out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-manucripts_align}"

export DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/DataSet/RealSyntheticBridge_v1}"
export NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-4}"
export NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}"
export MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}"
export SEED="${SEED:-42}"
export MAX_ANCHORS="${MAX_ANCHORS:-0}"
export OVERWRITE="${OVERWRITE:-0}"

bash scripts/data/build_real_conditioned_synthetic_bridge.sh

#!/usr/bin/env bash
#SBATCH --job-name=point2_eval
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/manucripts_align/bin/python}"

cd "${PROJECT_DIR}"
mkdir -p out

[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2
  exit 2
}

export PROJECT_DIR
export PYTHON_BIN
export DEVICE="${DEVICE:-cuda}"
export N_SAMPLES="${N_SAMPLES:-100}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export START_INDEX="${START_INDEX:-1}"
export IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
export MIN_ALIGNED_WINDOWS="${MIN_ALIGNED_WINDOWS:-5}"
export SCORE_MODE="${SCORE_MODE:-raw}"

bash scripts/eval_point2_both_architectures.sh

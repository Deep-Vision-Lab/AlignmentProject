#!/usr/bin/env bash
#SBATCH --job-name=shared_regions
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%J.out
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/manucripts_align/bin/python}"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PRETRAINED_LOCAL_ONLY=1
# Preserve Slurm's GPU visibility. All scientific options are explicit CLI args.
exec "$PYTHON_BIN" -u -m Evaluation.eval_shared_regions --device cuda "$@"

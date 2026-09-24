#!/usr/bin/env bash
#SBATCH --job-name=alignment_eval
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%J.out
set -euo pipefail
exec "${PYTHON_BIN:-/home/ahmedmas/.conda/envs/manucripts_align/bin/python}" evaluate.py --device cuda "$@"

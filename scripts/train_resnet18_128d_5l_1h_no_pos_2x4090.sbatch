#!/usr/bin/env bash
#SBATCH --job-name=resnet18_128d_5l_1h_no_pos
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
set -euo pipefail
export NPROC=2
export JOB_NAME="${JOB_NAME:-resnet18_128d_5l_1h_no_pos_${SLURM_JOB_ID}}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ahmedmas/.conda/envs/manucripts_align/bin/python}"
bash "${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}/scripts/train_resnet18_128d_5l_1h_no_pos.sh"

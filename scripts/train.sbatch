#!/usr/bin/env bash
# Same resources and training defaults as the existing compact-model launcher.
#SBATCH --job-name=alignment_fusion
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
set -euo pipefail
FUSION_MODE=${FUSION_MODE:-concat}
USE_GATED_FUSION=${USE_GATED_FUSION:-0}
export NPROC=2
export JOB_NAME="${JOB_NAME:-resnet18_128d_5l_1h_no_pos_${FUSION_MODE}_gate${USE_GATED_FUSION}_${SLURM_JOB_ID:?Submit with sbatch}}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ahmedmas/.conda/envs/manucripts_align/bin/python}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}}"
exec bash "$PROJECT_DIR/scripts/train_resnet18_128d_5l_1h_no_pos.sh" \
    --fusion-mode "$FUSION_MODE" --use-gated-fusion "$USE_GATED_FUSION" "$@"

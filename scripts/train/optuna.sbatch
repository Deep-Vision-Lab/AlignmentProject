#!/usr/bin/env bash
#SBATCH --job-name=alignment_optuna
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
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate manucripts_align
exec python scripts/train/run_optuna.py "$@"

#!/bin/bash
#
# Usage:
#   sbatch scripts/sbatch_offline_d3tw_ngram.slurm
#
#   sbatch --export=ALL,CONDA_ENV=manucripts_align,JOB_ID=test_1epoch,EPOCHS=1,NUM_NEGATIVES=2,NUM_SAMPLES=10000,DATA_DIR=DataSet/Synthetic_Arabic_10000 scripts/sbatch_offline_d3tw_ngram.slurm
#
#SBATCH --job-name=AlignProj
#SBATCH --output=out/%x_%J.out
#SBATCH --error=out/%x_%J.err
#SBATCH --time=03-00:00:00
#SBATCH --gpus=rtx_6000:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=6
##SBATCH --qos=jelsana

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p out logs

if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export WANDB_MODE="${WANDB_MODE:-offline}"

bash scripts/offline_train_d3tw_ngram.sh

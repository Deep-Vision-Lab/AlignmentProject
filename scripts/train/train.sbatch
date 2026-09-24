#!/usr/bin/env bash

#SBATCH --job-name=alignment_train
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out

set -euo pipefail

# Use the SLURM job name as the experiment/run name.
RUN_NAME="${SLURM_JOB_NAME}"

echo "Job name: $SLURM_JOB_NAME"
echo "Run name: $RUN_NAME"

# Conda
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate manucripts_align

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Training
exec python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    train.py \
    --run-name "$RUN_NAME" \
    "$@"
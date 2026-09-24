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
exec "${PYTHON_BIN:-/home/ahmedmas/.conda/envs/manucripts_align/bin/python}" -m torch.distributed.run --standalone --nproc_per_node=2 train.py "$@"

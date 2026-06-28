#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --task=6

#SBATCH --mail-user=ahmedmas@post.bgu.ac.il
#SBATCH --mail-type=ALL


set -euo pipefail

: "${JOB_ID:?Set JOB_ID}"
: "${model_dir:?Set model_dir, e.g. AlignmentProject_clone}"

env=${env:-manucripts_align}
DATA_DIR=${DATA_DIR:-DataSet/Synthetic_Arabic_10000}
# Default offline character-pooling run: non-overlapping 16px windows.
# These remain overridable through sbatch --export.
WINDOW_SIZE=${WINDOW_SIZE:-16}
WINDOW_OVERLAP_MODE=${WINDOW_OVERLAP_MODE:-no_overlap}
TEXT_EMBEDDER_TYPE=${TEXT_EMBEDDER_TYPE:-orthogonal_char}
NEGATIVE_MODE=${NEGATIVE_MODE:-length_controlled}
NUM_NEGATIVES=${NUM_NEGATIVES:-10}
D3TW_WEIGHT=${D3TW_WEIGHT:-1.0}
CHAR_POOL_WEIGHT=${CHAR_POOL_WEIGHT:-0.5}
CHAR_POOL_TAU=${CHAR_POOL_TAU:-0.07}
CHAR_POOL_WARMUP_EPOCHS=${CHAR_POOL_WARMUP_EPOCHS:-5}
CHAR_POOL_RAMP_EPOCHS=${CHAR_POOL_RAMP_EPOCHS:-10}
CHAR_POOL_METHOD=${CHAR_POOL_METHOD:-hard_mean}
CHAR_POOL_DETACH_ALIGNMENT=${CHAR_POOL_DETACH_ALIGNMENT:-true}
CHAR_POOL_SKIP_SPACES=${CHAR_POOL_SKIP_SPACES:-false}
CHAR_POOL_MIN_WINDOWS_PER_CHAR=${CHAR_POOL_MIN_WINDOWS_PER_CHAR:-1}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}

mkdir -p out
cd "$HOME/BGU-Lab/$model_dir"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "$env"

python train.py \
  --job_id "$JOB_ID" \
  --data_dir "$DATA_DIR" \
  --alignment_loss_type d3tw_char_pool \
  --text_embedder_type "$TEXT_EMBEDDER_TYPE" \
  --window_size "$WINDOW_SIZE" \
  --window_overlap_mode "$WINDOW_OVERLAP_MODE" \
  --negative_mode "$NEGATIVE_MODE" \
  --num_negatives "$NUM_NEGATIVES" \
  --d3tw_weight "$D3TW_WEIGHT" \
  --char_pool_weight "$CHAR_POOL_WEIGHT" \
  --char_pool_tau "$CHAR_POOL_TAU" \
  --char_pool_warmup_epochs "$CHAR_POOL_WARMUP_EPOCHS" \
  --char_pool_ramp_epochs "$CHAR_POOL_RAMP_EPOCHS" \
  --char_pool_method "$CHAR_POOL_METHOD" \
  --char_pool_detach_alignment "$CHAR_POOL_DETACH_ALIGNMENT" \
  --char_pool_skip_spaces "$CHAR_POOL_SKIP_SPACES" \
  --char_pool_min_windows_per_char "$CHAR_POOL_MIN_WINDOWS_PER_CHAR" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR"

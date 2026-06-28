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
DATA_DIR=${DATA_DIR:-DataSet/Synthetic_Arabic_50000}
WINDOW_SIZE=${WINDOW_SIZE:-32}
WINDOW_OVERLAP_MODE=${WINDOW_OVERLAP_MODE:-no_overlap}
TEXT_EMBEDDER_TYPE=${TEXT_EMBEDDER_TYPE:-char}
NEGATIVE_MODE=${NEGATIVE_MODE:-length_controlled}
NUM_NEGATIVES=${NUM_NEGATIVES:-10}
D3TW_WEIGHT=${D3TW_WEIGHT:-1.0}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}

NGRAM_MIN_N=${NGRAM_MIN_N:-1}
NGRAM_MAX_N=${NGRAM_MAX_N:-3}
NGRAM_MIN_FREQ=${NGRAM_MIN_FREQ:-2}
NGRAM_MAX_VOCAB_SIZE=${NGRAM_MAX_VOCAB_SIZE:-5000}
NGRAM_TOKENIZER_MODE=${NGRAM_TOKENIZER_MODE:-greedy_longest}
NGRAM_SKIP_SPACES=${NGRAM_SKIP_SPACES:-true}
NGRAM_INCLUDE_LIGATURES=${NGRAM_INCLUDE_LIGATURES:-true}

TOKEN_POOL_WEIGHT=${TOKEN_POOL_WEIGHT:-0.5}
TOKEN_POOL_TAU=${TOKEN_POOL_TAU:-0.07}
TOKEN_POOL_WARMUP_EPOCHS=${TOKEN_POOL_WARMUP_EPOCHS:-5}
TOKEN_POOL_RAMP_EPOCHS=${TOKEN_POOL_RAMP_EPOCHS:-10}
TOKEN_POOL_DETACH_ALIGNMENT=${TOKEN_POOL_DETACH_ALIGNMENT:-true}
TOKEN_POOL_MIN_WINDOWS_PER_TOKEN=${TOKEN_POOL_MIN_WINDOWS_PER_TOKEN:-1}

USE_CHAR_AUX_LOSS=${USE_CHAR_AUX_LOSS:-true}
CHAR_AUX_WEIGHT=${CHAR_AUX_WEIGHT:-0.25}
CHAR_AUX_TAU=${CHAR_AUX_TAU:-0.07}
CHAR_AUX_WARMUP_EPOCHS=${CHAR_AUX_WARMUP_EPOCHS:-8}
CHAR_AUX_RAMP_EPOCHS=${CHAR_AUX_RAMP_EPOCHS:-10}

mkdir -p out
cd "$HOME/BGU-Lab/$model_dir"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "$env"

# These are cluster/offline training jobs, but W&B should sync online by
# default. Override with WANDB_MODE=offline if the compute node has no network.
export WANDB_MODE="${WANDB_MODE:-online}"

python train.py \
  --job_id "$JOB_ID" \
  --data_dir "$DATA_DIR" \
  --alignment_loss_type d3tw_char_pool \
  --text_unit_type ngram \
  --text_embedder_type "$TEXT_EMBEDDER_TYPE" \
  --window_size "$WINDOW_SIZE" \
  --window_overlap_mode "$WINDOW_OVERLAP_MODE" \
  --negative_mode "$NEGATIVE_MODE" \
  --num_negatives "$NUM_NEGATIVES" \
  --d3tw_weight "$D3TW_WEIGHT" \
  --ngram_min_n "$NGRAM_MIN_N" \
  --ngram_max_n "$NGRAM_MAX_N" \
  --ngram_min_freq "$NGRAM_MIN_FREQ" \
  --ngram_max_vocab_size "$NGRAM_MAX_VOCAB_SIZE" \
  --ngram_tokenizer_mode "$NGRAM_TOKENIZER_MODE" \
  --ngram_skip_spaces "$NGRAM_SKIP_SPACES" \
  --ngram_include_ligatures "$NGRAM_INCLUDE_LIGATURES" \
  --token_pool_weight "$TOKEN_POOL_WEIGHT" \
  --token_pool_tau "$TOKEN_POOL_TAU" \
  --token_pool_warmup_epochs "$TOKEN_POOL_WARMUP_EPOCHS" \
  --token_pool_ramp_epochs "$TOKEN_POOL_RAMP_EPOCHS" \
  --token_pool_detach_alignment "$TOKEN_POOL_DETACH_ALIGNMENT" \
  --token_pool_min_windows_per_token "$TOKEN_POOL_MIN_WINDOWS_PER_TOKEN" \
  --use_char_aux_loss "$USE_CHAR_AUX_LOSS" \
  --char_aux_weight "$CHAR_AUX_WEIGHT" \
  --char_aux_tau "$CHAR_AUX_TAU" \
  --char_aux_warmup_epochs "$CHAR_AUX_WARMUP_EPOCHS" \
  --char_aux_ramp_epochs "$CHAR_AUX_RAMP_EPOCHS" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR"

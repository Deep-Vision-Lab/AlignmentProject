#!/usr/bin/env bash
#
# Run with:
#   sbatch scripts/sbatch_d3tw_ngram_w32_light_orthogonal_10k_v1.sbatch
#
# Optional:
#   CONDA_ENV=manucripts_align sbatch --export=ALL scripts/sbatch_d3tw_ngram_w32_light_orthogonal_10k_v1.sbatch
#
#SBATCH --job-name=d3tw_ngram
#SBATCH --output=out/%x_%J.out
#SBATCH --error=out/%x_%J.err
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=03-00:00:00
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8

#SBATCH --mail-type=All
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject_clone}}"
cd "${PROJECT_DIR}"
mkdir -p out logs

if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export NUM_SAMPLES=10000
export WANDB_MODE=online
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GRADIENT_CHECK_ENABLED=1
export GRADIENT_CHECK_INTERVAL=50
export GRADIENT_CHECK_FIRST_N_BATCHES=3
export GRADIENT_FAIL_FAST=1
export VERBOSE_TRAIN_PRINTS=0

python train.py \
  --job_id d3tw_ngram_w32_light_orthogonal_10k_v1 \
  --data_dir DataSet/Synthetic_Arabic \
  --alignment_loss_type d3tw_char_pool \
  --loss_type hybrid \
  --text_unit_type ngram \
  --text_embedder_type orthogonal_char \
  --window_size 32 \
  --window_overlap_mode light_overlap \
  --negative_mode length_controlled \
  --num_negatives 10 \
  --ngram_min_n 1 \
  --ngram_max_n 3 \
  --ngram_min_freq 2 \
  --ngram_max_vocab_size 5000 \
  --ngram_skip_spaces true \
  --ngram_include_ligatures true \
  --token_pool_weight 0.25 \
  --token_pool_tau 0.07 \
  --token_pool_warmup_epochs 8 \
  --token_pool_ramp_epochs 12 \
  --token_pool_detach_alignment true \
  --use_char_aux_loss true \
  --char_aux_weight 0.25 \
  --char_aux_tau 0.07 \
  --char_aux_warmup_epochs 8 \
  --char_aux_ramp_epochs 10 \
  --use_bigram_token_loss false \
  --sequence_encoder_type bilstm \
  --epochs 60 \
  --learning_rate 1e-4

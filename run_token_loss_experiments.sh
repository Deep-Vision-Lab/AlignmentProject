#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic_100000}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
NUM_NEG="${NUM_NEG:-10}"
DRY_RUN="${DRY_RUN:-0}"

run_experiment() {
  local job_id="$1"
  local overlap_mode="$2"
  local negative_mode="$3"

  local cmd=(
    python train.py
    --job_id "$job_id"
    --data_dir "$DATA_DIR"
    --alignment_loss_type d3tw_char_pool
    --text_embedder_type char
    --window_size 32
    --window_overlap_mode "$overlap_mode"
    --negative_mode "$negative_mode"
    --num_negatives "$NUM_NEG"
    --char_pool_method hard_mean
    --char_pool_weight 0.5
    --char_pool_tau 0.07
    --char_pool_warmup_epochs 5
    --char_pool_ramp_epochs 10
    --use_bigram_token_loss true
    --bigram_token_weight 0.25
    --bigram_token_tau 0.07
    --bigram_token_warmup_epochs 8
    --bigram_token_ramp_epochs 10
    --bigram_token_fusion mean
    --epochs "$EPOCHS"
    --learning_rate "$LR"
  )

  printf '\n'
  printf 'Running %s\n' "$job_id"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
}

run_experiment B1_charpool_bigram_w32_nooverlap no_overlap length_controlled
run_experiment B2_charpool_bigram_w32_lightoverlap light_overlap length_controlled
run_experiment B3_charpool_bigram_w32_lightoverlap_dotconf light_overlap dot_confusion

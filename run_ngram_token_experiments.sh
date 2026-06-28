#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic_100000}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
NUM_NEG="${NUM_NEG:-10}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_cmd python train.py \
  --job_id N1_ngram_d3tw_tokenpool_w32_nooverlap \
  --data_dir "$DATA_DIR" \
  --alignment_loss_type d3tw_char_pool \
  --text_unit_type ngram \
  --text_embedder_type char \
  --window_size 32 \
  --window_overlap_mode no_overlap \
  --negative_mode length_controlled \
  --num_negatives "$NUM_NEG" \
  --ngram_min_n 1 \
  --ngram_max_n 3 \
  --ngram_min_freq 2 \
  --ngram_max_vocab_size 5000 \
  --token_pool_weight 0.5 \
  --token_pool_tau 0.07 \
  --token_pool_warmup_epochs 5 \
  --token_pool_ramp_epochs 10 \
  --use_char_aux_loss true \
  --char_aux_weight 0.25 \
  --epochs "$EPOCHS" \
  --learning_rate "$LR"

run_cmd python train.py \
  --job_id N2_ngram_d3tw_tokenpool_w32_lightoverlap \
  --data_dir "$DATA_DIR" \
  --alignment_loss_type d3tw_char_pool \
  --text_unit_type ngram \
  --text_embedder_type char \
  --window_size 32 \
  --window_overlap_mode light_overlap \
  --negative_mode length_controlled \
  --num_negatives "$NUM_NEG" \
  --ngram_min_n 1 \
  --ngram_max_n 3 \
  --ngram_min_freq 2 \
  --ngram_max_vocab_size 5000 \
  --token_pool_weight 0.5 \
  --token_pool_tau 0.07 \
  --token_pool_warmup_epochs 5 \
  --token_pool_ramp_epochs 10 \
  --use_char_aux_loss true \
  --char_aux_weight 0.25 \
  --epochs "$EPOCHS" \
  --learning_rate "$LR"

run_cmd python train.py \
  --job_id N3_ngram_d3tw_tokenpool_w32_lightoverlap_dotconf \
  --data_dir "$DATA_DIR" \
  --alignment_loss_type d3tw_char_pool \
  --text_unit_type ngram \
  --text_embedder_type char \
  --window_size 32 \
  --window_overlap_mode light_overlap \
  --negative_mode dot_confusion \
  --num_negatives "$NUM_NEG" \
  --ngram_min_n 1 \
  --ngram_max_n 3 \
  --ngram_min_freq 2 \
  --ngram_max_vocab_size 5000 \
  --token_pool_weight 0.5 \
  --token_pool_tau 0.07 \
  --token_pool_warmup_epochs 5 \
  --token_pool_ramp_epochs 10 \
  --use_char_aux_loss true \
  --char_aux_weight 0.25 \
  --epochs "$EPOCHS" \
  --learning_rate "$LR"

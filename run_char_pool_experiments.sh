#!/bin/bash
# Launch only the requested D3TW-guided character-pooling experiments M1-M2.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR=${DATA_DIR:-DataSet/Synthetic_Arabic_10000}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}
NUM_NEG=${NUM_NEG:-10}
DRY_RUN=${DRY_RUN:-0}
PYTHON=${PYTHON:-python}

run_experiment() {
  local job_id="$1"
  local overlap_mode="$2"
  local negative_mode="$3"
  local cmd=(
    "$PYTHON" train.py
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
    --epochs "$EPOCHS"
    --learning_rate "$LR"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'cd %q &&' "$ROOT"
    printf ' %q' "${cmd[@]}"
    echo
  else
    (cd "$ROOT" && "${cmd[@]}")
  fi
}

run_experiment M1_d3tw_charpool_w32_nooverlap no_overlap length_controlled
run_experiment M2_d3tw_charpool_w32_lightoverlap light_overlap length_controlled

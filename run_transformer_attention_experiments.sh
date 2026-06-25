#!/bin/bash
# Launch only the new Transformer / optional attention experiments T1-T4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-DataSet/Synthetic_Arabic_10000}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}
NUM_NEG=${NUM_NEG:-10}
DRY_RUN=${DRY_RUN:-0}
USE_SLURM=${USE_SLURM:-0}
SBATCH_TEMPLATE=${SBATCH_TEMPLATE:-"$PROJECT_ROOT/transformer_attention_sbatch_template.sbatch"}
ENV_NAME=${ENV_NAME:-manucripts_align}
MODEL_DIR=${MODEL_DIR:-AlignmentProject_clone}

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "$@"
    echo
  else
    "$@"
  fi
}

run_train_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  cd %q' "$PROJECT_ROOT"
    printf ' &&'
    printf ' %q' "$@"
    echo
  else
    (cd "$PROJECT_ROOT" && "$@")
  fi
}

add_export() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    export_arg+="${key}=${value},"
  fi
}

submit_or_run() {
  local job_id="$1"
  local alignment_loss_type="$2"
  local text_embedder_type="$3"
  local negative_mode="$4"
  local ctc_weight="$5"
  local d3tw_weight="$6"
  local contrastive_ctc_loss_type="$7"
  local contrastive_ctc_tau="$8"
  local use_cross_attention="$9"
  local cross_attention_weight="${10}"

  echo
  echo "Experiment: $job_id"
  echo "  alignment_loss_type=$alignment_loss_type data=$DATA_DIR epochs=$EPOCHS lr=$LR"

  if [[ "$USE_SLURM" == "1" ]]; then
    export_arg="ALL,"
    add_export JOB_ID "$job_id"
    add_export env "$ENV_NAME"
    add_export model_dir "$MODEL_DIR"
    add_export DATA_DIR "$DATA_DIR"
    add_export ALIGNMENT_LOSS_TYPE "$alignment_loss_type"
    add_export TEXT_EMBEDDER_TYPE "$text_embedder_type"
    add_export NEGATIVE_MODE "$negative_mode"
    add_export NUM_NEGATIVES "$NUM_NEG"
    add_export CTC_WEIGHT "$ctc_weight"
    add_export D3TW_WEIGHT "$d3tw_weight"
    add_export CONTRASTIVE_CTC_LOSS_TYPE "$contrastive_ctc_loss_type"
    add_export CONTRASTIVE_CTC_TAU "$contrastive_ctc_tau"
    add_export USE_CROSS_ATTENTION "$use_cross_attention"
    add_export CROSS_ATTENTION_WEIGHT "$cross_attention_weight"
    add_export EPOCHS "$EPOCHS"
    add_export LR "$LR"
    export_arg="${export_arg%,}"
    run_or_print sbatch --job-name="$job_id" --export="$export_arg" "$SBATCH_TEMPLATE"
    return
  fi

  cmd=(python train.py
    --job_id "$job_id"
    --data_dir "$DATA_DIR"
    --alignment_loss_type "$alignment_loss_type"
    --sequence_encoder_type transformer
    --window_size 32
    --stride_ratio 0.5
    --negative_mode "$negative_mode"
    --num_negatives "$NUM_NEG"
    --transformer_num_layers 2
    --transformer_num_heads 4
    --transformer_ff_dim 512
    --transformer_dropout 0.1
    --transformer_positional_encoding sinusoidal
    --epochs "$EPOCHS"
    --learning_rate "$LR"
  )

  if [[ "$text_embedder_type" != "none" ]]; then
    cmd+=(--text_embedder_type "$text_embedder_type")
  fi
  if [[ -n "$contrastive_ctc_loss_type" ]]; then
    cmd+=(--contrastive_ctc_loss_type "$contrastive_ctc_loss_type")
  fi
  if [[ -n "$contrastive_ctc_tau" ]]; then
    cmd+=(--contrastive_ctc_tau "$contrastive_ctc_tau")
  fi
  if [[ -n "$ctc_weight" ]]; then
    cmd+=(--ctc_weight "$ctc_weight")
  fi
  if [[ -n "$d3tw_weight" ]]; then
    cmd+=(--d3tw_weight "$d3tw_weight")
  fi
  if [[ "$use_cross_attention" == "1" ]]; then
    cmd+=(--use_cross_attention)
    cmd+=(--cross_attention_type text_to_image)
    cmd+=(--cross_attention_num_heads 4)
    cmd+=(--cross_attention_dropout 0.1)
    cmd+=(--cross_attention_weight "$cross_attention_weight")
  fi

  run_train_or_print "${cmd[@]}"
}

echo "============================================================"
echo "Transformer / Attention Experiment Launcher (T1-T4 only)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "EPOCHS=$EPOCHS LR=$LR NUM_NEG=$NUM_NEG"
echo "USE_SLURM=$USE_SLURM DRY_RUN=$DRY_RUN"
echo "============================================================"

submit_or_run "T1_transformer_d3tw_w32_lenctrl" \
  "d3tw" "orthogonal_char" "length_controlled" \
  "" "" "" "" "0" ""

submit_or_run "T2_transformer_contrastive_ctc_w32_lenctrl" \
  "contrastive_ctc" "none" "length_controlled" \
  "" "" "infonce" "0.1" "0" ""

submit_or_run "T3_transformer_contrastive_ctc_d3tw_w32_dotconf" \
  "contrastive_ctc_d3tw" "orthogonal_char" "dot_confusion" \
  "1.0" "0.5" "infonce" "0.1" "0" ""

submit_or_run "T4_transformer_d3tw_crossattn_w32_lenctrl" \
  "d3tw" "orthogonal_char" "length_controlled" \
  "" "" "" "" "1" "0.2"

echo
echo "Done."

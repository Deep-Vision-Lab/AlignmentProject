#!/bin/bash
################################################################################################
# SLURM template for Transformer / attention Arabic alignment experiments.
################################################################################################
#SBATCH --partition=rtx6000
#SBATCH --qos=jelsana
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=ALL

set -euo pipefail

env="${env:-manucripts_align}"
export TRANSFORMER_NUM_LAYERS="${TRANSFORMER_NUM_LAYERS:-2}"
export TRANSFORMER_NUM_HEADS="${TRANSFORMER_NUM_HEADS:-4}"
export TRANSFORMER_FF_DIM="${TRANSFORMER_FF_DIM:-512}"
export TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
export TRANSFORMER_POSITIONAL_ENCODING="${TRANSFORMER_POSITIONAL_ENCODING:-sinusoidal}"
export CROSS_ATTENTION_TYPE="${CROSS_ATTENTION_TYPE:-text_to_image}"
export CROSS_ATTENTION_NUM_HEADS="${CROSS_ATTENTION_NUM_HEADS:-4}"
export CROSS_ATTENTION_DROPOUT="${CROSS_ATTENTION_DROPOUT:-0.1}"

mkdir -p out

echo "============================================================"
echo "Job started: $(date)"
echo "SLURM_JOBID:              ${SLURM_JOBID:-NA}"
echo "ENVIRONMENT:              ${env:-<unset>}"
echo "PROJECT:                  ${model_dir:-<unset>}"
echo "JOB_ID:                   ${JOB_ID:-<unset>}"
echo "DATA_DIR:                 ${DATA_DIR:-<default>}"
echo "ALIGNMENT_LOSS_TYPE:      ${ALIGNMENT_LOSS_TYPE:-d3tw}"
echo "TEXT_EMBEDDER_TYPE:       ${TEXT_EMBEDDER_TYPE:-<default>}"
echo "NEGATIVE_MODE:            ${NEGATIVE_MODE:-length_controlled}"
echo "NUM_NEGATIVES:            ${NUM_NEGATIVES:-10}"
echo "USE_CROSS_ATTENTION:      ${USE_CROSS_ATTENTION:-0}"
echo "CROSS_ATTENTION_WEIGHT:   ${CROSS_ATTENTION_WEIGHT:-0.0}"
echo "EPOCHS:                   ${EPOCHS:-50}"
echo "LR:                       ${LR:-1e-4}"
echo "============================================================"

if [[ -z "${JOB_ID:-}" ]]; then
  echo "ERROR: JOB_ID must be exported."
  exit 1
fi
if [[ -z "${model_dir:-}" ]]; then
  echo "ERROR: model_dir must be exported, e.g. model_dir=AlignmentProject_clone."
  exit 1
fi

cd "$HOME/BGU-Lab/$model_dir"

module load anaconda
eval "$(conda shell.bash hook)"
conda activate "$env"

cmd=(python train.py
  --job_id "$JOB_ID"
  --data_dir "${DATA_DIR:-DataSet/Synthetic_Arabic_10000}"
  --alignment_loss_type "${ALIGNMENT_LOSS_TYPE:-d3tw}"
  --sequence_encoder_type transformer
  --window_size "${WINDOW_SIZE:-32}"
  --stride_ratio "${STRIDE_RATIO:-0.5}"
  --negative_mode "${NEGATIVE_MODE:-length_controlled}"
  --num_negatives "${NUM_NEGATIVES:-10}"
  --transformer_num_layers "$TRANSFORMER_NUM_LAYERS"
  --transformer_num_heads "$TRANSFORMER_NUM_HEADS"
  --transformer_ff_dim "$TRANSFORMER_FF_DIM"
  --transformer_dropout "$TRANSFORMER_DROPOUT"
  --transformer_positional_encoding "$TRANSFORMER_POSITIONAL_ENCODING"
  --epochs "${EPOCHS:-50}"
  --learning_rate "${LR:-1e-4}"
)

append_kv() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" && "$value" != "none" ]]; then
    cmd+=("$flag" "$value")
  fi
}

append_kv --text_embedder_type "${TEXT_EMBEDDER_TYPE:-}"
append_kv --ctc_weight "${CTC_WEIGHT:-}"
append_kv --d3tw_weight "${D3TW_WEIGHT:-}"
append_kv --contrastive_ctc_loss_type "${CONTRASTIVE_CTC_LOSS_TYPE:-}"
append_kv --contrastive_ctc_tau "${CONTRASTIVE_CTC_TAU:-}"

if [[ "${USE_CROSS_ATTENTION:-0}" == "1" ]]; then
  cmd+=(--use_cross_attention)
  cmd+=(--cross_attention_type "$CROSS_ATTENTION_TYPE")
  cmd+=(--cross_attention_num_heads "$CROSS_ATTENTION_NUM_HEADS")
  cmd+=(--cross_attention_dropout "$CROSS_ATTENTION_DROPOUT")
  cmd+=(--cross_attention_weight "${CROSS_ATTENTION_WEIGHT:-0.0}")
fi

echo "Training command:"
printf '  %q' "${cmd[@]}"
echo
"${cmd[@]}"

echo "Job finished: $(date)"

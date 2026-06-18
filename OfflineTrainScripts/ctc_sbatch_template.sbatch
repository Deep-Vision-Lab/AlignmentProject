#!/bin/bash
################################################################################################
# SLURM template for CTC-family Arabic alignment experiments.
################################################################################################
#SBATCH --partition=jelsana
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --tasks=6
#SBATCH --mail-type=ALL

set -euo pipefail

env="${env:-manucripts_align}"
export CTC_WEIGHT="${CTC_WEIGHT:-1.0}"
export D3TW_WEIGHT="${D3TW_WEIGHT:-1.0}"
export CONTRASTIVE_CTC_LOSS_TYPE="${CONTRASTIVE_CTC_LOSS_TYPE:-infonce}"
export CONTRASTIVE_CTC_TAU="${CONTRASTIVE_CTC_TAU:-0.1}"
export CONTRASTIVE_CTC_MARGIN="${CONTRASTIVE_CTC_MARGIN:-0.2}"

mkdir -p out

echo "============================================================"
echo "Job started: $(date)"
echo "SLURM_JOBID:              ${SLURM_JOBID:-NA}"
echo "SLURM_JOB_NODELIST:       ${SLURM_JOB_NODELIST:-NA}"
echo "ENVIRONMENT:              ${env:-<unset>}"
echo "PROJECT:                  ${model_dir:-<unset>}"
echo "JOB_ID:                   ${JOB_ID:-<unset>}"
echo "DATA_DIR:                 ${DATA_DIR:-<default>}"
echo "ALIGNMENT_LOSS_TYPE:      ${ALIGNMENT_LOSS_TYPE:-d3tw}"
echo "TEXT_EMBEDDER_TYPE:       ${TEXT_EMBEDDER_TYPE:-<default>}"
echo "WINDOW_SIZE:              ${WINDOW_SIZE:-<default>}"
echo "STRIDE_RATIO:             ${STRIDE_RATIO:-<default>}"
echo "NEGATIVE_MODE:            ${NEGATIVE_MODE:-<default>}"
echo "NUM_NEGATIVES:            ${NUM_NEGATIVES:-<default>}"
echo "CTC_WEIGHT:               ${CTC_WEIGHT:-1.0}"
echo "D3TW_WEIGHT:              ${D3TW_WEIGHT:-1.0}"
echo "CONTRASTIVE_CTC_LOSS_TYPE:${CONTRASTIVE_CTC_LOSS_TYPE:-infonce}"
echo "CONTRASTIVE_CTC_TAU:      ${CONTRASTIVE_CTC_TAU:-0.1}"
echo "EPOCHS:                   ${EPOCHS:-<default>}"
echo "LR:                       ${LR:-<default>}"
echo "GENERATE_FIGURES:         ${GENERATE_FIGURES:-False}"
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
  --alignment_loss_type "${ALIGNMENT_LOSS_TYPE:-d3tw}"
)

append_kv() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" && "$value" != "none" ]]; then
    cmd+=("$flag" "$value")
  fi
}

append_kv --data_dir "${DATA_DIR:-}"
append_kv --text_embedder_type "${TEXT_EMBEDDER_TYPE:-}"
append_kv --window_size "${WINDOW_SIZE:-}"
append_kv --stride_ratio "${STRIDE_RATIO:-}"
append_kv --negative_mode "${NEGATIVE_MODE:-}"
append_kv --num_negatives "${NUM_NEGATIVES:-}"
append_kv --ctc_weight "${CTC_WEIGHT:-}"
append_kv --d3tw_weight "${D3TW_WEIGHT:-}"
append_kv --contrastive_ctc_loss_type "${CONTRASTIVE_CTC_LOSS_TYPE:-}"
append_kv --contrastive_ctc_tau "${CONTRASTIVE_CTC_TAU:-}"
append_kv --epochs "${EPOCHS:-}"
append_kv --learning_rate "${LR:-}"

echo "Training command:"
printf '  %q' "${cmd[@]}"; echo
"${cmd[@]}"

ckpt="Weights/${JOB_ID}/model_latest.pth"
if [[ ! -f "$ckpt" ]]; then
  echo "WARNING: expected checkpoint not found after training: $ckpt"
fi

if [[ "$(echo "${GENERATE_FIGURES:-False}" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
  echo "Generating diagnostic figures for $JOB_ID"
  fig_dir="outputs/experiments/${JOB_ID}/figures"
  mkdir -p "$fig_dir"

  if [[ -f "$ckpt" ]]; then
    python paper_figures/fig05_pos_neg_costs.py \
      --checkpoint "$ckpt" \
      --num_samples "${FIG_NUM_SAMPLES:-20}" \
      --num_negatives "${NUM_NEGATIVES:-10}" \
      --negative_mode "${NEGATIVE_MODE:-length_controlled}" \
      --cost_type ctc \
      --output_dir "$fig_dir" \
      --device cuda || true

    python paper_figures/fig08_hard_negative_case.py \
      --checkpoint "$ckpt" \
      --sentence_idx "${FIG_SENTENCE_IDX:-0}" \
      --num_negatives "${NUM_NEGATIVES:-10}" \
      --negative_mode "${NEGATIVE_MODE:-length_controlled}" \
      --cost_type ctc \
      --output_dir "$fig_dir" \
      --device cuda || true

    case "${ALIGNMENT_LOSS_TYPE:-}" in
      ctc_d3tw|contrastive_ctc_d3tw)
        python paper_figures/fig05_pos_neg_costs.py \
          --checkpoint "$ckpt" \
          --num_samples "${FIG_NUM_SAMPLES:-20}" \
          --num_negatives "${NUM_NEGATIVES:-10}" \
          --negative_mode "${NEGATIVE_MODE:-length_controlled}" \
          --cost_type d3tw \
          --output_dir "$fig_dir" \
          --device cuda || true
        python paper_figures/fig08_hard_negative_case.py \
          --checkpoint "$ckpt" \
          --sentence_idx "${FIG_SENTENCE_IDX:-0}" \
          --num_negatives "${NUM_NEGATIVES:-10}" \
          --negative_mode "${NEGATIVE_MODE:-length_controlled}" \
          --cost_type d3tw \
          --output_dir "$fig_dir" \
          --device cuda || true
        ;;
    esac
  fi
fi

echo "Job finished: $(date)"

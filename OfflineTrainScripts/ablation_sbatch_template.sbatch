#!/bin/bash
################################################################################################
# Generic SLURM template for Arabic alignment ablation experiments.
# This template is driven by environment variables exported from run_ablations.sh.
################################################################################################
#SBATCH --partition=jelsana
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --tasks=6
#SBATCH --mail-type=FAIL,END

set -euo pipefail

mkdir -p out

echo "============================================================"
echo "Job started: $(date)"
echo "SLURM_JOBID:        ${SLURM_JOBID:-NA}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-NA}"
echo "ENVIRONMENT:        ${env:-<unset>}"
echo "PROJECT:            ${model_dir:-<unset>}"
echo "JOB_ID:             ${JOB_ID:-<unset>}"
echo "DATA_DIR:           ${DATA_DIR:-<default>}"
echo "TEXT_EMBEDDER_TYPE: ${TEXT_EMBEDDER_TYPE:-<default>}"
echo "WINDOW_SIZE:        ${WINDOW_SIZE:-<default>}"
echo "STRIDE_RATIO:       ${STRIDE_RATIO:-<default>}"
echo "NEGATIVE_MODE:      ${NEGATIVE_MODE:-<default>}"
echo "NUM_NEGATIVES:      ${NUM_NEGATIVES:-<default>}"
echo "MULTI_SCALE_ENABLED:${MULTI_SCALE_ENABLED:-False}"
echo "MULTI_SCALE_WINDOWS:${MULTI_SCALE_WINDOW_SIZES:-<default>}"
echo "LOSS_TYPE:          ${CONTRASTIVE_LOSS_TYPE:-<default>}"
echo "DTW_NORM:           ${DTW_COST_NORMALIZATION:-<default>}"
echo "FINETUNE:           ${FINETUNE:-False}"
echo "PRETRAINED_WEIGHTS: ${PRETRAINED_WEIGHTS:-<none>}"
echo "RESUME:             ${RESUME:-<none>}"
echo "GENERATE_FIGURES:   ${GENERATE_FIGURES:-False}"
echo "============================================================"

if [[ -z "${JOB_ID:-}" ]]; then
  echo "ERROR: JOB_ID must be exported."
  exit 1
fi
if [[ -z "${env:-}" ]]; then
  echo "ERROR: env must be exported, e.g. env=manucripts_align."
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

# Export env vars under the names Parameters.py actually reads via os.environ.get().
# The run_ablations.sh launcher uses longer descriptive names; map them here.
export TEXT_EMBEDDER="${TEXT_EMBEDDER_TYPE:-${TEXT_EMBEDDER:-fasttext}}"
export NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
export MULTI_SCALE_ENABLED="${MULTI_SCALE_ENABLED:-False}"
export WINDOW_SIZE="${WINDOW_SIZE:-}"
export STRIDE_RATIO="${STRIDE_RATIO:-}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-}"
# Parameters.py reads LOSS_TYPE, not CONTRASTIVE_LOSS_TYPE
export LOSS_TYPE="${CONTRASTIVE_LOSS_TYPE:-${LOSS_TYPE:-margin}}"
export CONTRASTIVE_LOSS_TYPE="$LOSS_TYPE"
# Parameters.py reads DTW_NORM, not DTW_COST_NORMALIZATION
export DTW_NORM="${DTW_COST_NORMALIZATION:-${DTW_NORM:-max_len}}"
export DTW_COST_NORMALIZATION="$DTW_NORM"
export USE_BLANK_TOKEN="${USE_BLANK_TOKEN:-}"

# Build the train.py command in a backwards-compatible way: only pass a CLI
# option if train.py --help advertises it. This prevents crashes if a flag has
# not been implemented yet and allows Parameters.py/env-var fallback.
TRAIN_HELP="$(python train.py --help 2>/dev/null || true)"
cmd=(python train.py --job_id "$JOB_ID")

supports_arg() {
  local flag="$1"
  echo "$TRAIN_HELP" | grep -q -- "$flag"
}
append_kv_if_supported() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]] && supports_arg "$flag"; then
    cmd+=("$flag" "$value")
  elif [[ -n "$value" ]]; then
    echo "NOTE: train.py does not advertise $flag; keeping value in environment only."
  fi
}
append_flag_if_true_supported() {
  local flag="$1"
  local value="$2"
  local v="$(echo "$value" | tr '[:upper:]' '[:lower:]')"
  if [[ "$v" == "true" || "$v" == "1" || "$v" == "yes" ]]; then
    if supports_arg "$flag"; then
      cmd+=("$flag")
    else
      echo "NOTE: train.py does not advertise $flag; keeping value in environment only."
    fi
  fi
}

append_kv_if_supported "--data_dir"             "${DATA_DIR:-}"
append_kv_if_supported "--pretrained_weights"  "${PRETRAINED_WEIGHTS:-}"
append_kv_if_supported "--resume"              "${RESUME:-}"
append_flag_if_true_supported "--finetune"      "${FINETUNE:-False}"

# Use the flag names that train.py actually advertises (verified against --help output).
append_kv_if_supported "--text_embedder"       "${TEXT_EMBEDDER:-}"       # train.py: --text_embedder
append_kv_if_supported "--window_size"         "${WINDOW_SIZE:-}"
append_kv_if_supported "--stride_ratio"        "${STRIDE_RATIO:-}"
append_kv_if_supported "--negative_mode"       "${NEGATIVE_MODE:-}"
append_kv_if_supported "--loss_type"           "${LOSS_TYPE:-}"           # train.py: --loss_type
append_kv_if_supported "--diagonal_prior_weight" "${DIAGONAL_PRIOR_WEIGHT:-}"
append_flag_if_true_supported "--multi_scale"  "${MULTI_SCALE_ENABLED:-False}"  # train.py: --multi_scale

# Multi-scale windows may be implemented as nargs='+'; pass only if supported.
if [[ -n "${MULTI_SCALE_WINDOW_SIZES:-}" ]] && supports_arg "--multi_scale_window_sizes"; then
  # shellcheck disable=SC2206
  ms_ws=( ${MULTI_SCALE_WINDOW_SIZES} )
  cmd+=(--multi_scale_window_sizes "${ms_ws[@]}")
fi

append_flag_if_true_supported "--use_blank_token" "${USE_BLANK_TOKEN:-False}"

echo "Training command:"
printf '  %q' "${cmd[@]}"; echo
"${cmd[@]}"

ckpt="Weights/${JOB_ID}/model_latest.pth"
if [[ ! -f "$ckpt" ]]; then
  echo "WARNING: expected checkpoint not found after training: $ckpt"
fi

# Optional post-training figure generation. This is diagnostic only and should
# not fail the training job if a plotting dependency is missing.
if [[ "$(echo "${GENERATE_FIGURES:-False}" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
  echo "Generating diagnostic figures for $JOB_ID"
  fig_dir="outputs/experiments/${JOB_ID}/figures"
  mkdir -p "$fig_dir"

  if [[ -f "$ckpt" ]]; then
    python paper_figures/fig05_pos_neg_costs.py \
      --checkpoint "$ckpt" \
      --num_samples "${FIG_NUM_SAMPLES:-20}" \
      --num_negatives "${NUM_NEGATIVES:-10}" \
      --output_dir "$fig_dir" \
      --device cuda || true

    python paper_figures/fig08_hard_negative_case.py \
      --checkpoint "$ckpt" \
      --sentence_idx "${FIG_SENTENCE_IDX:-0}" \
      --num_negatives "${NUM_NEGATIVES:-10}" \
      --output_dir "$fig_dir" \
      --device cuda || true
  fi
fi

echo "Job finished: $(date)"

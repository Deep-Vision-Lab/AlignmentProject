#!/bin/bash
# Launch only the CTC-family experiments E8-E11 from OfflineTrainScripts/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR=${DATA_DIR:-DataSet/Synthetic_Arabic_100000}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}
NUM_NEG=${NUM_NEG:-10}
DRY_RUN=${DRY_RUN:-0}
# OfflineTrainScripts is intended for cluster submission by default.
# Override with USE_SLURM=0 to run train.py directly from the project root.
USE_SLURM=${USE_SLURM:-1}
SBATCH_TEMPLATE=${SBATCH_TEMPLATE:-"$SCRIPT_DIR/ctc_sbatch_template.sbatch"}
ENV_NAME=${ENV_NAME:-manucripts_align}
MODEL_DIR=${MODEL_DIR:-AlignmentProject_clone}
GENERATE_FIGURES=${GENERATE_FIGURES:-False}
SUBMISSION_LOG=${SUBMISSION_LOG:-"$SCRIPT_DIR/out/ctc_submit_$(date +%Y%m%d_%H%M%S).log"}
SBATCH_RETRIES=${SBATCH_RETRIES:-5}
SBATCH_RETRY_SLEEP=${SBATCH_RETRY_SLEEP:-15}
SBATCH_TIMEOUT=${SBATCH_TIMEOUT:-60}
SUBMIT_FAILURES=0

mkdir -p "$SCRIPT_DIR/out"

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  %q' "$@"
    echo
  else
    "$@"
  fi
}

submit_sbatch() {
  local attempt=1
  while true; do
    if [[ "$DRY_RUN" == "1" ]]; then
      run_or_print "$@"
      return 0
    fi

    if timeout "${SBATCH_TIMEOUT}s" "$@"; then
      return 0
    fi

    if [[ "$attempt" -ge "$SBATCH_RETRIES" ]]; then
      return 1
    fi

    echo "WARN: sbatch failed on attempt $attempt/$SBATCH_RETRIES; retrying in ${SBATCH_RETRY_SLEEP}s..."
    sleep "$SBATCH_RETRY_SLEEP"
    attempt=$((attempt + 1))
  done
}

add_export() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    export_arg+="${key}=${value},"
  fi
}

run_train_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  cd %q &&' "$PROJECT_ROOT"
    printf ' %q' "$@"
    echo
  else
    (cd "$PROJECT_ROOT" && "$@")
  fi
}

submit_exp() {
  local job_id="$1"
  local alignment_loss_type="$2"
  local text_embedder_type="$3"
  local window_size="$4"
  local stride_ratio="$5"
  local negative_mode="$6"
  local num_negatives="$7"
  local ctc_weight="$8"
  local d3tw_weight="$9"
  local contrastive_ctc_loss_type="${10}"
  local contrastive_ctc_tau="${11}"

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
    add_export WINDOW_SIZE "$window_size"
    add_export STRIDE_RATIO "$stride_ratio"
    add_export NEGATIVE_MODE "$negative_mode"
    add_export NUM_NEGATIVES "$num_negatives"
    add_export CTC_WEIGHT "$ctc_weight"
    add_export D3TW_WEIGHT "$d3tw_weight"
    add_export CONTRASTIVE_CTC_LOSS_TYPE "$contrastive_ctc_loss_type"
    add_export CONTRASTIVE_CTC_TAU "$contrastive_ctc_tau"
    add_export EPOCHS "$EPOCHS"
    add_export LR "$LR"
    add_export GENERATE_FIGURES "$GENERATE_FIGURES"
    export_arg="${export_arg%,}"
    local submit_output
    if submit_output=$(submit_sbatch sbatch --job-name="$job_id" --export="$export_arg" "$SBATCH_TEMPLATE" 2>&1); then
      echo "$submit_output"
      submit_output="${submit_output//$'\n'/ }"
      printf '%s OK %s %s\n' "$(date)" "$job_id" "$submit_output" >> "$SUBMISSION_LOG"
    else
      echo "$submit_output"
      echo "ERROR: sbatch submission failed for $job_id"
      submit_output="${submit_output//$'\n'/ }"
      printf '%s ERROR %s %s\n' "$(date)" "$job_id" "$submit_output" >> "$SUBMISSION_LOG"
      SUBMIT_FAILURES=$((SUBMIT_FAILURES + 1))
    fi
    return
  fi

  cmd=(python train.py
    --job_id "$job_id"
    --data_dir "$DATA_DIR"
    --alignment_loss_type "$alignment_loss_type"
    --window_size "$window_size"
    --stride_ratio "$stride_ratio"
    --negative_mode "$negative_mode"
    --epochs "$EPOCHS"
    --learning_rate "$LR"
  )

  if [[ "$text_embedder_type" != "none" ]]; then
    cmd+=(--text_embedder_type "$text_embedder_type")
  fi
  if [[ "$num_negatives" != "0" ]]; then
    cmd+=(--num_negatives "$num_negatives")
  fi
  if [[ "$ctc_weight" != "" ]]; then
    cmd+=(--ctc_weight "$ctc_weight")
  fi
  if [[ "$d3tw_weight" != "" ]]; then
    cmd+=(--d3tw_weight "$d3tw_weight")
  fi
  if [[ "$contrastive_ctc_loss_type" != "" ]]; then
    cmd+=(--contrastive_ctc_loss_type "$contrastive_ctc_loss_type")
  fi
  if [[ "$contrastive_ctc_tau" != "" ]]; then
    cmd+=(--contrastive_ctc_tau "$contrastive_ctc_tau")
  fi

  run_train_or_print "${cmd[@]}"
}

echo "============================================================"
echo "CTC Experiment Launcher (E8-E11 only)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "SBATCH_TEMPLATE=$SBATCH_TEMPLATE"
echo "SUBMISSION_LOG=$SUBMISSION_LOG"
echo "DATA_DIR=$DATA_DIR"
echo "EPOCHS=$EPOCHS LR=$LR NUM_NEG=$NUM_NEG"
echo "USE_SLURM=$USE_SLURM DRY_RUN=$DRY_RUN"
echo "SBATCH_RETRIES=$SBATCH_RETRIES SBATCH_TIMEOUT=${SBATCH_TIMEOUT}s"
echo "============================================================"

submit_exp "E8_ctc_only_w32_lenctrl" \
  "ctc" "none" "32" "0.5" "length_controlled" "0" \
  "" "" "" ""

submit_exp "E9_contrastive_ctc_w32_lenctrl" \
  "contrastive_ctc" "none" "32" "0.5" "length_controlled" "$NUM_NEG" \
  "" "" "infonce" "0.1"

submit_exp "E10_ctc_d3tw_w32_dotconf" \
  "ctc_d3tw" "orthogonal_char" "32" "0.5" "dot_confusion" "$NUM_NEG" \
  "1.0" "0.5" "" ""

submit_exp "E11_contrastive_ctc_d3tw_w32_dotconf" \
  "contrastive_ctc_d3tw" "orthogonal_char" "32" "0.5" "dot_confusion" "$NUM_NEG" \
  "1.0" "0.5" "infonce" "0.1"

echo
if [[ "$SUBMIT_FAILURES" -gt 0 ]]; then
  echo "Done with $SUBMIT_FAILURES submission failure(s). See: $SUBMISSION_LOG"
  exit 1
fi
echo "Done."

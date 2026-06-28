#!/usr/bin/env bash
set -euo pipefail

# Submit the three offline D3TW-char-pool + bigram-token experiments.
#
# Usage:
#   ./submit_bigram_offline_jobs.sh
#   DRY_RUN=1 ./submit_bigram_offline_jobs.sh
#   DATA_DIR=DataSet/Synthetic_Arabic_50000 EPOCHS=20 LR=1e-4 NUM_NEG=10 ./submit_bigram_offline_jobs.sh

MODEL_DIR="${MODEL_DIR:-AlignmentProject_clone}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
SBATCH_TEMPLATE="${SBATCH_TEMPLATE:-char_pool_sbatch_template.sbatch}"

DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic_100000}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
NUM_NEG="${NUM_NEG:-10}"
DRY_RUN="${DRY_RUN:-0}"

submit_job() {
  local job_id="$1"
  local overlap_mode="$2"
  local negative_mode="$3"

  local export_vars
  export_vars="ALL"
  export_vars+=",JOB_ID=${job_id}"
  export_vars+=",model_dir=${MODEL_DIR}"
  export_vars+=",env=${CONDA_ENV}"
  export_vars+=",DATA_DIR=${DATA_DIR}"
  export_vars+=",WINDOW_SIZE=32"
  export_vars+=",WINDOW_OVERLAP_MODE=${overlap_mode}"
  export_vars+=",TEXT_EMBEDDER_TYPE=char"
  export_vars+=",NEGATIVE_MODE=${negative_mode}"
  export_vars+=",NUM_NEGATIVES=${NUM_NEG}"
  export_vars+=",EPOCHS=${EPOCHS}"
  export_vars+=",LR=${LR}"
  export_vars+=",USE_BIGRAM_TOKEN_LOSS=true"
  export_vars+=",BIGRAM_TOKEN_WEIGHT=0.25"
  export_vars+=",BIGRAM_TOKEN_TAU=0.07"
  export_vars+=",BIGRAM_TOKEN_WARMUP_EPOCHS=8"
  export_vars+=",BIGRAM_TOKEN_RAMP_EPOCHS=10"
  export_vars+=",BIGRAM_TOKEN_FUSION=mean"

  local cmd=(
    sbatch
    "--job-name=${job_id}"
    "--export=${export_vars}"
    "${SBATCH_TEMPLATE}"
  )

  printf '\nSubmitting %s\n' "${job_id}"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

submit_job B1_charpool_bigram_w32_nooverlap no_overlap length_controlled
submit_job B2_charpool_bigram_w32_lightoverlap light_overlap length_controlled
submit_job B3_charpool_bigram_w32_lightoverlap_dotconf light_overlap dot_confusion

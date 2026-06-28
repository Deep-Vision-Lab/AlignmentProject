#!/usr/bin/env bash
set -euo pipefail

# Submit the three offline n-gram/token-level D3TW experiments.
#
# Usage:
#   ./submit_ngram_offline_jobs.sh
#   DRY_RUN=1 ./submit_ngram_offline_jobs.sh
#   DATA_DIR=DataSet/Synthetic_Arabic_50000 EPOCHS=20 LR=1e-4 NUM_NEG=10 ./submit_ngram_offline_jobs.sh

MODEL_DIR="${MODEL_DIR:-AlignmentProject_clone}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
SBATCH_TEMPLATE="${SBATCH_TEMPLATE:-ngram_token_sbatch_template.sbatch}"

DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic_100000}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
NUM_NEG="${NUM_NEG:-10}"
DRY_RUN="${DRY_RUN:-0}"
WANDB_MODE="${WANDB_MODE:-online}"

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
  export_vars+=",NGRAM_MIN_N=1"
  export_vars+=",NGRAM_MAX_N=3"
  export_vars+=",NGRAM_MIN_FREQ=2"
  export_vars+=",NGRAM_MAX_VOCAB_SIZE=5000"
  export_vars+=",TOKEN_POOL_WEIGHT=0.5"
  export_vars+=",TOKEN_POOL_TAU=0.07"
  export_vars+=",TOKEN_POOL_WARMUP_EPOCHS=5"
  export_vars+=",TOKEN_POOL_RAMP_EPOCHS=10"
  export_vars+=",USE_CHAR_AUX_LOSS=true"
  export_vars+=",CHAR_AUX_WEIGHT=0.25"
  export_vars+=",WANDB_MODE=${WANDB_MODE}"

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

submit_job N1_ngram_d3tw_tokenpool_w32_nooverlap no_overlap length_controlled
submit_job N2_ngram_d3tw_tokenpool_w32_lightoverlap light_overlap length_controlled
submit_job N3_ngram_d3tw_tokenpool_w32_lightoverlap_dotconf light_overlap dot_confusion

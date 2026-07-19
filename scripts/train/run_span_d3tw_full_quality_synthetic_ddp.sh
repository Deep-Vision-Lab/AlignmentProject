#!/usr/bin/env bash
# Submit full-quality multi-GPU DDP training on 10,000 synthetic Arabic samples.
#
# Default usage:
#   bash scripts/train/run_span_d3tw_full_quality_synthetic_ddp.sh
#
# Two GPUs with BATCH_SIZE=8 gives a global batch size of 16. Override settings
# through environment variables, for example NUM_GPUS=4 BATCH_SIZE=4.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_span_d3tw_full_quality_synthetic_ddp.sh" >&2
  echo "Override settings through environment variables, not command-line flags." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATASET_TYPE="synthetic"
export LANGUAGE="${LANGUAGE:-Arabic}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"

# Synthetic data already contains generated variations. Disable the real-only
# manifest repetition and real scan augmentation paths.
export REAL_AUGMENT=0
export REAL_TRAIN_SAMPLES_PER_EPOCH=0

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "${PROJECT_DIR}/DataSet/Synthetic_Arabic_10000" ]]; then
    DATA_DIR="${PROJECT_DIR}/DataSet/Synthetic_Arabic_10000"
  elif [[ -d "${PROJECT_DIR}/DataSet/Synthetic_Arabic" ]]; then
    DATA_DIR="${PROJECT_DIR}/DataSet/Synthetic_Arabic"
  else
    echo "ERROR: no synthetic Arabic dataset directory was found." >&2
    echo "Expected DataSet/Synthetic_Arabic_10000 or DataSet/Synthetic_Arabic." >&2
    exit 1
  fi
fi
export DATA_DIR

[[ -d "${DATA_DIR}/images" ]] || {
  echo "ERROR: synthetic images directory not found: ${DATA_DIR}/images" >&2
  exit 1
}
[[ -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic texts directory not found: ${DATA_DIR}/texts" >&2
  exit 1
}

DETECTED_SAMPLES="$(find "${DATA_DIR}/images" -maxdepth 1 -type f -name 'img1_*.png' | wc -l | tr -d ' ')"
if (( DETECTED_SAMPLES < NUM_SAMPLES )); then
  echo "ERROR: requested ${NUM_SAMPLES} synthetic samples, but only ${DETECTED_SAMPLES} img1 files were found in ${DATA_DIR}/images." >&2
  exit 1
fi

export JOB_ID="${JOB_ID:-synthetic_arabic_fullquality_10k_ddp2}"

echo "Synthetic DDP profile"
echo "  dataset type       = ${DATASET_TYPE}"
echo "  data directory     = ${DATA_DIR}"
echo "  detected samples   = ${DETECTED_SAMPLES}"
echo "  requested samples  = ${NUM_SAMPLES}"
echo "  job id             = ${JOB_ID}"
echo "  real augmentation  = ${REAL_AUGMENT}"

exec bash "${SCRIPT_DIR}/run_span_d3tw_full_quality_real_augmented_ddp.sh"

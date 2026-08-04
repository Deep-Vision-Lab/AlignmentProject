#!/usr/bin/env bash
set -euo pipefail

# The data root must contain Synthetic_Arabic_1, ..., Synthetic_Arabic_4.
DATA_ROOT="${DATA_ROOT:-DataSet}"
JOB_ID="${JOB_ID:-synthetic_arabic_4x3000_augmented}"

export SYNTHETIC_DATASET_FOLDERS="${SYNTHETIC_DATASET_FOLDERS:-Synthetic_Arabic_1,Synthetic_Arabic_2,Synthetic_Arabic_3,Synthetic_Arabic_4}"
export SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER="${SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER:-3000}"
export SYNTHETIC_VALID_SAMPLES_PER_FOLDER="${SYNTHETIC_VALID_SAMPLES_PER_FOLDER:-500}"
export SYNTHETIC_TEST_SAMPLES_PER_FOLDER="${SYNTHETIC_TEST_SAMPLES_PER_FOLDER:-500}"
export SYNTHETIC_AUGMENT="${SYNTHETIC_AUGMENT:-1}"
export SYNTHETIC_SCALE_MIN="${SYNTHETIC_SCALE_MIN:-0.85}"
export SYNTHETIC_SCALE_MAX="${SYNTHETIC_SCALE_MAX:-1.0}"
export SYNTHETIC_TRANSLATE_PCT="${SYNTHETIC_TRANSLATE_PCT:-0.05}"
export SYNTHETIC_AUGMENT_PROB="${SYNTHETIC_AUGMENT_PROB:-1.0}"

python train_synthetic_augmented.py \
  --job_id "$JOB_ID" \
  --data_dir "$DATA_ROOT" \
  --dataset_type synthetic \
  "$@"

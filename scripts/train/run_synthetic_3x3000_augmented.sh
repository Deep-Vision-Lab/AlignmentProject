#!/usr/bin/env bash
set -euo pipefail

# DATA_ROOT must contain Synthetic_Arabic_1, Synthetic_Arabic_2, Synthetic_Arabic_3.
DATA_ROOT="${DATA_ROOT:-DataSet}"
JOB_ID="${JOB_ID:-synthetic_arabic_3x3000_dense_mixed_augmented}"

export SYNTHETIC_DATASET_FOLDERS="${SYNTHETIC_DATASET_FOLDERS:-Synthetic_Arabic_1,Synthetic_Arabic_2,Synthetic_Arabic_3}"
export SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER="${SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER:-3000}"
export SYNTHETIC_VALID_SAMPLES_PER_FOLDER="${SYNTHETIC_VALID_SAMPLES_PER_FOLDER:-500}"
export SYNTHETIC_TEST_SAMPLES_PER_FOLDER="${SYNTHETIC_TEST_SAMPLES_PER_FOLDER:-500}"

# 9,000 raw training samples x 2 virtual copies = 18,000 samples per epoch.
export SYNTHETIC_AUGMENT_COPIES_PER_SAMPLE="${SYNTHETIC_AUGMENT_COPIES_PER_SAMPLE:-2}"

# Mode mixture:
#   35% aligned cross-line injection
#   30% same-line two-region composition
#   25% one aligned region + one unaligned donor region
#   10% mild full-line augmentation
export SYNTHETIC_INJECTION_PROB="${SYNTHETIC_INJECTION_PROB:-0.35}"
export SYNTHETIC_TWO_REGION_PROB="${SYNTHETIC_TWO_REGION_PROB:-0.30}"
export SYNTHETIC_ALIGNED_UNALIGNED_PROB="${SYNTHETIC_ALIGNED_UNALIGNED_PROB:-0.25}"

# Wider fragments and smaller gaps reduce empty background.
export SYNTHETIC_FRAGMENT_MIN_FRACTION="${SYNTHETIC_FRAGMENT_MIN_FRACTION:-0.20}"
export SYNTHETIC_FRAGMENT_MAX_FRACTION="${SYNTHETIC_FRAGMENT_MAX_FRACTION:-0.40}"
export SYNTHETIC_SOURCE_GAP_FRACTION="${SYNTHETIC_SOURCE_GAP_FRACTION:-0.04}"
export SYNTHETIC_CANVAS_GAP_FRACTION="${SYNTHETIC_CANVAS_GAP_FRACTION:-0.08}"
export SYNTHETIC_MISMATCH_SPAN_DISTANCE="${SYNTHETIC_MISMATCH_SPAN_DISTANCE:-0.12}"

# Mild appearance changes only.
export SYNTHETIC_SCALE_MIN="${SYNTHETIC_SCALE_MIN:-0.90}"
export SYNTHETIC_SCALE_MAX="${SYNTHETIC_SCALE_MAX:-1.00}"
export SYNTHETIC_TRANSLATE_PCT="${SYNTHETIC_TRANSLATE_PCT:-0.04}"
export SYNTHETIC_CONTRAST="${SYNTHETIC_CONTRAST:-0.10}"

python train_synthetic_augmented.py \
  --job_id "$JOB_ID" \
  --data_dir "$DATA_ROOT" \
  --dataset_type synthetic \
  "$@"

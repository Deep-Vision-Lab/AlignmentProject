#!/usr/bin/env bash
# R0: conservative real-domain adaptation using unique real lines only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest_full_pairs.jsonl}"
[[ -f "${DATA_DIR}/${REAL_MANIFEST_NAME}" ]] || {
  echo "ERROR: missing full real manifest ${DATA_DIR}/${REAL_MANIFEST_NAME}" >&2
  echo "Run scripts/data/build_full_line_pair_manifest.py first." >&2
  exit 2
}

export REAL_UNIQUE_LINE_ADAPTATION=1
export REAL_USE_EXTRA_NO_SHARED=0
export USE_IMAGE_PAIR_CONTRASTIVE=0
export IMAGE_PAIR_LOSS_WEIGHT=0
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0
export USE_SEQUENCE_ALIGNMENT_RANKING=0
export SEQUENCE_RANKING_WEIGHT=0

# Preserve the Stage-1 semantic anchor and adapt gently to manuscript appearance.
export USE_LOCAL_HARD_NEGATIVES=1
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.10}"
export LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.20}"
export LOCAL_HARD_NEGATIVE_TOP_K="${LOCAL_HARD_NEGATIVE_TOP_K:-8}"
export LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES="${LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES:-2}"
export LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH="${LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH:-8}"

export AUGMENT=0
export REAL_AUGMENT=0
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
export NUM_SAMPLES=0
export REAL_TRAIN_FRACTION="${REAL_TRAIN_FRACTION:-0.80}"
export REAL_VALID_FRACTION="${REAL_VALID_FRACTION:-0.10}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export EPOCHS="${EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-r0-image-text}"

exec bash "${SCRIPT_DIR}/run_real_finetune.sh"

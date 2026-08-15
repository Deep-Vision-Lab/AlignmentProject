#!/usr/bin/env bash
# R2: add the full no-shared relationship pool with a weak sequence objective.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest_full_pairs.jsonl}"
[[ -f "${DATA_DIR}/${REAL_MANIFEST_NAME}" ]] || {
  echo "ERROR: missing full real manifest ${DATA_DIR}/${REAL_MANIFEST_NAME}" >&2
  exit 2
}

# Keep the proven joint-real masking/ranking implementation, but make it much
# gentler than the failed 5-epoch pilot and expose the full no-shared pool.
export REAL_UNIQUE_LINE_ADAPTATION=0
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
export REAL_CLEAN_VIEWS_PER_CYCLE="${REAL_CLEAN_VIEWS_PER_CYCLE:-1}"
export REAL_AUG_VIEWS_PER_CYCLE="${REAL_AUG_VIEWS_PER_CYCLE:-1}"
export REAL_EFFECTIVE_EPOCH_MULTIPLIER="${REAL_EFFECTIVE_EPOCH_MULTIPLIER:-2}"
export REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.75}"
export REAL_AUG_STITCH_PROB=0

export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.10}"
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.08}"
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT="${SEQUENCE_CONSISTENCY_LOSS_WEIGHT:-0.015}"
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.03}"

export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export EPOCHS="${EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-r2-full-discrimination}"

exec bash "${SCRIPT_DIR}/run_stage1_joint_real_discrimination.sh"

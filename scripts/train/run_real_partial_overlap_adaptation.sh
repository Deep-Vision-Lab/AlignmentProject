#!/usr/bin/env bash
# Conservative partial-overlap real adaptation experiment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID.}"
: "${PRETRAINED_WEIGHTS:?Set PRETRAINED_WEIGHTS.}"

export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest_full_pairs.jsonl}"
if [[ ! -f "${DATA_DIR}/${REAL_MANIFEST_NAME}" ]]; then
  python scripts/data/build_full_line_pair_manifest.py \
    --root "${DATA_DIR}" \
    --output "${DATA_DIR}/${REAL_MANIFEST_NAME}"
fi

# Same leakage-safe 80/10/10 group split as the joint-real curriculum.
export NUM_SAMPLES=0
export REAL_TRAIN_FRACTION="${REAL_TRAIN_FRACTION:-0.80}"
export REAL_VALID_FRACTION="${REAL_VALID_FRACTION:-0.10}"
export REAL_SPLIT_BY_PAIR_ID=1
export REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
export REAL_USE_EXTRA_NO_SHARED=1
export REAL_EXTRA_EXCLUDE_EVAL_PAGES=1

# Online appearance perturbation remains conservative; no legacy whole-pair
# stitching is needed because the partial-overlap dataset performs its own
# train-only composition.
export AUGMENT=1
export REAL_AUGMENT=1
export REAL_CLEAN_VIEWS_PER_CYCLE="${REAL_CLEAN_VIEWS_PER_CYCLE:-1}"
export REAL_AUG_VIEWS_PER_CYCLE="${REAL_AUG_VIEWS_PER_CYCLE:-1}"
export REAL_EFFECTIVE_EPOCH_MULTIPLIER="${REAL_EFFECTIVE_EPOCH_MULTIPLIER:-4}"
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
export REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.85}"
export REAL_AUG_STITCH_PROB=0

# R2 showed that negatives can also form long paths at loose thresholds. Keep
# no-shared at 50% of every epoch, but spend more of the scarce positive half on
# controlled partial-overlap examples. Overall default exposure is therefore:
#   20% canonical high/medium positives
#   30% train-only partial-overlap positives
#   50% no_shared_content negatives
export NO_SHARED_IMAGE_OBJECTIVE=joint_partial_overlap
export REAL_PARTIAL_OVERLAP_POSITIVE_FRACTION="${REAL_PARTIAL_OVERLAP_POSITIVE_FRACTION:-0.60}"
export REAL_PARTIAL_OVERLAP_MAX_SHARED_ISLANDS="${REAL_PARTIAL_OVERLAP_MAX_SHARED_ISLANDS:-3}"
export REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB="${REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB:-0.75}"
export REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB="${REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB:-0.20}"
export REAL_PARTIAL_OVERLAP_EDGE_DISTRACTOR_PROB="${REAL_PARTIAL_OVERLAP_EDGE_DISTRACTOR_PROB:-0.55}"
export REAL_PARTIAL_OVERLAP_MAX_TEXT_CHARS="${REAL_PARTIAL_OVERLAP_MAX_TEXT_CHARS:-150}"

# Keep image-text as the anchor and give the controlled positive islands a
# slightly stronger image-pair signal than R2, without allowing it to dominate.
export USE_LOCAL_HARD_NEGATIVES=1
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.15}"
export LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.20}"
export LOCAL_HARD_NEGATIVE_TOP_K="${LOCAL_HARD_NEGATIVE_TOP_K:-8}"
export LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES="${LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES:-2}"

export USE_IMAGE_PAIR_CONTRASTIVE=1
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.10}"
export IMAGE_PAIR_MARGIN="${IMAGE_PAIR_MARGIN:-0.25}"
export IMAGE_PAIR_TOP_K="${IMAGE_PAIR_TOP_K:-8}"
# Partial-overlap pairs contain intentionally unmatched regions, so a global
# order-consistency loss would create false supervision.
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0

export USE_SEQUENCE_ALIGNMENT_RANKING=1
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.03}"
# The latest R2 fixed-manifest sweep had its best structural discrimination at
# T=0.50. extra_real_training_v4 uses this value in the same cosine-thresholded
# soft local-alignment objective, so train in the useful real-data regime.
export SEQUENCE_RANKING_THRESHOLD="${SEQUENCE_RANKING_THRESHOLD:-0.50}"
export SEQUENCE_RANKING_GAP="${SEQUENCE_RANKING_GAP:--0.30}"
export SEQUENCE_RANKING_TEMPERATURE="${SEQUENCE_RANKING_TEMPERATURE:-0.03}"
export SEQUENCE_RANKING_PATH_MARGIN="${SEQUENCE_RANKING_PATH_MARGIN:-0.03}"
export SEQUENCE_RANKING_SCORE_MARGIN="${SEQUENCE_RANKING_SCORE_MARGIN:-0.25}"
# Encourage a meaningful local path (roughly ten of ~125 windows) without
# forcing every canonical positive to align globally.
export SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR="${SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR:-0.08}"
export SEQUENCE_RANKING_MAX_POS_SAMPLES_PER_BATCH="${SEQUENCE_RANKING_MAX_POS_SAMPLES_PER_BATCH:-4}"
export SEQUENCE_RANKING_MAX_NEG_SAMPLES_PER_BATCH="${SEQUENCE_RANKING_MAX_NEG_SAMPLES_PER_BATCH:-4}"

# Keep the text-side negative budget unchanged; the current bottleneck is
# positive image-image structure, not the number of text negatives.
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-partial-overlap}"

exec bash "${SCRIPT_DIR}/run_real_finetune.sh"

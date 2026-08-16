#!/usr/bin/env bash
# Clean Stage-2-style adaptation from Stage 1:
#   * no generic real-data augmentation;
#   * canonical high/medium positives stay untouched;
#   * canonical no_shared A/B pairs stay untouched as sequence-ranking negatives;
#   * each safe no-shared row additionally has an offline synthetic partner with
#     1--3 bbox-exact aligned islands and otherwise unrelated real content.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Preserve the clean Stage-2 data policy.
export AUGMENT=0
export REAL_AUGMENT=0
export REAL_AUG_STITCH_PROB=0
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
export REAL_USE_EXTRA_NO_SHARED="${REAL_USE_EXTRA_NO_SHARED:-1}"
export REAL_EXTRA_EXCLUDE_EVAL_PAGES="${REAL_EXTRA_EXCLUDE_EVAL_PAGES:-1}"
export REAL_SYNTHETIC_PARTNER_MANIFEST="${REAL_SYNTHETIC_PARTNER_MANIFEST:-${PROJECT_DIR}/DataSet/ArabicDatasetSyntheticPartners/dataset_manifest.jsonl}"

[[ -f "${REAL_SYNTHETIC_PARTNER_MANIFEST}" ]] || {
  echo "ERROR: synthetic-partner manifest is missing:" >&2
  echo "  ${REAL_SYNTHETIC_PARTNER_MANIFEST}" >&2
  echo "Build it first with scripts/data/build_no_shared_synthetic_partners.py" >&2
  exit 2
}

export NO_SHARED_IMAGE_OBJECTIVE="sequence_ranking_partial_overlap"
export USE_SEQUENCE_ALIGNMENT_RANKING="${USE_SEQUENCE_ALIGNMENT_RANKING:-1}"
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.10}"

export SEQUENCE_RANKING_THRESHOLD="${SEQUENCE_RANKING_THRESHOLD:-0.65}"
export SEQUENCE_RANKING_GAP="${SEQUENCE_RANKING_GAP:--0.30}"
export SEQUENCE_RANKING_TEMPERATURE="${SEQUENCE_RANKING_TEMPERATURE:-0.03}"
export SEQUENCE_RANKING_PATH_MARGIN="${SEQUENCE_RANKING_PATH_MARGIN:-0.03}"
export SEQUENCE_RANKING_SCORE_MARGIN="${SEQUENCE_RANKING_SCORE_MARGIN:-0.25}"
export SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR="${SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR:-0.15}"
export SEQUENCE_RANKING_PATH_COMPONENT_WEIGHT="${SEQUENCE_RANKING_PATH_COMPONENT_WEIGHT:-1.0}"
export SEQUENCE_RANKING_SCORE_COMPONENT_WEIGHT="${SEQUENCE_RANKING_SCORE_COMPONENT_WEIGHT:-0.20}"
export SEQUENCE_RANKING_POSITIVE_FLOOR_WEIGHT="${SEQUENCE_RANKING_POSITIVE_FLOOR_WEIGHT:-0.50}"
export SEQUENCE_RANKING_MAX_POS_SAMPLES_PER_BATCH="${SEQUENCE_RANKING_MAX_POS_SAMPLES_PER_BATCH:-4}"
export SEQUENCE_RANKING_MAX_NEG_SAMPLES_PER_BATCH="${SEQUENCE_RANKING_MAX_NEG_SAMPLES_PER_BATCH:-4}"

# A synthetic partner is only partially corresponding, so a global position/order
# consistency loss would be semantically wrong. The local equal-text pair loss and
# direct local sequence-ranking objective remain active.
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0

# Keep the established text-side negative pool.
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"

exec bash "${SCRIPT_DIR}/run_real_finetune.sh"

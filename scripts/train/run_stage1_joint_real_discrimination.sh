#!/usr/bin/env bash
# Clean Stage-1 -> joint real discrimination training.
# Uses ONLY DataSet/ArabicDataset and generates fresh augmentation online.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
case "$(basename "${DATA_DIR}")" in
  ArabicDatasetRealAug10K|*Aug10K*|*aug10k*)
    echo "ERROR: this launcher must use the original ArabicDataset, not a pre-generated augmented dataset: ${DATA_DIR}" >&2
    exit 2
    ;;
esac
[[ -f "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: canonical original real manifest missing: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}

# Full canonical manifest, split by pair/page groups.  Augmentation is applied
# only after the split, so validation/test never receive augmented siblings.
export NUM_SAMPLES="${NUM_SAMPLES:-0}"
export REAL_TRAIN_FRACTION="${REAL_TRAIN_FRACTION:-0.80}"
export REAL_VALID_FRACTION="${REAL_VALID_FRACTION:-0.10}"
export REAL_SPLIT_BY_PAIR_ID=1
export REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
export REAL_USE_EXTRA_NO_SHARED=1
export REAL_EXTRA_EXCLUDE_EVAL_PAGES=1

# Every epoch contains both clean and newly generated augmented views.  A 1:2
# cycle gives ~33% clean / ~67% augmented exposure.  With multiplier=6 this is
# roughly two clean + four fresh augmented exposures per underlying training row.
export AUGMENT=1
export REAL_AUGMENT=1
export REAL_CLEAN_VIEWS_PER_CYCLE="${REAL_CLEAN_VIEWS_PER_CYCLE:-1}"
export REAL_AUG_VIEWS_PER_CYCLE="${REAL_AUG_VIEWS_PER_CYCLE:-2}"
export REAL_EFFECTIVE_EPOCH_MULTIPLIER="${REAL_EFFECTIVE_EPOCH_MULTIPLIER:-6}"
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
export REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-1.0}"
# First clean experiment: appearance/ink/scan augmentation only.  Stitching can
# be ablated later without changing this run's interpretation.
export REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0.0}"

# New final interpretation of real data: keep semantic image-text supervision,
# mine hard local confusions relatively, preserve true positive image-pair spans,
# and rank coherent positive sequences above no-shared sequences.
export NO_SHARED_IMAGE_OBJECTIVE=joint_real
export USE_LOCAL_HARD_NEGATIVES=1
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.15}"
export LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.20}"
export LOCAL_HARD_NEGATIVE_TOP_K="${LOCAL_HARD_NEGATIVE_TOP_K:-8}"
export LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES="${LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES:-2}"
export LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH="${LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH:-8}"

export USE_IMAGE_PAIR_CONTRASTIVE=1
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.25}"
export IMAGE_PAIR_MARGIN="${IMAGE_PAIR_MARGIN:-0.25}"
export IMAGE_PAIR_TOP_K="${IMAGE_PAIR_TOP_K:-8}"
export IMAGE_PAIR_MAX_SAMPLES_PER_BATCH="${IMAGE_PAIR_MAX_SAMPLES_PER_BATCH:-8}"
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT="${SEQUENCE_CONSISTENCY_LOSS_WEIGHT:-0.05}"

export USE_SEQUENCE_ALIGNMENT_RANKING=1
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

# Keep the known text-negative recipe isolated so this experiment changes only
# the real-data curriculum and image-image discrimination.
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-joint-real-discrimination}"

exec bash "${SCRIPT_DIR}/run_real_finetune.sh"

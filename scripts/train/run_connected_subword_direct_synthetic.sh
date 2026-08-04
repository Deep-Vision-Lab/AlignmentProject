#!/usr/bin/env bash
# Direct connected-subword supervision from four balanced synthetic datasets.
# Exact renderer boxes replace Span-DTW; augmentations preserve box geometry.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: JOB_ID=<name> bash scripts/train/run_connected_subword_direct_synthetic.sh" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID to the synthetic output weights-folder name.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DataSet}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
SYNTHETIC_DATA_DIRS="${SYNTHETIC_DATA_DIRS:-${DATA_ROOT}/Synthetic_Arabic_1,${DATA_ROOT}/Synthetic_Arabic_2,${DATA_ROOT}/Synthetic_Arabic_3,${DATA_ROOT}/Synthetic_Arabic_4}"
SYNTHETIC_SAMPLES_PER_DIR="${SYNTHETIC_SAMPLES_PER_DIR:-3000}"
SYNTHETIC_REQUIRE_FULL_PER_DIR="${SYNTHETIC_REQUIRE_FULL_PER_DIR:-1}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
FONT_PATH="${DIRECT_SUBWORD_FONT:-${PROJECT_DIR}/Fonts/Arslan_Wessam_B.ttf}"
ZERO_SHOT_PROFILE="${ZERO_SHOT_PROFILE:-0}"

[[ "${SYNTHETIC_SAMPLES_PER_DIR}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: SYNTHETIC_SAMPLES_PER_DIR must be a positive integer." >&2
  exit 2
}
[[ -f "${FONT_PATH}" ]] || { echo "ERROR: font not found: ${FONT_PATH}" >&2; exit 2; }
[[ "${ZERO_SHOT_PROFILE}" == "0" ]] || {
  echo "ERROR: ZERO_SHOT_PROFILE moves image geometry and invalidates fixed subword boxes." >&2
  echo "Use DIRECT_SUBWORD_BOX_SAFE_AUGMENT=1 instead." >&2
  exit 2
}

IFS=',' read -r -a SYNTHETIC_ROOTS <<< "${SYNTHETIC_DATA_DIRS}"
[[ "${#SYNTHETIC_ROOTS[@]}" -gt 0 ]] || {
  echo "ERROR: SYNTHETIC_DATA_DIRS is empty." >&2
  exit 2
}
for root in "${SYNTHETIC_ROOTS[@]}"; do
  root="${root//[[:space:]]/}"
  [[ -d "${root}/images" && -d "${root}/texts" ]] || {
    echo "ERROR: synthetic source must contain images/ and texts/: ${root}" >&2
    exit 2
  }
done

TOTAL_SAMPLES=$((SYNTHETIC_SAMPLES_PER_DIR * ${#SYNTHETIC_ROOTS[@]}))
NUM_SAMPLES="${NUM_SAMPLES:-${TOTAL_SAMPLES}}"
if (( NUM_SAMPLES != TOTAL_SAMPLES )); then
  echo "ERROR: NUM_SAMPLES must equal sources * SYNTHETIC_SAMPLES_PER_DIR (${TOTAL_SAMPLES})." >&2
  exit 2
fi

# Build or refresh only the selected 1..N sidecars in every source. Each source
# keeps its own subword_boxes directory, resolved from the source image path.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  for root in "${SYNTHETIC_ROOTS[@]}"; do
    root="${root//[[:space:]]/}"
    echo "Preparing direct-subword boxes: ${root} (1-${SYNTHETIC_SAMPLES_PER_DIR})"
    WINDOW_SIZE="${WINDOW_SIZE:-32}" \
    STRIDE_RATIO="${STRIDE_RATIO:-0.25}" \
    WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}" \
    python scripts/data/build_connected_subword_boxes_window_validated.py \
      --data-dir "${root}" \
      --font "${FONT_PATH}" \
      --font-size "${DIRECT_SUBWORD_FONT_SIZE:-90}" \
      --padding "${DIRECT_SUBWORD_PADDING:-20}" \
      --canvas-width "${LINE_WIDTH:-1024}" \
      --canvas-height "${LINE_HEIGHT:-128}" \
      --start-index 1 \
      --end-index "${SYNTHETIC_SAMPLES_PER_DIR}" \
      --snap-radius "${DIRECT_SUBWORD_SNAP_RADIUS:-8}" \
      --overlay-count "${DIRECT_SUBWORD_OVERLAY_COUNT:-5}"
  done
fi

export DATA_DIR SYNTHETIC_DATA_DIRS SYNTHETIC_SAMPLES_PER_DIR
export SYNTHETIC_REQUIRE_FULL_PER_DIR NUM_SAMPLES
export DATASET_TYPE=synthetic
export LOAD_PAIRED_LINES=1
export DIRECT_SUBWORD_SUPERVISION=1
# Leave DIRECT_SUBWORD_BOX_DIR unset so each image resolves its own sibling
# <source>/subword_boxes directory.
unset DIRECT_SUBWORD_BOX_DIR
export DIRECT_SUBWORD_STRICT_BOXES="${DIRECT_SUBWORD_STRICT_BOXES:-1}"
export DIRECT_SUBWORD_BOX_SAFE_AUGMENT="${DIRECT_SUBWORD_BOX_SAFE_AUGMENT:-1}"
export DIRECT_SUBWORD_AUGMENT_PROBABILITY="${DIRECT_SUBWORD_AUGMENT_PROBABILITY:-0.85}"
export DIRECT_SUBWORD_CLEAN_PROBABILITY="${DIRECT_SUBWORD_CLEAN_PROBABILITY:-0.15}"
export DIRECT_SUBWORD_NOISE_STD_MAX="${DIRECT_SUBWORD_NOISE_STD_MAX:-10.0}"
export DIRECT_SUBWORD_TEMPERATURE="${DIRECT_SUBWORD_TEMPERATURE:-0.07}"
export DIRECT_SUBWORD_BCE_TEMPERATURE="${DIRECT_SUBWORD_BCE_TEMPERATURE:-0.10}"
export DIRECT_SUBWORD_SIMILARITY_THRESHOLD="${DIRECT_SUBWORD_SIMILARITY_THRESHOLD:-0.20}"
export DIRECT_SUBWORD_FOCAL_GAMMA="${DIRECT_SUBWORD_FOCAL_GAMMA:-1.5}"
export DIRECT_SUBWORD_POSITIVE_BOOST="${DIRECT_SUBWORD_POSITIVE_BOOST:-2.0}"
export DIRECT_SUBWORD_REGION_WEIGHT="${DIRECT_SUBWORD_REGION_WEIGHT:-1.0}"
export DIRECT_SUBWORD_CONTEXT_REGION_WEIGHT="${DIRECT_SUBWORD_CONTEXT_REGION_WEIGHT:-0.15}"
export DIRECT_SUBWORD_LOCALIZATION_WEIGHT="${DIRECT_SUBWORD_LOCALIZATION_WEIGHT:-1.0}"
export DIRECT_SUBWORD_CONTEXT_LOCALIZATION_WEIGHT="${DIRECT_SUBWORD_CONTEXT_LOCALIZATION_WEIGHT:-0.25}"
export DIRECT_SUBWORD_ATTENTION_WEIGHT="${DIRECT_SUBWORD_ATTENTION_WEIGHT:-0.10}"
export DIRECT_SUBWORD_OUTSIDE_WEIGHT="${DIRECT_SUBWORD_OUTSIDE_WEIGHT:-0.25}"
export DIRECT_SUBWORD_OUTSIDE_MARGIN="${DIRECT_SUBWORD_OUTSIDE_MARGIN:-0.25}"
export DIRECT_SUBWORD_OUTSIDE_TOP_K="${DIRECT_SUBWORD_OUTSIDE_TOP_K:-8}"
export DIRECT_SUBWORD_OUTSIDE_MIN_INK="${DIRECT_SUBWORD_OUTSIDE_MIN_INK:-0.01}"
export DIRECT_SUBWORD_USE_INK_WEIGHTING="${DIRECT_SUBWORD_USE_INK_WEIGHTING:-1}"
export DIRECT_SUBWORD_INK_FLOOR="${DIRECT_SUBWORD_INK_FLOOR:-0.05}"
export IMAGE_TEXT_LOSS_ON_BOTH_LINES=1
export USE_IMAGE_PAIR_CONTRASTIVE=0
export USE_LOCAL_HARD_NEGATIVES=0
export NUM_NEGATIVES=0
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=0
export SPAN_DTW_BACKEND=torch
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-30}"
export ZERO_SHOT_PROFILE=0
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-direct-connected-subword-multisource}"

printf '%s\n' \
  "Direct connected-subword multi-source training" \
  "  sources             = ${SYNTHETIC_DATA_DIRS}" \
  "  samples per source  = ${SYNTHETIC_SAMPLES_PER_DIR}" \
  "  total samples       = ${NUM_SAMPLES}" \
  "  augmentation        = box-safe appearance/stroke" \
  "  alignment backend   = renderer intervals (no DTW)"

exec bash scripts/train/run_connected_subword_synthetic.sh

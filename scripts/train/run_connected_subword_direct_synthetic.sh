#!/usr/bin/env bash
# Direct connected-subword synthetic supervision: validated renderer boxes, no Span-DTW.
# Run from the repository root on the login node. The delegated launcher submits Slurm.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: JOB_ID=<name> bash scripts/train/run_connected_subword_direct_synthetic.sh" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID to the synthetic output weights-folder name.}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
FONT_PATH="${DIRECT_SUBWORD_FONT:-${PROJECT_DIR}/Fonts/Arslan_Wessam_B.ttf}"
ZERO_SHOT_PROFILE="${ZERO_SHOT_PROFILE:-0}"

[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}
[[ -f "${FONT_PATH}" ]] || { echo "ERROR: font not found: ${FONT_PATH}" >&2; exit 2; }
[[ "${ZERO_SHOT_PROFILE}" == "0" ]] || {
  echo "ERROR: ZERO_SHOT_PROFILE changes image geometry and invalidates fixed subword boxes." >&2
  exit 2
}

# Build or refresh sidecars before the ordinary launcher submits Slurm. The
# builder hashes its renderer configuration and rewrites stale sidecars.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  python scripts/data/build_connected_subword_boxes.py \
    --data-dir "${DATA_DIR}" \
    --font "${FONT_PATH}" \
    --font-size "${DIRECT_SUBWORD_FONT_SIZE:-90}" \
    --padding "${DIRECT_SUBWORD_PADDING:-20}" \
    --canvas-width "${LINE_WIDTH:-1024}" \
    --canvas-height "${LINE_HEIGHT:-128}" \
    --snap-radius "${DIRECT_SUBWORD_SNAP_RADIUS:-8}" \
    --overlay-count "${DIRECT_SUBWORD_OVERLAY_COUNT:-20}"
fi

export DATA_DIR
export DATASET_TYPE=synthetic
export LOAD_PAIRED_LINES=1
export DIRECT_SUBWORD_SUPERVISION=1
export DIRECT_SUBWORD_BOX_DIR="${DIRECT_SUBWORD_BOX_DIR:-${DATA_DIR}/subword_boxes}"
export DIRECT_SUBWORD_STRICT_BOXES="${DIRECT_SUBWORD_STRICT_BOXES:-1}"
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
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-30}"
export ZERO_SHOT_PROFILE=0
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-direct-connected-subword-synthetic}"

exec bash scripts/train/run_connected_subword_synthetic.sh

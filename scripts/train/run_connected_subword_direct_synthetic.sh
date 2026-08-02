#!/usr/bin/env bash
# Direct connected-subword synthetic supervision: renderer-derived boxes, no Span-DTW.
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

[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}
[[ -f "${FONT_PATH}" ]] || { echo "ERROR: font not found: ${FONT_PATH}" >&2; exit 2; }

# Build sidecars once on the login node before the ordinary launcher submits Slurm.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  python scripts/data/build_connected_subword_boxes.py \
    --data-dir "${DATA_DIR}" \
    --font "${FONT_PATH}" \
    --font-size "${DIRECT_SUBWORD_FONT_SIZE:-90}" \
    --padding "${DIRECT_SUBWORD_PADDING:-20}" \
    --canvas-width "${LINE_WIDTH:-1024}" \
    --canvas-height "${LINE_HEIGHT:-128}"
fi

export DATA_DIR
export DATASET_TYPE=synthetic
export DIRECT_SUBWORD_SUPERVISION=1
export DIRECT_SUBWORD_BOX_DIR="${DIRECT_SUBWORD_BOX_DIR:-${DATA_DIR}/subword_boxes}"
export DIRECT_SUBWORD_STRICT_BOXES="${DIRECT_SUBWORD_STRICT_BOXES:-1}"
export DIRECT_SUBWORD_TEMPERATURE="${DIRECT_SUBWORD_TEMPERATURE:-0.07}"
export DIRECT_SUBWORD_REGION_WEIGHT="${DIRECT_SUBWORD_REGION_WEIGHT:-1.0}"
export DIRECT_SUBWORD_LOCALIZATION_WEIGHT="${DIRECT_SUBWORD_LOCALIZATION_WEIGHT:-1.0}"
export DIRECT_SUBWORD_OUTSIDE_WEIGHT="${DIRECT_SUBWORD_OUTSIDE_WEIGHT:-0.25}"
export DIRECT_SUBWORD_OUTSIDE_MARGIN="${DIRECT_SUBWORD_OUTSIDE_MARGIN:-0.25}"
export DIRECT_SUBWORD_OUTSIDE_TOP_K="${DIRECT_SUBWORD_OUTSIDE_TOP_K:-8}"
export IMAGE_TEXT_LOSS_ON_BOTH_LINES=1
export USE_IMAGE_PAIR_CONTRASTIVE=0
export USE_LOCAL_HARD_NEGATIVES=0
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=0
export ZERO_SHOT_PROFILE="${ZERO_SHOT_PROFILE:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-direct-connected-subword-synthetic}"

exec bash scripts/train/run_connected_subword_synthetic.sh

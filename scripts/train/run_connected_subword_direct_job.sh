#!/usr/bin/env bash
# Slurm-side setup and training for direct connected-subword supervision.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID for the training checkpoint directory.}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
FONT_PATH="${DIRECT_SUBWORD_FONT:-${PROJECT_DIR}/Fonts/Arslan_Wessam_B.ttf}"

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "ERROR: run_connected_subword_direct_job.sh must run inside Slurm." >&2
  exit 2
}
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}
[[ -f "${FONT_PATH}" ]] || {
  echo "ERROR: font not found: ${FONT_PATH}" >&2
  exit 2
}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Preparing validated connected-subword sidecars inside Slurm job ${SLURM_JOB_ID}"
python scripts/data/build_connected_subword_boxes_window_validated.py \
  --data-dir "${DATA_DIR}" \
  --font "${FONT_PATH}" \
  --font-size "${DIRECT_SUBWORD_FONT_SIZE:-90}" \
  --padding "${DIRECT_SUBWORD_PADDING:-20}" \
  --canvas-width "${LINE_WIDTH:-1024}" \
  --canvas-height "${LINE_HEIGHT:-128}" \
  --snap-radius "${DIRECT_SUBWORD_SNAP_RADIUS:-8}" \
  --overlay-count "${DIRECT_SUBWORD_OVERLAY_COUNT:-20}"

echo "Sidecars are ready; starting direct connected-subword training"
exec bash scripts/train/run_connected_subword_direct_synthetic.sh

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
NUM_SAMPLES="${NUM_SAMPLES:-8000}"
PREP_WORKERS="${DIRECT_SUBWORD_PREP_WORKERS:-8}"

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
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: NUM_SAMPLES must be a positive integer." >&2
  exit 2
}
[[ "${PREP_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: DIRECT_SUBWORD_PREP_WORKERS must be a positive integer." >&2
  exit 2
}
if (( PREP_WORKERS > NUM_SAMPLES )); then
  PREP_WORKERS="${NUM_SAMPLES}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
export WINDOW_SIZE STRIDE_RATIO
export PYTHONUNBUFFERED=1

CHUNK_SIZE=$(( (NUM_SAMPLES + PREP_WORKERS - 1) / PREP_WORKERS ))
PIDS=()
RANGES=()

echo "Preparing validated connected-subword sidecars inside Slurm job ${SLURM_JOB_ID}"
echo "  samples      = 1-${NUM_SAMPLES}"
echo "  workers      = ${PREP_WORKERS}"
echo "  chunk size   = ${CHUNK_SIZE}"
echo "  window/stride= ${WINDOW_SIZE}/$(python - <<'PY'
import os
window = int(os.environ.get('WINDOW_SIZE', '32'))
ratio = float(os.environ.get('STRIDE_RATIO', '0.25'))
print(max(1, int(window * ratio)))
PY
)"

for (( worker=0; worker<PREP_WORKERS; worker++ )); do
  START_INDEX=$(( worker * CHUNK_SIZE + 1 ))
  END_INDEX=$(( START_INDEX + CHUNK_SIZE - 1 ))
  if (( START_INDEX > NUM_SAMPLES )); then
    break
  fi
  if (( END_INDEX > NUM_SAMPLES )); then
    END_INDEX="${NUM_SAMPLES}"
  fi

  OVERLAY_COUNT=0
  if (( worker == 0 )); then
    OVERLAY_COUNT="${DIRECT_SUBWORD_OVERLAY_COUNT:-20}"
  fi

  echo "  worker ${worker}: samples ${START_INDEX}-${END_INDEX}"
  python scripts/data/build_connected_subword_boxes_window_validated.py \
    --data-dir "${DATA_DIR}" \
    --font "${FONT_PATH}" \
    --font-size "${DIRECT_SUBWORD_FONT_SIZE:-90}" \
    --padding "${DIRECT_SUBWORD_PADDING:-20}" \
    --canvas-width "${LINE_WIDTH:-1024}" \
    --canvas-height "${LINE_HEIGHT:-128}" \
    --start-index "${START_INDEX}" \
    --end-index "${END_INDEX}" \
    --snap-radius "${DIRECT_SUBWORD_SNAP_RADIUS:-8}" \
    --overlay-count "${OVERLAY_COUNT}" &
  PIDS+=("$!")
  RANGES+=("${START_INDEX}-${END_INDEX}")
done

FAILED=0
for index in "${!PIDS[@]}"; do
  PID="${PIDS[$index]}"
  RANGE="${RANGES[$index]}"
  if wait "${PID}"; then
    echo "Sidecar worker completed: ${RANGE}"
  else
    STATUS=$?
    echo "ERROR: sidecar worker failed for samples ${RANGE} with exit code ${STATUS}." >&2
    FAILED=1
  fi
done

if (( FAILED != 0 )); then
  exit 1
fi

echo "Sidecars are ready; starting direct connected-subword training"
exec bash scripts/train/run_connected_subword_direct_synthetic.sh

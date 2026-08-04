#!/usr/bin/env bash
# Slurm-side setup and training for balanced direct connected-subword supervision.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID for the training checkpoint directory.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DataSet}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
SYNTHETIC_DATA_DIRS="${SYNTHETIC_DATA_DIRS:-${DATA_ROOT}/Synthetic_Arabic_1,${DATA_ROOT}/Synthetic_Arabic_2,${DATA_ROOT}/Synthetic_Arabic_3,${DATA_ROOT}/Synthetic_Arabic_4}"
SYNTHETIC_SAMPLES_PER_DIR="${SYNTHETIC_SAMPLES_PER_DIR:-3000}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
FONT_PATH="${DIRECT_SUBWORD_FONT:-${PROJECT_DIR}/Fonts/Arslan_Wessam_B.ttf}"
PREP_WORKERS="${DIRECT_SUBWORD_PREP_WORKERS:-8}"

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "ERROR: run_connected_subword_direct_job.sh must run inside Slurm." >&2
  exit 2
}
[[ -f "${FONT_PATH}" ]] || {
  echo "ERROR: font not found: ${FONT_PATH}" >&2
  exit 2
}
for name in SYNTHETIC_SAMPLES_PER_DIR PREP_WORKERS; do
  value="${!name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${name} must be a positive integer." >&2
    exit 2
  }
done

IFS=',' read -r -a SYNTHETIC_ROOTS <<< "${SYNTHETIC_DATA_DIRS}"
SOURCE_COUNT="${#SYNTHETIC_ROOTS[@]}"
(( SOURCE_COUNT > 0 )) || { echo "ERROR: SYNTHETIC_DATA_DIRS is empty." >&2; exit 2; }
for root in "${SYNTHETIC_ROOTS[@]}"; do
  root="${root//[[:space:]]/}"
  [[ -d "${root}/images" && -d "${root}/texts" ]] || {
    echo "ERROR: synthetic source must contain images/ and texts/: ${root}" >&2
    exit 2
  }
done

TOTAL_SAMPLES=$((SOURCE_COUNT * SYNTHETIC_SAMPLES_PER_DIR))
NUM_SAMPLES="${NUM_SAMPLES:-${TOTAL_SAMPLES}}"
(( NUM_SAMPLES == TOTAL_SAMPLES )) || {
  echo "ERROR: NUM_SAMPLES must equal ${TOTAL_SAMPLES}." >&2
  exit 2
}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE PYTHONUNBUFFERED=1

WORKERS_PER_SOURCE=$((PREP_WORKERS / SOURCE_COUNT))
if (( WORKERS_PER_SOURCE < 1 )); then
  WORKERS_PER_SOURCE=1
fi
if (( WORKERS_PER_SOURCE > SYNTHETIC_SAMPLES_PER_DIR )); then
  WORKERS_PER_SOURCE="${SYNTHETIC_SAMPLES_PER_DIR}"
fi
CHUNK_SIZE=$(( (SYNTHETIC_SAMPLES_PER_DIR + WORKERS_PER_SOURCE - 1) / WORKERS_PER_SOURCE ))
PIDS=()
LABELS=()

echo "Preparing validated connected-subword sidecars inside Slurm job ${SLURM_JOB_ID}"
echo "  sources             = ${SYNTHETIC_DATA_DIRS}"
echo "  samples per source  = ${SYNTHETIC_SAMPLES_PER_DIR}"
echo "  total samples       = ${TOTAL_SAMPLES}"
echo "  workers per source  = ${WORKERS_PER_SOURCE}"
echo "  window size         = ${WINDOW_SIZE}"
echo "  stride ratio        = ${STRIDE_RATIO}"

for source_index in "${!SYNTHETIC_ROOTS[@]}"; do
  root="${SYNTHETIC_ROOTS[$source_index]//[[:space:]]/}"
  source_name="$(basename "${root}")"
  for (( worker=0; worker<WORKERS_PER_SOURCE; worker++ )); do
    START_INDEX=$((worker * CHUNK_SIZE + 1))
    END_INDEX=$((START_INDEX + CHUNK_SIZE - 1))
    if (( START_INDEX > SYNTHETIC_SAMPLES_PER_DIR )); then
      break
    fi
    if (( END_INDEX > SYNTHETIC_SAMPLES_PER_DIR )); then
      END_INDEX="${SYNTHETIC_SAMPLES_PER_DIR}"
    fi

    OVERLAY_COUNT=0
    if (( worker == 0 )); then
      OVERLAY_COUNT="${DIRECT_SUBWORD_OVERLAY_COUNT:-5}"
    fi

    echo "  ${source_name} worker ${worker}: samples ${START_INDEX}-${END_INDEX}"
    python scripts/data/build_connected_subword_boxes_window_validated.py \
      --data-dir "${root}" \
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
    LABELS+=("${source_name}:${START_INDEX}-${END_INDEX}")
  done
done

FAILED=0
for index in "${!PIDS[@]}"; do
  PID="${PIDS[$index]}"
  LABEL="${LABELS[$index]}"
  if wait "${PID}"; then
    echo "Sidecar worker completed: ${LABEL}"
  else
    STATUS=$?
    echo "ERROR: sidecar worker failed for ${LABEL} with exit code ${STATUS}." >&2
    FAILED=1
  fi
done
(( FAILED == 0 )) || exit 1

echo "All source sidecars are ready; starting direct connected-subword training"
export DATA_ROOT DATA_DIR SYNTHETIC_DATA_DIRS SYNTHETIC_SAMPLES_PER_DIR NUM_SAMPLES
exec bash scripts/train/run_connected_subword_direct_synthetic.sh

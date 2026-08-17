#!/usr/bin/env bash
# Run one already-submitted SLURM stage and update the experiment tracker on start/exit.
set -uo pipefail

: "${PIPELINE_TRACKER_JSON:?PIPELINE_TRACKER_JSON is required}"
: "${PIPELINE_STAGE:?PIPELINE_STAGE is required}"
: "${TRACKED_STAGE_SCRIPT:?TRACKED_STAGE_SCRIPT is required}"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
TRACKER="${PROJECT_DIR}/scripts/pipeline/experiment_tracker.py"
JOB_ID="${SLURM_JOB_ID:-}"
JOB_NAME="${SLURM_JOB_NAME:-}"
LOG_PATH="${PIPELINE_LOG_PATH:-}"
if [[ -z "${LOG_PATH}" && -n "${JOB_ID}" && -n "${JOB_NAME}" ]]; then
  LOG_PATH="${PROJECT_DIR}/out/${JOB_NAME}_${JOB_ID}.out"
fi

python "${TRACKER}" running \
  --tracker "${PIPELINE_TRACKER_JSON}" \
  --stage "${PIPELINE_STAGE}" \
  --job-id "${JOB_ID}" \
  --log-path "${LOG_PATH}" || true

finish_tracker() {
  local rc="$1"
  python "${TRACKER}" finish \
    --tracker "${PIPELINE_TRACKER_JSON}" \
    --stage "${PIPELINE_STAGE}" \
    --exit-code "${rc}" || true
}

trap 'rc=$?; finish_tracker "$rc"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

bash "${TRACKED_STAGE_SCRIPT}"

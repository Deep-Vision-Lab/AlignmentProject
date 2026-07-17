#!/usr/bin/env bash
# Submit the full-quality compositional Arabic span-DTW training job.
#
# Run:
#   bash scripts/train/run_span_d3tw_full_quality.sh
#
# Optional overrides:
#   EPOCHS=220 bash scripts/train/run_span_d3tw_full_quality.sh
#   PRETRAINED_WEIGHTS=Weights/old_run/model_latest.pth bash scripts/train/run_span_d3tw_full_quality.sh

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_span_d3tw_full_quality.sh"
  echo "Override settings through environment variables, not command-line flags."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
mkdir -p out logs

sbatch --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  "${SCRIPT_DIR}/sbatch_span_d3tw_full_quality.sbatch"

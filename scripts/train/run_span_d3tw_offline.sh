#!/usr/bin/env bash
#
# Submit the recommended improve_neg offline span-D3TW job.
#
# The sbatch now uses train_fast_image_pair.py by default. It keeps the image-pair
# loss but avoids the slowest path by not running full Span-DTW on line2 unless
# IMAGE_TEXT_LOSS_ON_BOTH_LINES=1 is explicitly set.
#
# Default speed-oriented settings:
#   TRAIN_SCRIPT=train_fast_image_pair.py
#   IMAGE_TEXT_LOSS_ON_BOTH_LINES=0
#   IMAGE_PAIR_MAX_SAMPLES_PER_BATCH=8
#   LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES=2
#   SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0.0
#
# Override with environment variables, for example:
#   IMAGE_PAIR_MAX_SAMPLES_PER_BATCH=16 bash scripts/train/run_span_d3tw_offline.sh
#   IMAGE_TEXT_LOSS_ON_BOTH_LINES=1 bash scripts/train/run_span_d3tw_offline.sh

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_span_d3tw_offline.sh"
  echo "This wrapper does not accept flags; override settings with environment variables or edit scripts/train/sbatch_span_d3tw_win32_offline.sbatch."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
mkdir -p out logs

sbatch --export=ALL,PROJECT_DIR="${PROJECT_DIR}" "${SCRIPT_DIR}/sbatch_span_d3tw_win32_offline.sbatch"

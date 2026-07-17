#!/usr/bin/env bash
# Submit the two-GPU DDP full-quality real-data augmentation job.
#
# Run:
#   bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
#
# BATCH_SIZE is per GPU. The default 8 on two GPUs keeps global batch size 16.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh"
  echo "Override settings through environment variables, not command-line flags."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
mkdir -p out logs

sbatch --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
  "${SCRIPT_DIR}/sbatch_span_d3tw_full_quality_real_augmented_ddp.sbatch"

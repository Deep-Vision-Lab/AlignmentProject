#!/usr/bin/env bash
#
# Submit the recommended improve_neg offline span-D3TW job without passing flags.

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

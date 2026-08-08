#!/usr/bin/env bash
# Run the unaligned evaluator while keeping Slurm stdout/stderr under project/out.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${ROOT}/out}"
exec bash "${ROOT}/Evaluation/run_unaligned_evaluation_safe.sh"

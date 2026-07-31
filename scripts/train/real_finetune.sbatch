#!/usr/bin/env bash
# Compatibility shim. The real launcher must run with bash from the login node.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: do not submit this file with sbatch." >&2
  echo "Use: bash scripts/train/run_real_finetune.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_real_finetune.sh" "$@"

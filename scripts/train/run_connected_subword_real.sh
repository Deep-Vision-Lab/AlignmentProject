#!/usr/bin/env bash
# Stage 2: train on real Arabic lines, initialized only from the Stage-1 synthetic checkpoint.
# Run from the same connected-subword model branch. This script submits its own Slurm job.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: JOB_ID=<name> SYNTHETIC_WEIGHTS=<model_latest.pth> bash scripts/train/run_connected_subword_real.sh" >&2
  exit 2
fi

: "${JOB_ID:?Set JOB_ID to the real-data output weights-folder name.}"
: "${SYNTHETIC_WEIGHTS:?Set SYNTHETIC_WEIGHTS to the Stage-1 synthetic model_latest.pth.}"
[[ -f "${SYNTHETIC_WEIGHTS}" ]] || {
  echo "ERROR: synthetic checkpoint not found: ${SYNTHETIC_WEIGHTS}" >&2
  exit 2
}

export PRETRAINED_WEIGHTS="${SYNTHETIC_WEIGHTS}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-connected-subword-real}"
exec bash scripts/train/run_connected_subword_finetune.sh

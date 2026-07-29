#!/usr/bin/env bash
# Backward-compatible shortcut for the optimized ViT SLURM/DDP launcher.
# Usage: bash run_train_vit.sh [job_id] [data_dir] [pretrained_weights]
set -euo pipefail

if (( $# > 3 )); then
  echo "Usage: bash run_train_vit.sh [job_id] [data_dir] [pretrained_weights]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR

if (( $# >= 1 )) && [[ -n "$1" ]]; then
  export JOB_ID="$1"
fi
if (( $# >= 2 )) && [[ -n "$2" ]]; then
  export DATA_DIR="$2"
fi
if (( $# >= 3 )) && [[ -n "$3" ]]; then
  export PRETRAINED_WEIGHTS="$3"
fi

exec bash "${PROJECT_DIR}/scripts/train/run_vit_span_d3tw_full_quality.sh"

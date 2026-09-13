#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-$ROOT/DataSet/Synthetic63}"
JOB_NAME="${JOB_NAME:-restore_rgb_pretrain_s16}"

echo "Submitting restoration Stage A"
echo "  dataset = $DATASET"
echo "  job     = $JOB_NAME"
echo "  goal    = local window encoder must reconstruct the complete original RGB window"

exec sbatch \
  --export=ALL,DATASET="$DATASET",JOB_NAME="$JOB_NAME",RESTORATION_TRAINING_STAGE=pretrain,RESTORATION_NUM_NEGATIVES=0,RESTORATION_EPOCH_PROBE=0 \
  scripts/train_restoration_positive_dtw_2x4090.sbatch

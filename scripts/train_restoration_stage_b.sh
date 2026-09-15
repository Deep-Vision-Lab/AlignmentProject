#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-$ROOT/DataSet/Synthetic63}"
JOB_NAME="${JOB_NAME:-res18_tinyvit_dtw_s16}"

echo "Submitting ResNet-18 + ViT-Tiny letter-DTW training"
echo "  dataset = $DATASET"
echo "  job     = $JOB_NAME"
echo "  decoder = none"
echo "  loss    = positive letter-DTW + negative transcript margin-DTW"

exec sbatch   --export=ALL,DATASET="$DATASET",JOB_NAME="$JOB_NAME",RESTORATION_TRAINING_STAGE=align,RESTORATION_NUM_NEGATIVES=10   scripts/train_restoration_positive_dtw_2x4090.sbatch

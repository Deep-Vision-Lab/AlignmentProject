#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-$ROOT/DataSet/Synthetic63}"
STAGE_A_WEIGHTS="${STAGE_A_WEIGHTS:-$ROOT/Weights/restore_rgb_pretrain_s16/model_best.pth}"
JOB_NAME="${JOB_NAME:-restore_fused_rgb_dtw_s16}"

if [[ ! -f "$STAGE_A_WEIGHTS" ]]; then
  echo "ERROR: Stage-A checkpoint not found: $STAGE_A_WEIGHTS" >&2
  echo "Run first: bash scripts/train_restoration_stage_a.sh" >&2
  exit 2
fi

echo "Submitting restoration Stage B"
echo "  dataset = $DATASET"
echo "  init    = $STAGE_A_WEIGHTS"
echo "  job     = $JOB_NAME"
echo "  goal    = Transformer context + local/context fusion + positive/negative DTW"

exec sbatch \
  --export=ALL,DATASET="$DATASET",JOB_NAME="$JOB_NAME",RESTORATION_TRAINING_STAGE=align,RESTORATION_NUM_NEGATIVES=10,RESTORATION_EPOCH_PROBE=1 \
  scripts/train_restoration_positive_dtw_2x4090.sbatch \
  --weights "$STAGE_A_WEIGHTS"

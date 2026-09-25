#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TYPE="${1:-}"

case "$TYPE" in
    real)
        DATASET="$PROJECT_ROOT/DataSet/ArabicDataset"
        DATASET_TYPE="real"
        ;;

    synthetic)
        DATASET="$PROJECT_ROOT/DataSet/Synthetic63"
        DATASET_TYPE="synthetic"
        ;;

    *)
        echo "Usage:"
        echo "  bash scripts/train/train_local.sh real [train.py options]"
        echo "  bash scripts/train/train_local.sh synthetic [train.py options]"
        exit 1
        ;;
esac

shift
RUN_NAME="${TYPE}_local_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Training type : $TYPE"
echo "Dataset       : $DATASET"
echo "Run name      : $RUN_NAME"
echo "Weights       : $PROJECT_ROOT/Weights/$RUN_NAME"
echo "Python        : $(command -v python)"
echo "========================================"

cd "$PROJECT_ROOT"

exec python train.py \
    --dataset "$DATASET" \
    --dataset-type "$DATASET_TYPE" \
    --run-name "$RUN_NAME" \
    "$@"
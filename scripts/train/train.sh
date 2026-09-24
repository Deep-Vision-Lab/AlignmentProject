#!/usr/bin/env bash

set -euo pipefail

TYPE="${1:-}"

case "$TYPE" in
    real)
        DATASET="$PWD/DataSet/ArabicDataset"
        DATASET_TYPE="real"
        ;;

    synthetic)
        DATASET="$PWD/DataSet/Synthetic63"
        DATASET_TYPE="synthetic"
        ;;

    *)
        echo "Usage:"
        echo "  bash scripts/train/train.sh real"
        echo "  bash scripts/train/train.sh synthetic"
        exit 1
        ;;
esac

# Automatically create a unique experiment/job name.
RUN_NAME="${TYPE}_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "Training type : $TYPE"
echo "Dataset       : $DATASET"
echo "Run name      : $RUN_NAME"
echo "Weights       : Weights/$RUN_NAME"
echo "========================================"

sbatch \
    --job-name="$RUN_NAME" \
    scripts/train/train.sbatch \
    --dataset "$DATASET" \
    --dataset-type "$DATASET_TYPE"
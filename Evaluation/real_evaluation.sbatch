#!/usr/bin/env bash
#SBATCH --gpus=rtx_4090:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/ahmedmas/BGU-Lab/AlignmentProject/out/%x_%J.out
#SBATCH --mail-type=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
cd "$PROJECT_DIR"

mkdir -p out

eval "$(conda shell.bash hook)"
conda activate "${env:-manucripts_align}"

: "${WEIGHTS:?WEIGHTS is required}"
: "${MODEL_TAG:?MODEL_TAG is required}"
: "${RUN_TAG:?RUN_TAG is required}"

N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"

test -f "$WEIGHTS" || {
    echo "Checkpoint not found: $WEIGHTS" >&2
    exit 2
}

for LABEL in high_match medium_match; do
    OUTPUT_DIR="$PROJECT_DIR/Results/Evaluation/$MODEL_TAG/Real_Experiments/$RUN_TAG/$LABEL"

    echo "============================================================"
    echo "Real evaluation"
    echo "Model:       $MODEL_TAG"
    echo "Run:         $RUN_TAG"
    echo "Label:       $LABEL"
    echo "Samples:     $N_SAMPLES"
    echo "Checkpoint:  $WEIGHTS"
    echo "Output:      $OUTPUT_DIR"
    echo "============================================================"

    WEIGHTS="$WEIGHTS" \
    REAL_DATA_DIR="$PROJECT_DIR/DataSet/ArabicDataset" \
    REAL_SPLIT=test \
    REAL_LABELS="$LABEL" \
    START_INDEX="$START_INDEX" \
    N_SAMPLES="$N_SAMPLES" \
    FEATURE=contextual \
    SCORE_MODE=auto \
    THRESHOLD=0.45 \
    GAP=-0.30 \
    SW_INK_AWARE=1 \
    HEATMAP_SOURCE=dp-score \
    DEVICE=cuda \
    RESULTS_DIR="$OUTPUT_DIR" \
    bash Evaluation/run_real_dataset_evaluations.sh
done

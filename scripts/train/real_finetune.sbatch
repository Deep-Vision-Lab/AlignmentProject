#!/usr/bin/env bash
#SBATCH --qos=jelsana
#SBATCH --gpus=rtx_6000:1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/ahmedmas/BGU-Lab/AlignmentProject/out/%x_%J.out
#SBATCH --mail-type=ALL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
cd "$PROJECT_DIR"

mkdir -p out

eval "$(conda shell.bash hook)"
conda activate "${env:-manucripts_align}"

: "${JOB_ID:?JOB_ID is required}"
: "${PRETRAINED_WEIGHTS:?PRETRAINED_WEIGHTS is required}"
: "${AUGMENT:?AUGMENT must be 0 or 1}"

EPOCHS="${EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-32}"

if [[ "$AUGMENT" == "1" ]]; then
    AUGMENT_ARGUMENT="--augment"
elif [[ "$AUGMENT" == "0" ]]; then
    AUGMENT_ARGUMENT="--no-augment"
else
    echo "AUGMENT must be 0 or 1, got: $AUGMENT" >&2
    exit 2
fi

test -f "$PRETRAINED_WEIGHTS" || {
    echo "Checkpoint not found: $PRETRAINED_WEIGHTS" >&2
    exit 2
}

export TRAIN_SEED
export DATASET_SPLIT_SEED
export REAL_MAX_TEXT_SPAN_CHARS

# Keep all lines from the same page-pair group in one split.
export REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"

# Deterministic real-image preprocessing for train, validation and test.
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"

# Training-only real augmentation.
export REAL_AUGMENT="$AUGMENT"
export REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.85}"
export REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0.25}"
export REAL_AUG_STITCH_MAX_TEXT_CHARS="${REAL_AUG_STITCH_MAX_TEXT_CHARS:-120}"
export REAL_AUG_MORPH_PROB="${REAL_AUG_MORPH_PROB:-0.25}"
export REAL_AUG_SPECKLE_PROB="${REAL_AUG_SPECKLE_PROB:-0.12}"
export REAL_AUG_BLUR_PROB="${REAL_AUG_BLUR_PROB:-0.15}"
export REAL_AUG_NOISE_PROB="${REAL_AUG_NOISE_PROB:-0.18}"

export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-2}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-finetuning}"

echo "============================================================"
echo "Real fine-tuning"
echo "Job ID:               $JOB_ID"
echo "Branch:               $(git branch --show-current)"
echo "Commit:               $(git rev-parse HEAD)"
echo "Pretrained weights:   $PRETRAINED_WEIGHTS"
echo "Augmentation:         $AUGMENT"
echo "Epochs:               $EPOCHS"
echo "Learning rate:        $LEARNING_RATE"
echo "Training seed:        $TRAIN_SEED"
echo "Dataset split seed:   $DATASET_SPLIT_SEED"
echo "Real span limit:      $REAL_MAX_TEXT_SPAN_CHARS"
echo "============================================================"

python scripts/train/train_real_optimized.py \
    --job_id "$JOB_ID" \
    --data_dir "$PROJECT_DIR/DataSet/ArabicDataset" \
    --dataset_type real \
    "$AUGMENT_ARGUMENT" \
    --pretrained_weights "$PRETRAINED_WEIGHTS" \
    --finetune \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --window_size 32 \
    --window_overlap_mode light_overlap \
    --num_negatives 10 \
    --use_local_hard_negatives \
    --use_image_pair_contrastive

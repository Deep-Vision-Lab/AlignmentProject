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
        echo "  RUN_NAME=name FUSION_MODE=sum USE_GATED_FUSION=1 bash scripts/train/train.sh real [train.py flags]"
        echo "  bash scripts/train/train.sh synthetic [train.py flags]"
        exit 1
        ;;
esac
shift

# Fusion is configured through this wrapper's environment; negative settings
# are train.py flags. Reject lookalike environment variables that would be lost.
for setting in NEGATIVE_DTW_WEIGHT NEGATIVE_COUNT NEGATIVE_MARGIN NEGATIVE_SEVERITY \
               NEGATIVE_OPERATIONS NEGATIVE_WARMUP_EPOCHS NEGATIVE_CURRICULUM_EPOCHS \
               NEGATIVE_SEED NEGATIVE_LOSS_TYPE NEGATIVE_TARGET_MIN NEGATIVE_TARGET_MAX \
               NEGATIVE_TARGET_MODE NEGATIVE_TARGET_MIN_RATIO NEGATIVE_TARGET_MAX_RATIO \
               NEGATIVE_SOFTNESS HARD_NEGATIVE_K RANKING_AUX_WEIGHT \
               WRONG_LETTER_UNLIKELIHOOD_WEIGHT DTW_NORMALIZATION \
               STRONG_NEGATIVE_COUNT LOCAL_SUBSTITUTION_COUNT ORDER_NEGATIVE_COUNT \
               STRONG_NEGATIVE_WEIGHT WRONG_LETTER_WEIGHT ORDER_NEGATIVE_WEIGHT \
               STRONG_NEGATIVE_SEVERITY LOCAL_SUBSTITUTION_SEVERITY ORDER_MARGIN; do
    if [[ -v $setting ]]; then
        flag="${setting,,}"
        echo "$setting is not a launcher setting; pass --${flag//_/-} instead" >&2
        exit 2
    fi
done
for argument in "$@"; do
    case "${argument%%=*}" in
        --fusion-mode|--use-gated-fusion)
            echo "Set fusion through FUSION_MODE and USE_GATED_FUSION for scripts/train/train.sh" >&2
            exit 2
            ;;
    esac
done

# RUN_NAME may be supplied for a named experiment; otherwise choose a unique name.
RUN_NAME="${RUN_NAME:-${TYPE}_$(date +%Y%m%d_%H%M%S)}"
FUSION_MODE="${FUSION_MODE:-concat}"
USE_GATED_FUSION="${USE_GATED_FUSION:-0}"

if [[ "$FUSION_MODE" != concat && "$FUSION_MODE" != sum ]]; then
    echo "FUSION_MODE must be concat or sum" >&2
    exit 2
fi
if [[ "$USE_GATED_FUSION" != 0 && "$USE_GATED_FUSION" != 1 ]]; then
    echo "USE_GATED_FUSION must be 0 or 1" >&2
    exit 2
fi
if [[ "$FUSION_MODE" == concat && "$USE_GATED_FUSION" == 1 ]]; then
    echo "concat + gated fusion is unsupported; use sum + gate or concat + no gate" >&2
    exit 2
fi

export FUSION_MODE USE_GATED_FUSION

echo "========================================"
echo "Training type : $TYPE"
echo "Dataset       : $DATASET"
echo "Run name      : $RUN_NAME"
echo "Weights       : Weights/$RUN_NAME"
echo "Fusion mode   : $FUSION_MODE"
echo "Gated fusion  : $USE_GATED_FUSION"
echo "========================================"

if [[ "$(uname -s)" == "Darwin" ]]; then
    python train.py \
        --dataset "$DATASET" \
        --dataset-type "$DATASET_TYPE" \
        --run-name "$RUN_NAME" \
        --fusion-mode "$FUSION_MODE" \
        --use-gated-fusion "$USE_GATED_FUSION" \
        --device auto \
        "$@"
else
    sbatch \
        --job-name="$RUN_NAME" \
        scripts/train/train.sbatch \
        --dataset "$DATASET" \
        --dataset-type "$DATASET_TYPE" \
        "$@"
fi

#!/usr/bin/env bash
# Offline BGU SLURM presets. First argument: baseline | a | b | c.
# All four runs keep the same group split, seed, architecture and typed negatives.
set -euo pipefail
MODE="${1:-}"
case "$MODE" in
    baseline|a|b|c) ;;
    *) echo "Usage: RUN_NAME=unique [DATASET_PATH=...] [GPUS=2] bash $0 baseline|a|b|c" >&2; exit 2 ;;
esac
shift
export DATASET_PATH="${DATASET_PATH:-$PWD/DataSet/ArabicData}"
export FUSION_MODE="${FUSION_MODE:-sum}"
export USE_GATED_FUSION="${USE_GATED_FUSION:-0}"
export GPUS="${GPUS:-2}"
export RUN_NAME="${RUN_NAME:-drop_dtw_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-Weights}"
geometry=()
objective=(--alignment-loss soft_dtw)
if [[ "$MODE" == a || "$MODE" == c ]]; then
    geometry=(--preserve-aspect-ratio --variable-width-images --variable-length-windows)
fi
if [[ "$MODE" == b || "$MODE" == c ]]; then
    objective=(--alignment-loss soft_drop_dtw --enable-padding-drop --enable-space-drop
               --enable-positive-coverage-constraint --min-positive-text-coverage 0.9
               --space-detection-mode foreground_ratio --space-foreground-threshold 0.05
               --space-drop-cost 0.1 --text-drop-cost 4.0)
fi
bash scripts/train/train.sh real \
    --output-root "$OUTPUT_ROOT" --checkpoint-every 1 \
    --books "${BOOKS:-A,B,C}" \
    --split-mode group --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --split-seed 42 --seed 42 --negative-seed 42 \
    --image-height 128 --image-width 1024 --window-width 32 --window-stride 16 \
    --cnn-type resnet18 --transformer-type tiny --batch-size "${BATCH_SIZE:-8}" \
    --dtw-cost-mode full_alphabet_nll --dtw-normalization aligned_mean \
    --space-mode optional --space-skip-penalty-ratio 1.0 \
    --negative-loss-type hybrid_typed --negative-count 4 --negative-dtw-weight 0 \
    --strong-negative-count 2 --local-substitution-count 1 --order-negative-count 1 \
    --strong-negative-weight 0.5 --wrong-letter-weight 0.5 --order-negative-weight 0.25 \
    --strong-negative-severity 0.75 --local-substitution-severity 0.30 \
    --negative-target-mode uniform_ratio \
    --negative-target-min-ratio 1.0 --negative-target-max-ratio 1.3 \
    --negative-curriculum-epochs 0 \
    --negative-warmup-epochs 0 --epochs "${EPOCHS:-20}" \
    "${geometry[@]}" "${objective[@]}" "$@"

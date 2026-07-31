#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS="${WEIGHTS:-${1:-}}"
SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-DataSet/Synthetic_Arabic}"
DATASET_SIZE="${DATASET_SIZE:-8000}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-9}"
RANDOM_BASELINE_TRIALS="${RANDOM_BASELINE_TRIALS:-256}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-Results/Evaluation/Synthetic_Quantitative}"
SUBMIT_SLURM="${SUBMIT_SLURM:-1}"

if [[ -z "$WEIGHTS" ]]; then
  cat >&2 <<'USAGE'
Usage:
  WEIGHTS=Weights/<job_id>/model_latest.pth \
    bash Evaluation/run_synthetic_quantitative_evaluation.sh

DATASET_SIZE must match the sample count used during training so the exact
seeded held-out test split can be reconstructed. MAX_TEST_SAMPLES=0 evaluates
all test samples (1600 when DATASET_SIZE=8000).
USAGE
  exit 2
fi

[[ -f "$WEIGHTS" ]] || { echo "Checkpoint not found: $WEIGHTS" >&2; exit 2; }
[[ -d "$SYNTHETIC_DATA_DIR/images" ]] || {
  echo "Synthetic images not found: $SYNTHETIC_DATA_DIR/images" >&2
  exit 2
}
[[ -d "$SYNTHETIC_DATA_DIR/masks" ]] || {
  echo "Synthetic masks not found: $SYNTHETIC_DATA_DIR/masks" >&2
  exit 2
}

export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-0}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-0}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-0}"
export SYNTHETIC_BINARIZE="${SYNTHETIC_BINARIZE:-0}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"

if [[ -z "${SLURM_JOB_ID:-}" && "$SUBMIT_SLURM" == "1" ]]; then
  mkdir -p out
  submission="$({
    sbatch \
      --job-name="${SLURM_JOB_NAME:-synthetic_quant_eval}" \
      --output="$ROOT_DIR/out/%x_%J.out" \
      --chdir="$ROOT_DIR" \
      --partition="${SLURM_PARTITION:-jelsana}" \
      --gpus="${GPU_RESOURCE:-rtx_4090}:1" \
      --cpus-per-task="${CPUS_PER_TASK:-4}" \
      --mem="${SLURM_MEMORY:-32G}" \
      --time="${SLURM_TIME:-1-00:00:00}" \
      "$0"
  } 2>&1)" || {
    printf '%s\n' "$submission" >&2
    exit 1
  }
  printf '%s\n' "$submission"
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${env:-manucripts_align}"
fi

printf '%s\n' \
  "Held-out synthetic quantitative evaluation" \
  "  checkpoint       = $WEIGHTS" \
  "  data directory   = $SYNTHETIC_DATA_DIR" \
  "  dataset size     = $DATASET_SIZE" \
  "  split seed       = $SPLIT_SEED" \
  "  max test samples = $MAX_TEST_SAMPLES (0 means all)" \
  "  negatives/query  = $NEGATIVES_PER_QUERY" \
  "  feature          = $FEATURE" \
  "  output           = $RESULTS_DIR"

python -m Evaluation.synthetic_quantitative \
  --weights "$WEIGHTS" \
  --data-dir "$SYNTHETIC_DATA_DIR" \
  --dataset-size "$DATASET_SIZE" \
  --split-seed "$SPLIT_SEED" \
  --max-test-samples "$MAX_TEST_SAMPLES" \
  --feature "$FEATURE" \
  --score-mode "$SCORE_MODE" \
  --score-clip "$SCORE_CLIP" \
  --threshold "$THRESHOLD" \
  --gap "$GAP" \
  --negatives-per-query "$NEGATIVES_PER_QUERY" \
  --random-baseline-trials "$RANDOM_BASELINE_TRIALS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --device "$DEVICE" \
  --output-dir "$RESULTS_DIR"

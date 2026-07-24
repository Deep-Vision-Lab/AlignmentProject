#!/usr/bin/env bash
# Submit matched baseline and optimized one-epoch speed runs.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

BENCHMARK_SAMPLES="${BENCHMARK_SAMPLES:-1024}"
BENCHMARK_BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-16}"
BENCHMARK_GPUS="${BENCHMARK_GPUS:-2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
COMMON=(
  NUM_SAMPLES="${BENCHMARK_SAMPLES}"
  BATCH_SIZE="${BENCHMARK_BATCH_SIZE}"
  NUM_GPUS="${BENCHMARK_GPUS}"
  EPOCHS=1
  VALID_EVERY_N_EPOCHS=1
  VALID_MAX_BATCHES=1
  USE_WANDB=0
  MAX_TEXT_TOKEN_CHARS=2
  MAX_TEXT_SPAN_CHARS=2
  MAX_WINDOWS_PER_SPAN=3
  SPAN_INCLUDE_SPACE_CONTEXT=0
  SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
  SPAN_USE_BLANK_TRANSITIONS=1
  SPAN_BLANK_PENALTY=0.35
  LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES=2
  IMAGE_PAIR_EVERY_N_BATCHES=1
)

run_with_env() {
  local script="$1"
  shift
  env "${COMMON[@]}" "$@" bash "${script}"
}

echo "Submitting baseline benchmark"
run_with_env scripts/train/run_span_d3tw_full_quality.sh \
  SLURM_JOB_NAME="align_base_${STAMP}" \
  JOB_ID="benchmark_baseline_${STAMP}"

echo "Submitting optimized benchmark"
run_with_env scripts/train/run_span_d3tw_optimized.sh \
  SLURM_JOB_NAME="align_opt_${STAMP}" \
  JOB_ID="benchmark_optimized_${STAMP}" \
  GRADIENT_ACCUMULATION_STEPS=1 \
  PROFILE_TRAINING=1 \
  PROFILE_MAX_BATCHES=0 \
  ENABLE_NVTX=0 \
  FULL_CHECKPOINT_EVERY_N_EPOCHS=1 \
  MODEL_WEIGHTS_EVERY_N_EPOCHS=1

echo "Benchmark stamp: ${STAMP}"
echo "After both jobs finish, inspect:"
echo "  out/align_base_${STAMP}_*.out"
echo "  out/align_opt_${STAMP}_*.out"
echo "  logs/performance/benchmark_optimized_${STAMP}_epoch_001.json"

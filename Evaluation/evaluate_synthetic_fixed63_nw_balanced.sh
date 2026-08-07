#!/usr/bin/env bash
# Balanced Needleman-Wunsch region evaluation for the fixed-63 synthetic set.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"

export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/training_speed_optimization/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_balanced}"
export EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_trainopt_nw63_bal}"

# Reconnect short noisy breaks but keep word-sized mismatch valleys as holes.
export NW_REGION_SUPPORT_FLOOR="${NW_REGION_SUPPORT_FLOOR:-0.0}"
export NW_REGION_MAX_BRIDGE_STEPS="${NW_REGION_MAX_BRIDGE_STEPS:-2}"
export NW_REGION_BRIDGE_MEAN_FLOOR="${NW_REGION_BRIDGE_MEAN_FLOOR:--0.12}"
export NW_REGION_BRIDGE_HARD_FLOOR="${NW_REGION_BRIDGE_HARD_FLOOR:--0.35}"
export NW_REGION_FORCE_CONNECT_HOLE_STEPS="${NW_REGION_FORCE_CONNECT_HOLE_STEPS:-1}"

# Remove short, weak, ambiguous, or tiny secondary islands.
export NW_REGION_MIN_MATCH_STEPS="${NW_REGION_MIN_MATCH_STEPS:-3}"
export NW_REGION_MIN_MEAN_SCORE="${NW_REGION_MIN_MEAN_SCORE:-0.12}"
export NW_REGION_MIN_MUTUAL_Z="${NW_REGION_MIN_MUTUAL_Z:-0.15}"
export NW_REGION_MIN_SUPPORT_DENSITY="${NW_REGION_MIN_SUPPORT_DENSITY:-0.55}"
export NW_REGION_MIN_RUN_SCORE="${NW_REGION_MIN_RUN_SCORE:-1.50}"
export NW_REGION_MIN_RELATIVE_RUN_SCORE="${NW_REGION_MIN_RELATIVE_RUN_SCORE:-0.15}"

printf '%s\n' \
  "Balanced NW region settings" \
  "  reconnect: max_bridge=${NW_REGION_MAX_BRIDGE_STEPS}, force_hole=${NW_REGION_FORCE_CONNECT_HOLE_STEPS}" \
  "  bridge floors: mean=${NW_REGION_BRIDGE_MEAN_FLOOR}, hard=${NW_REGION_BRIDGE_HARD_FLOOR}" \
  "  keep run: min_steps=${NW_REGION_MIN_MATCH_STEPS}, mean_score=${NW_REGION_MIN_MEAN_SCORE}" \
  "  distinctiveness: mutual_z=${NW_REGION_MIN_MUTUAL_Z}, density=${NW_REGION_MIN_SUPPORT_DENSITY}" \
  "  run score: absolute=${NW_REGION_MIN_RUN_SCORE}, relative=${NW_REGION_MIN_RELATIVE_RUN_SCORE}" \
  "  output=${RESULTS_ROOT}"

exec bash "${ROOT}/Evaluation/evaluate_synthetic_fixed63_nw.sh"

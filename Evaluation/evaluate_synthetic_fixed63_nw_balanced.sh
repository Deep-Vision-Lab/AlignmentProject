#!/usr/bin/env bash
# Component-aware Needleman-Wunsch region evaluation for the fixed-63 synthetic set.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"

export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/training_speed_optimization/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_components}"
export EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_trainopt_nw63_cmp}"

# Strong cells seed an aligned component.  Lower-confidence neighboring cells
# may extend that component, which reconnects true words split by noisy windows.
export NW_COMPONENT_SEED_SCORE="${NW_COMPONENT_SEED_SCORE:-0.22}"
export NW_COMPONENT_SEED_MUTUAL_Z="${NW_COMPONENT_SEED_MUTUAL_Z:-0.25}"
export NW_COMPONENT_SEED_PERCENTILE="${NW_COMPONENT_SEED_PERCENTILE:-0.82}"
export NW_COMPONENT_SUPPORT_SCORE="${NW_COMPONENT_SUPPORT_SCORE:-0.04}"
export NW_COMPONENT_SUPPORT_MUTUAL_Z="${NW_COMPONENT_SUPPORT_MUTUAL_Z:--0.10}"
export NW_COMPONENT_SUPPORT_PERCENTILE="${NW_COMPONENT_SUPPORT_PERCENTILE:-0.62}"

# Reconnect nearby pieces of the same component, but leave word-sized mismatch
# regions disconnected.
export NW_COMPONENT_MAX_PATH_GAP="${NW_COMPONENT_MAX_PATH_GAP:-3}"
export NW_COMPONENT_MAX_WINDOW_GAP="${NW_COMPONENT_MAX_WINDOW_GAP:-2}"
export NW_COMPONENT_MERGE_PATH_GAP="${NW_COMPONENT_MERGE_PATH_GAP:-3}"
export NW_COMPONENT_MERGE_WINDOW_GAP="${NW_COMPONENT_MERGE_WINDOW_GAP:-2}"

# Remove weak/ambiguous islands.  The synthetic generator contains at most
# three true shared components, so evaluation never keeps more than three.
export NW_COMPONENT_MIN_MATCHES="${NW_COMPONENT_MIN_MATCHES:-4}"
export NW_COMPONENT_MIN_SEEDS="${NW_COMPONENT_MIN_SEEDS:-2}"
export NW_COMPONENT_MIN_MEAN_SCORE="${NW_COMPONENT_MIN_MEAN_SCORE:-0.12}"
export NW_COMPONENT_MIN_MEAN_MUTUAL_Z="${NW_COMPONENT_MIN_MEAN_MUTUAL_Z:-0.10}"
export NW_COMPONENT_MIN_MEAN_PERCENTILE="${NW_COMPONENT_MIN_MEAN_PERCENTILE:-0.72}"
export NW_COMPONENT_MIN_DENSITY="${NW_COMPONENT_MIN_DENSITY:-0.50}"
export NW_COMPONENT_MIN_SPAN_BALANCE="${NW_COMPONENT_MIN_SPAN_BALANCE:-0.55}"
export NW_COMPONENT_MIN_QUALITY="${NW_COMPONENT_MIN_QUALITY:-1.25}"
export NW_COMPONENT_MIN_RELATIVE_QUALITY="${NW_COMPONENT_MIN_RELATIVE_QUALITY:-0.35}"
export NW_COMPONENT_MAX_COMPONENTS="${NW_COMPONENT_MAX_COMPONENTS:-3}"

printf '%s\n' \
  "Component-aware NW region settings" \
  "  seeds: score=${NW_COMPONENT_SEED_SCORE}, z=${NW_COMPONENT_SEED_MUTUAL_Z}, pct=${NW_COMPONENT_SEED_PERCENTILE}" \
  "  support: score=${NW_COMPONENT_SUPPORT_SCORE}, z=${NW_COMPONENT_SUPPORT_MUTUAL_Z}, pct=${NW_COMPONENT_SUPPORT_PERCENTILE}" \
  "  reconnect: path_gap=${NW_COMPONENT_MAX_PATH_GAP}, window_gap=${NW_COMPONENT_MAX_WINDOW_GAP}" \
  "  keep: matches=${NW_COMPONENT_MIN_MATCHES}, seeds=${NW_COMPONENT_MIN_SEEDS}, density=${NW_COMPONENT_MIN_DENSITY}" \
  "  confidence: mean_score=${NW_COMPONENT_MIN_MEAN_SCORE}, mean_z=${NW_COMPONENT_MIN_MEAN_MUTUAL_Z}, mean_pct=${NW_COMPONENT_MIN_MEAN_PERCENTILE}" \
  "  quality: absolute=${NW_COMPONENT_MIN_QUALITY}, relative=${NW_COMPONENT_MIN_RELATIVE_QUALITY}, max_components=${NW_COMPONENT_MAX_COMPONENTS}" \
  "  output=${RESULTS_ROOT}"

exec bash "${ROOT}/Evaluation/evaluate_synthetic_fixed63_nw.sh"

#!/usr/bin/env bash
# Component-aware Needleman-Wunsch region evaluation for the fixed-63 synthetic set.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"

export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/training_speed_optimization/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_components_v3_local}"
export EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_trainopt_nw63_cmp3}"

# Strong cells seed an aligned component. Lower-confidence neighboring cells
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

# Reject tiny high-cosine islands. The generator's smallest true shared phrase
# is still a sustained fragment, so a 4-6-window accidental ridge is not enough.
export NW_COMPONENT_MIN_MATCHES="${NW_COMPONENT_MIN_MATCHES:-7}"
export NW_COMPONENT_MIN_SPAN_WINDOWS="${NW_COMPONENT_MIN_SPAN_WINDOWS:-7}"
export NW_COMPONENT_MIN_SPAN_FRACTION="${NW_COMPONENT_MIN_SPAN_FRACTION:-0.13}"

# Remove weak/ambiguous components. The synthetic generator contains at most
# three true shared components, so evaluation never keeps more than three.
export NW_COMPONENT_MIN_SEEDS="${NW_COMPONENT_MIN_SEEDS:-2}"
export NW_COMPONENT_MIN_MEAN_SCORE="${NW_COMPONENT_MIN_MEAN_SCORE:-0.12}"
export NW_COMPONENT_MIN_MEAN_MUTUAL_Z="${NW_COMPONENT_MIN_MEAN_MUTUAL_Z:-0.10}"
export NW_COMPONENT_MIN_MEAN_PERCENTILE="${NW_COMPONENT_MIN_MEAN_PERCENTILE:-0.72}"
export NW_COMPONENT_MIN_DENSITY="${NW_COMPONENT_MIN_DENSITY:-0.50}"
export NW_COMPONENT_MIN_SPAN_BALANCE="${NW_COMPONENT_MIN_SPAN_BALANCE:-0.55}"
export NW_COMPONENT_MIN_QUALITY="${NW_COMPONENT_MIN_QUALITY:-1.25}"
export NW_COMPONENT_MIN_RELATIVE_QUALITY="${NW_COMPONENT_MIN_RELATIVE_QUALITY:-0.35}"
export NW_COMPONENT_MAX_COMPONENTS="${NW_COMPONENT_MAX_COMPONENTS:-3}"

# Local-alignment policy: do not erase a credible local phrase merely because
# full-line global NW is negative. The synthetic set intentionally surrounds
# shared phrases with unrelated context. Tiny accidental matches are still
# rejected by the minimum size, density, distinctiveness, and quality filters.
export NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--1000000.0}"
export NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE="${NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE:-0.16}"

printf '%s\n' \
  "Component-aware NW region settings v3 local" \
  "  seeds: score=${NW_COMPONENT_SEED_SCORE}, z=${NW_COMPONENT_SEED_MUTUAL_Z}, pct=${NW_COMPONENT_SEED_PERCENTILE}" \
  "  support: score=${NW_COMPONENT_SUPPORT_SCORE}, z=${NW_COMPONENT_SUPPORT_MUTUAL_Z}, pct=${NW_COMPONENT_SUPPORT_PERCENTILE}" \
  "  reconnect: path_gap=${NW_COMPONENT_MAX_PATH_GAP}, window_gap=${NW_COMPONENT_MAX_WINDOW_GAP}" \
  "  minimum size: matches=${NW_COMPONENT_MIN_MATCHES}, span_windows=${NW_COMPONENT_MIN_SPAN_WINDOWS}, span_fraction=${NW_COMPONENT_MIN_SPAN_FRACTION}" \
  "  confidence: seeds=${NW_COMPONENT_MIN_SEEDS}, mean_score=${NW_COMPONENT_MIN_MEAN_SCORE}, mean_z=${NW_COMPONENT_MIN_MEAN_MUTUAL_Z}, mean_pct=${NW_COMPONENT_MIN_MEAN_PERCENTILE}" \
  "  quality: density=${NW_COMPONENT_MIN_DENSITY}, absolute=${NW_COMPONENT_MIN_QUALITY}, relative=${NW_COMPONENT_MIN_RELATIVE_QUALITY}, max_components=${NW_COMPONENT_MAX_COMPONENTS}" \
  "  global veto: disabled; credible local components are retained" \
  "  output=${RESULTS_ROOT}"

exec bash "${ROOT}/Evaluation/evaluate_synthetic_fixed63_nw.sh"

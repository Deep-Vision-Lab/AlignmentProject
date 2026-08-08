#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export EVAL_MODE=nw
export RUN_ID="${RUN_ID:-vit_augmented_fixed63_27k}"
export N_SAMPLES="${N_SAMPLES:-20}"

# Do not reject a locally strong aligned phrase because the global NW score is
# negative. The synthetic pairs intentionally contain unrelated context around
# one to three shared regions, so a negative full-line score is compatible with
# a real local alignment. Tiny accidental islands are still rejected by the
# component-v2 minimum match/span/confidence filters.
export NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--1000000.0}"

export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/use_vit_encoder/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_components_v3_local}"
exec bash "${ROOT}/Evaluation/run_fixed63_vit_eval.sh"

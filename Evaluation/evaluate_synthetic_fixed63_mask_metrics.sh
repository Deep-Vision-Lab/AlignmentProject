#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export EVAL_MODE=metrics
export RUN_ID="${RUN_ID:-vit_augmented_fixed63_27k}"
export N_SAMPLES="${N_SAMPLES:-0}"
export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/use_vit_encoder/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_mask_metrics}"
exec bash "${ROOT}/Evaluation/run_fixed63_vit_eval.sh"

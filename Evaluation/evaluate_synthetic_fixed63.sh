#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export EVAL_MODE=sw
export RUN_ID="${RUN_ID:-vit_augmented_fixed63_27k}"
export N_SAMPLES="${N_SAMPLES:-20}"
export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/use_vit_encoder/Synthetic_Experiments/${RUN_ID}/SmithWaterman}"
exec bash "${ROOT}/Evaluation/run_fixed63_vit_eval.sh"

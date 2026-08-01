#!/usr/bin/env bash
# Qualitative/per-pair evaluator for connected-subword checkpoints.
set -euo pipefail

export SPAN_TOKENIZATION_MODE="${SPAN_TOKENIZATION_MODE:-connected_subword}"
export SPAN_USE_BLANK_TRANSITIONS="${SPAN_USE_BLANK_TRANSITIONS:-1}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-16}"
export SPAN_CONNECTED_WINDOWS_PER_CHAR="${SPAN_CONNECTED_WINDOWS_PER_CHAR:-3}"
export SPAN_CONNECTED_EXTRA_WINDOWS="${SPAN_CONNECTED_EXTRA_WINDOWS:-1}"
export SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS="${SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS:-2}"
export SPAN_SPACE_MAX_WINDOWS="${SPAN_SPACE_MAX_WINDOWS:-3}"
export EVAL_PYTHON_MODULE=Evaluation.eval_connected_subword

# The canonical launcher does not expose its Python module as a variable, so use
# a temporary copy with the one module name changed. All Slurm/resource settings
# and output conventions remain identical.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TEMP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/evaluate_connected.XXXXXX.sh")"
trap 'rm -f "${TEMP_SCRIPT}"' EXIT
sed 's/python -m Evaluation\.eval_img_align_sw/python -m Evaluation.eval_connected_subword/' \
  "${SCRIPT_DIR}/evaluate.sh" > "${TEMP_SCRIPT}"
chmod +x "${TEMP_SCRIPT}"
bash "${TEMP_SCRIPT}"

#!/usr/bin/env bash
# Transcript-supervised evaluator for connected-subword checkpoints.
set -euo pipefail

export SPAN_TOKENIZATION_MODE="${SPAN_TOKENIZATION_MODE:-connected_subword}"
export SPAN_USE_BLANK_TRANSITIONS="${SPAN_USE_BLANK_TRANSITIONS:-1}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-16}"
export SPAN_CONNECTED_WINDOWS_PER_CHAR="${SPAN_CONNECTED_WINDOWS_PER_CHAR:-3}"
export SPAN_CONNECTED_EXTRA_WINDOWS="${SPAN_CONNECTED_EXTRA_WINDOWS:-1}"
export SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS="${SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS:-2}"
export SPAN_SPACE_MAX_WINDOWS="${SPAN_SPACE_MAX_WINDOWS:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TEMP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/transcript_connected.XXXXXX.sh")"
trap 'rm -f "${TEMP_SCRIPT}"' EXIT
sed 's/python -m Evaluation\.transcript_quantitative/python -m Evaluation.transcript_quantitative_connected/' \
  "${SCRIPT_DIR}/evaluate_transcript_quantitative.sh" > "${TEMP_SCRIPT}"
chmod +x "${TEMP_SCRIPT}"
bash "${TEMP_SCRIPT}"

#!/usr/bin/env bash
# Pinned treatment/baseline-comparable real NW protocol.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FEATURE=local
export SCORE_MODE=mutual-z
export EVAL_WINDOW_STRIDE=8

exec bash "${SCRIPT_DIR}/evaluate_nw_real.sh"

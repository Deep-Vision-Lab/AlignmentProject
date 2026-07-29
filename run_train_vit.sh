#!/usr/bin/env bash
# Backward-compatible shortcut. The optimized launcher lives under scripts/train.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR
exec bash "${PROJECT_DIR}/scripts/train/run_vit_span_d3tw_full_quality.sh" "$@"

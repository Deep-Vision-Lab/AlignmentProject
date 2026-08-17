#!/usr/bin/env bash
# Public bridge-training command. Architecture is selected by model_backend.py.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${PROJECT_DIR}/scripts/train/run_branch_real_synthetic_bridge.sh"

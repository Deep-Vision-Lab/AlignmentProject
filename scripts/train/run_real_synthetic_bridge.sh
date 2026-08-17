#!/usr/bin/env bash
# Public bridge-training command. Architecture is selected by model_backend.py.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
export JOB_ID="${JOB_ID:-$(git -C "${PROJECT_DIR}" branch --show-current | tr '/' '-')-real-synthetic-bridge-v2}"
exec bash "${PROJECT_DIR}/scripts/train/run_branch_real_synthetic_bridge.sh"

#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}"
exec bash "${PROJECT_DIR}/scripts/slurm/submit_full_research_pipeline.sh"

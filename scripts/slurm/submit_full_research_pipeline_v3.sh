#!/usr/bin/env bash
# Canonical model pipeline using the frozen RealSyntheticBridge V3 dataset.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
[[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && -s "${BRIDGE_DATA_DIR}/metadata.json" ]] || {
  echo "ERROR: Bridge V3 does not exist: ${BRIDGE_DATA_DIR}" >&2
  echo "Run first: bash scripts/slurm/submit_bridge_v3_dataset.sh" >&2
  exit 2
}
python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}"
exec bash "${PROJECT_DIR}/scripts/slurm/submit_full_research_pipeline.sh"

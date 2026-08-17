#!/usr/bin/env bash
# Canonical model pipeline using the frozen RealSyntheticBridge V3 dataset.
# Keep the dependency implementation in one place, but transform the historical V2
# labels/paths into V3 in a temporary script created beside the canonical submitter.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
[[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && -s "${BRIDGE_DATA_DIR}/metadata.json" ]] || {
  echo "ERROR: Bridge V3 does not exist: ${BRIDGE_DATA_DIR}" >&2
  echo "Run first: bash scripts/slurm/submit_bridge_v3_dataset.sh" >&2
  exit 2
}
python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}"

SOURCE="${PROJECT_DIR}/scripts/slurm/submit_full_research_pipeline.sh"
TMP="${PROJECT_DIR}/scripts/slurm/.submit_full_research_pipeline_v3_runtime_$$.sh"
trap 'rm -f "${TMP}"' EXIT
sed \
  -e 's/RealSyntheticBridge V2/RealSyntheticBridge V3/g' \
  -e 's/RealSyntheticBridge_v2/RealSyntheticBridge_v3/g' \
  -e 's/Bridge V2/Bridge V3/g' \
  -e 's/bridge_v2/bridge_v3/g' \
  -e 's/submit_bridge_v2_dataset\.sh/submit_bridge_v3_dataset.sh/g' \
  -e 's/smoke_test_real_synthetic_bridge\.py/smoke_test_real_synthetic_bridge_v3.py/g' \
  "${SOURCE}" > "${TMP}"
chmod +x "${TMP}"
bash "${TMP}"

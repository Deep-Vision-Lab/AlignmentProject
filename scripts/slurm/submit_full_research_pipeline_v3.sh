#!/usr/bin/env bash
# Canonical model pipeline using RealSyntheticBridge V3.
# When DEFER_BRIDGE_VALIDATION=1, D0 and S1 run independently. This V3 wrapper also
# removes artificial ordering between evaluation-only stages so real Bridge training
# can overlap the synthetic evaluations without violating weight/data dependencies.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"

if [[ "${DEFER_BRIDGE_VALIDATION:-0}" != "1" ]]; then
  [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && -s "${BRIDGE_DATA_DIR}/metadata.json" ]] || {
    echo "ERROR: Bridge V3 does not exist: ${BRIDGE_DATA_DIR}" >&2
    echo "Run first: bash scripts/slurm/submit_bridge_v3_dataset.sh" >&2
    exit 2
  }
  python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}"
else
  : "${BRIDGE_READY_JOB_ID:?DEFER_BRIDGE_VALIDATION=1 requires BRIDGE_READY_JOB_ID}"
  echo "Bridge V3 validation deferred until dependency job ${BRIDGE_READY_JOB_ID} completes."
  echo "S1 starts independently; Bridge-dependent work waits for D0."
fi

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
  -e 's/J3=$(submit_gpu_eval S3 "$J2"/J3=$(submit_gpu_eval S3 "$J1"/' \
  -e 's/J4_DEP="$J3"/J4_DEP="$J1"/' \
  -e 's/J4_DEP="${J3}:${BRIDGE_READY_JOB_ID}"/J4_DEP="${J1}:${BRIDGE_READY_JOB_ID}"/' \
  -e 's/J7=$(submit_gpu_eval S7A "$J6"/J7=$(submit_gpu_eval S7A "$J5"/' \
  -e 's/J8=$(submit_gpu_eval S7B "$J7"/J8=$(submit_gpu_eval S7B "$J5"/' \
  -e 's/J9=$(submit_gpu_eval S8 "$J8"/J9=$(submit_gpu_eval S8 "${J2}:${J3}:${J6}:${J7}:${J8}"/' \
  -e 's/S1 -> S2 -> S3/S1 -> S2 and S1 -> S3 (parallel evaluations)/' \
  -e 's/S3 + D0(/S1 + D0(/' \
  -e 's/S4 -> S5 -> S6 -> S7A -> S7B -> S8/S4 -> S5 -> {S6,S7A,S7B}; all evaluation branches -> S8/' \
  "${SOURCE}" > "${TMP}"
chmod +x "${TMP}"

echo "V3 dependency policy: D0 || S1; after S1, S2 || S3; after D0+S1, S4->S5; after S5, S6 || S7A || S7B; S8 waits for all evaluations."
bash "${TMP}"

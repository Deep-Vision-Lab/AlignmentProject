#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR is required}"; EXPECTED_BRANCH="${EXPECTED_BRANCH:?EXPECTED_BRANCH is required}"; EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"; cd "${PROJECT_DIR}"; export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "ERROR: branch changed while waiting" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo "ERROR: commit changed while waiting" >&2; exit 2; }
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"; [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing Bridge V3 manifest" >&2; exit 2; }
python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}" --expected-negatives "${NEGATIVES_PER_ANCHOR:-8}"
python scripts/data/validate_bridge_v3_font_size.py --data-dir "${BRIDGE_DATA_DIR}" --min-font-size "${MIN_FONT_SIZE:-42}"
python scripts/data/validate_bridge_v3_dense_layout.py --data-dir "${BRIDGE_DATA_DIR}" --min-recorded-fill "${MIN_LINE_FILL_RATIO:-0.90}" --min-pixel-span 0.84 --expected-negatives "${NEGATIVES_PER_ANCHOR:-8}"
python scripts/data/validate_bridge_v3_real_augmentation.py --data-dir "${BRIDGE_DATA_DIR}"
RUN_PREFIX="${RUN_PREFIX:?RUN_PREFIX is required}" SYNTH_EPOCHS="${SYNTH_EPOCHS:-20}" BRIDGE_EPOCHS="${BRIDGE_EPOCHS:-25}" BRIDGE_LR="${BRIDGE_LR:-1e-6}" FINAL_THRESHOLD="${FINAL_THRESHOLD:-0.50}" BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR}" bash scripts/slurm/submit_full_research_pipeline_v3.sh

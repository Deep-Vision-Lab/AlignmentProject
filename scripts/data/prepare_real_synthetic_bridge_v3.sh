#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_DIR}"; export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"; REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"; REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"; CONDA_ENV="${CONDA_ENV:-manucripts_align}"
EXPECTED_NEGATIVES="${NEGATIVES_PER_ANCHOR:-8}"; MIN_FONT_SIZE="${MIN_FONT_SIZE:-42}"; MIN_LINE_FILL_RATIO="${MIN_LINE_FILL_RATIO:-0.90}"
if command -v conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"; fi
if [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && "${REBUILD_BRIDGE}" != "1" ]]; then
  if python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}" --expected-negatives "${EXPECTED_NEGATIVES}" \
    && python scripts/data/validate_bridge_v3_font_size.py --data-dir "${BRIDGE_DATA_DIR}" --min-font-size "${MIN_FONT_SIZE}" \
    && python scripts/data/validate_bridge_v3_dense_layout.py --data-dir "${BRIDGE_DATA_DIR}" --min-recorded-fill "${MIN_LINE_FILL_RATIO}" --min-pixel-span 0.84 --expected-negatives "${EXPECTED_NEGATIVES}"; then
      echo "Existing dense Bridge V3 is valid; reusing ${BRIDGE_DATA_DIR}"; exit 0
  fi
fi
DATA_DIR="${REAL_DATA_DIR}" OUTPUT_DIR="${BRIDGE_DATA_DIR}" OVERWRITE=1 MAX_ANCHORS="${MAX_ANCHORS:-0}" NEGATIVES_PER_ANCHOR="${EXPECTED_NEGATIVES}" MIN_FONT_SIZE="${MIN_FONT_SIZE}" MIN_LINE_FILL_RATIO="${MIN_LINE_FILL_RATIO}" bash scripts/data/build_real_conditioned_synthetic_bridge.sh

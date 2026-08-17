#!/usr/bin/env bash
# Idempotent CPU preparation for RealSyntheticBridge V2.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"

if [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && -s "${BRIDGE_DATA_DIR}/metadata.json" && "${REBUILD_BRIDGE}" != "1" ]]; then
  echo "Bridge V2 already exists; validating and reusing ${BRIDGE_DATA_DIR}"
  python scripts/data/smoke_test_real_synthetic_bridge.py --data-dir "${BRIDGE_DATA_DIR}"
  exit 0
fi

echo "Building Bridge V2 at ${BRIDGE_DATA_DIR}"
DATA_DIR="${REAL_DATA_DIR}" \
OUTPUT_DIR="${BRIDGE_DATA_DIR}" \
OVERWRITE=1 \
MAX_ANCHORS="${MAX_ANCHORS:-0}" \
NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-4}" \
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}" \
MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}" \
SEED="${SEED:-42}" \
bash scripts/data/build_real_conditioned_synthetic_bridge.sh

#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_DIR}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"; CONDA_ENV="${CONDA_ENV:-manucripts_align}"
if command -v conda >/dev/null 2>&1; then source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"; fi
CURRENT_VERSION=0
if [[ -s "${BRIDGE_DATA_DIR}/metadata.json" ]]; then CURRENT_VERSION="$(python - "${BRIDGE_DATA_DIR}/metadata.json" <<'PY'
import json,sys
try: print(int(json.load(open(sys.argv[1],encoding='utf-8')).get('dataset_version',0)))
except Exception: print(0)
PY
)"; fi
if [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && "${CURRENT_VERSION}" == "3" && "${REBUILD_BRIDGE}" != "1" ]]; then
  echo "Bridge V3 already exists; organizing/validating and reusing ${BRIDGE_DATA_DIR}"
  python scripts/data/organize_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}"
  python scripts/data/smoke_test_real_synthetic_bridge_v3.py --data-dir "${BRIDGE_DATA_DIR}"
  exit 0
fi
DATA_DIR="${REAL_DATA_DIR}" OUTPUT_DIR="${BRIDGE_DATA_DIR}" OVERWRITE=1 MAX_ANCHORS="${MAX_ANCHORS:-0}" NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-4}" NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}" MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}" MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}" SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-5}" SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-12}" MAX_SENTENCE_CHARS="${MAX_SENTENCE_CHARS:-100}" MAX_FONT_CHUNK_WORDS="${MAX_FONT_CHUNK_WORDS:-3}" BLUR_PROB="${BLUR_PROB:-0.65}" BLUR_MAX_RADIUS="${BLUR_MAX_RADIUS:-1.15}" NOISE_PROB="${NOISE_PROB:-0.80}" NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-9.0}" CONTRAST_MIN="${CONTRAST_MIN:-0.88}" CONTRAST_MAX="${CONTRAST_MAX:-1.14}" BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.90}" BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.08}" SEED="${SEED:-42}" bash scripts/data/build_real_conditioned_synthetic_bridge.sh

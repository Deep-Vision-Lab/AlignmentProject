#!/usr/bin/env bash
# Idempotent CPU preparation for the current RealSyntheticBridge V3 revision.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
EXPECTED_REVISION="v3.2-glyphsafe-fullwidth-8neg"
EXPECTED_NEGATIVES="${NEGATIVES_PER_ANCHOR:-8}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

CURRENT_REVISION=""
CURRENT_NEGATIVES=0
CURRENT_LAYOUT=0
if [[ -s "${BRIDGE_DATA_DIR}/metadata.json" ]]; then
  read -r CURRENT_REVISION CURRENT_NEGATIVES CURRENT_LAYOUT < <(python - "${BRIDGE_DATA_DIR}/metadata.json" <<'PY'
import json,sys
try:
    m=json.load(open(sys.argv[1],encoding='utf-8'))
    print(m.get('dataset_revision',''), int(m.get('negatives_per_anchor',0)), int(m.get('layout_version',0)))
except Exception:
    print('',0,0)
PY
)
fi

if [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" \
      && "${CURRENT_REVISION}" == "${EXPECTED_REVISION}" \
      && "${CURRENT_NEGATIVES}" == "${EXPECTED_NEGATIVES}" \
      && "${CURRENT_LAYOUT}" == "2" \
      && "${REBUILD_BRIDGE}" != "1" ]]; then
  echo "Current Bridge V3 already exists; validating and reusing ${BRIDGE_DATA_DIR}"
  python scripts/data/smoke_test_real_synthetic_bridge_v3.py \
    --data-dir "${BRIDGE_DATA_DIR}" --expected-negatives "${EXPECTED_NEGATIVES}"
  exit 0
fi

if [[ -d "${BRIDGE_DATA_DIR}" ]]; then
  echo "Existing Bridge directory is stale/incompatible and will be rebuilt."
  echo "current_revision=${CURRENT_REVISION:-<none>} expected_revision=${EXPECTED_REVISION}"
  echo "current_negatives=${CURRENT_NEGATIVES} expected_negatives=${EXPECTED_NEGATIVES}"
  echo "current_layout=${CURRENT_LAYOUT} expected_layout=2"
fi

echo "Building Bridge V3 at ${BRIDGE_DATA_DIR}"
DATA_DIR="${REAL_DATA_DIR}" OUTPUT_DIR="${BRIDGE_DATA_DIR}" OVERWRITE=1 \
MAX_ANCHORS="${MAX_ANCHORS:-0}" NEGATIVES_PER_ANCHOR="${EXPECTED_NEGATIVES}" \
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}" MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}" \
MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}" SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-8}" \
SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-16}" MIN_SENTENCE_CHARS="${MIN_SENTENCE_CHARS:-36}" \
MAX_SENTENCE_CHARS="${MAX_SENTENCE_CHARS:-130}" MAX_FONT_CHUNK_WORDS="${MAX_FONT_CHUNK_WORDS:-1}" \
MIN_LINE_FILL_RATIO="${MIN_LINE_FILL_RATIO:-0.75}" MAX_FONT_SIZE="${MAX_FONT_SIZE:-88}" \
BLUR_PROB="${BLUR_PROB:-0.65}" BLUR_MAX_RADIUS="${BLUR_MAX_RADIUS:-1.15}" \
NOISE_PROB="${NOISE_PROB:-0.80}" NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-9.0}" \
CONTRAST_MIN="${CONTRAST_MIN:-0.88}" CONTRAST_MAX="${CONTRAST_MAX:-1.14}" \
BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.90}" BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.08}" \
SEED="${SEED:-42}" \
bash scripts/data/build_real_conditioned_synthetic_bridge.sh

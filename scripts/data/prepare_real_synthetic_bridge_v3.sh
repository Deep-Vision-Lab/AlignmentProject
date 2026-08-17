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
MIN_FONT_SIZE="${MIN_FONT_SIZE:-42}"

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
  echo "Bridge V3 exists; checking current scientific + readable-font policy before reuse."
  if python scripts/data/smoke_test_real_synthetic_bridge_v3.py \
       --data-dir "${BRIDGE_DATA_DIR}" --expected-negatives "${EXPECTED_NEGATIVES}" \
     && python scripts/data/validate_bridge_v3_font_size.py \
       --data-dir "${BRIDGE_DATA_DIR}" --min-font-size "${MIN_FONT_SIZE}"; then
    echo "Existing Bridge V3 passes readable-font policy; reusing ${BRIDGE_DATA_DIR}"
    exit 0
  fi
  echo "Existing Bridge V3 uses stale/tiny rendering; rebuilding it."
fi

if [[ -d "${BRIDGE_DATA_DIR}" ]]; then
  echo "Existing Bridge directory will be rebuilt."
  echo "current_revision=${CURRENT_REVISION:-<none>} expected_revision=${EXPECTED_REVISION}"
  echo "current_negatives=${CURRENT_NEGATIVES} expected_negatives=${EXPECTED_NEGATIVES}"
  echo "current_layout=${CURRENT_LAYOUT} expected_layout=2"
  echo "required_min_font_size=${MIN_FONT_SIZE}"
fi

echo "Building readable Bridge V3 at ${BRIDGE_DATA_DIR}"
DATA_DIR="${REAL_DATA_DIR}" OUTPUT_DIR="${BRIDGE_DATA_DIR}" OVERWRITE=1 \
MAX_ANCHORS="${MAX_ANCHORS:-0}" NEGATIVES_PER_ANCHOR="${EXPECTED_NEGATIVES}" \
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}" MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}" \
MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}" MIN_POSITIVE_CHARS="${MIN_POSITIVE_CHARS:-4}" \
MAX_PHRASE_CHARS="${MAX_PHRASE_CHARS:-8}" MAX_PHRASE_WORDS="${MAX_PHRASE_WORDS:-2}" \
SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-5}" SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-7}" \
MIN_SENTENCE_CHARS="${MIN_SENTENCE_CHARS:-22}" MAX_SENTENCE_CHARS="${MAX_SENTENCE_CHARS:-40}" \
MAX_FONT_CHUNK_WORDS="${MAX_FONT_CHUNK_WORDS:-2}" FONT_SIZE="${FONT_SIZE:-60}" \
MIN_FONT_SIZE="${MIN_FONT_SIZE}" MAX_FONT_SIZE="${MAX_FONT_SIZE:-74}" \
MIN_LINE_FILL_RATIO="${MIN_LINE_FILL_RATIO:-0.65}" PADDING="${PADDING:-8}" \
SEGMENT_GAP_MIN_PX="${SEGMENT_GAP_MIN_PX:-3}" SEGMENT_GAP_MAX_PX="${SEGMENT_GAP_MAX_PX:-7}" \
BLUR_PROB="${BLUR_PROB:-0.65}" BLUR_MAX_RADIUS="${BLUR_MAX_RADIUS:-1.15}" \
NOISE_PROB="${NOISE_PROB:-0.80}" NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-9.0}" \
CONTRAST_MIN="${CONTRAST_MIN:-0.88}" CONTRAST_MAX="${CONTRAST_MAX:-1.14}" \
BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.90}" BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.08}" \
SEED="${SEED:-42}" \
bash scripts/data/build_real_conditioned_synthetic_bridge.sh

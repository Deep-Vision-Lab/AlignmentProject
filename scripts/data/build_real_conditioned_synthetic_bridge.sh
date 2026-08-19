#!/usr/bin/env bash
# Build RealSyntheticBridge V3 once, offline on CPU.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-8}"
NEGATIVE_NGRAM="${NEGATIVE_NGRAM:-3}"
MIN_OVERLAP_WORD_CHARS="${MIN_OVERLAP_WORD_CHARS:-1}"
MAX_SHARED_ISLANDS="${MAX_SHARED_ISLANDS:-3}"
MIN_POSITIVE_CHARS="${MIN_POSITIVE_CHARS:-4}"
MAX_PHRASE_CHARS="${MAX_PHRASE_CHARS:-6}"
MAX_PHRASE_WORDS="${MAX_PHRASE_WORDS:-2}"

# Dense readable-line policy.
SENTENCE_MIN_WORDS="${SENTENCE_MIN_WORDS:-6}"
SENTENCE_MAX_WORDS="${SENTENCE_MAX_WORDS:-10}"
MIN_SENTENCE_CHARS="${MIN_SENTENCE_CHARS:-28}"
MAX_SENTENCE_CHARS="${MAX_SENTENCE_CHARS:-55}"
MAX_FONT_CHUNK_WORDS="${MAX_FONT_CHUNK_WORDS:-2}"
FONT_SIZE="${FONT_SIZE:-60}"
MIN_FONT_SIZE="${MIN_FONT_SIZE:-42}"
MAX_FONT_SIZE="${MAX_FONT_SIZE:-64}"
MIN_LINE_FILL_RATIO="${MIN_LINE_FILL_RATIO:-0.90}"
PADDING="${PADDING:-8}"
SEGMENT_GAP_MIN_PX="${SEGMENT_GAP_MIN_PX:-2}"
SEGMENT_GAP_MAX_PX="${SEGMENT_GAP_MAX_PX:-6}"

# Synthetic appearance augmentation.
BLUR_PROB="${BLUR_PROB:-0.65}"
BLUR_MAX_RADIUS="${BLUR_MAX_RADIUS:-1.15}"
NOISE_PROB="${NOISE_PROB:-0.80}"
NOISE_SIGMA_MAX="${NOISE_SIGMA_MAX:-9.0}"
CONTRAST_MIN="${CONTRAST_MIN:-0.88}"
CONTRAST_MAX="${CONTRAST_MAX:-1.14}"
BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.90}"
BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.08}"

# Real-anchor appearance augmentation bundle. No geometric transform is permitted.
REAL_AUG_SEED="${REAL_AUG_SEED:-4242}"
REAL_BLUR_MIN_RADIUS="${REAL_BLUR_MIN_RADIUS:-0.15}"
REAL_BLUR_MAX_RADIUS="${REAL_BLUR_MAX_RADIUS:-1.00}"
REAL_NOISE_MIN_SIGMA="${REAL_NOISE_MIN_SIGMA:-2.0}"
REAL_NOISE_MAX_SIGMA="${REAL_NOISE_MAX_SIGMA:-8.0}"
REAL_CONTRAST_MIN="${REAL_CONTRAST_MIN:-0.90}"
REAL_CONTRAST_MAX="${REAL_CONTRAST_MAX:-1.12}"
REAL_GAMMA_MIN="${REAL_GAMMA_MIN:-0.88}"
REAL_GAMMA_MAX="${REAL_GAMMA_MAX:-1.12}"
REAL_SALT_PEPPER_MIN_PROB="${REAL_SALT_PEPPER_MIN_PROB:-0.001}"
REAL_SALT_PEPPER_MAX_PROB="${REAL_SALT_PEPPER_MAX_PROB:-0.006}"

SEED="${SEED:-42}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"
# Use all CPUs allocated by SLURM for the expensive rendering stage. Separate
# worker processes avoid the Python GIL; each worker is internally single-threaded.
export BRIDGE_BUILD_WORKERS="${BRIDGE_BUILD_WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --negatives-per-anchor "${NEGATIVES_PER_ANCHOR}"
  --negative-ngram "${NEGATIVE_NGRAM}"
  --min-overlap-word-chars "${MIN_OVERLAP_WORD_CHARS}"
  --max-shared-islands "${MAX_SHARED_ISLANDS}"
  --min-positive-chars "${MIN_POSITIVE_CHARS}"
  --max-phrase-chars "${MAX_PHRASE_CHARS}"
  --max-phrase-words "${MAX_PHRASE_WORDS}"
  --sentence-min-words "${SENTENCE_MIN_WORDS}"
  --sentence-max-words "${SENTENCE_MAX_WORDS}"
  --min-sentence-chars "${MIN_SENTENCE_CHARS}"
  --max-sentence-chars "${MAX_SENTENCE_CHARS}"
  --max-font-chunk-words "${MAX_FONT_CHUNK_WORDS}"
  --font-size "${FONT_SIZE}"
  --min-font-size "${MIN_FONT_SIZE}"
  --max-font-size "${MAX_FONT_SIZE}"
  --min-line-fill-ratio "${MIN_LINE_FILL_RATIO}"
  --padding "${PADDING}"
  --segment-gap-min-px "${SEGMENT_GAP_MIN_PX}"
  --segment-gap-max-px "${SEGMENT_GAP_MAX_PX}"
  --blur-prob "${BLUR_PROB}"
  --blur-max-radius "${BLUR_MAX_RADIUS}"
  --noise-prob "${NOISE_PROB}"
  --noise-sigma-max "${NOISE_SIGMA_MAX}"
  --contrast-min "${CONTRAST_MIN}"
  --contrast-max "${CONTRAST_MAX}"
  --brightness-min "${BRIGHTNESS_MIN}"
  --brightness-max "${BRIGHTNESS_MAX}"
  --seed "${SEED}"
  --max-anchors "${MAX_ANCHORS}"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then args+=(--overwrite); fi
if [[ -n "${BRIDGE_FONTS:-}" ]]; then args+=(--fonts "${BRIDGE_FONTS}"); fi

python scripts/data/build_real_conditioned_synthetic_bridge_v3_parallel.py "${args[@]}"

[[ -s "${OUTPUT_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing bridge manifest" >&2; exit 2; }
[[ -s "${OUTPUT_DIR}/metadata.json" ]] || { echo "ERROR: missing bridge metadata" >&2; exit 2; }
[[ -d "${OUTPUT_DIR}/masks" ]] || { echo "ERROR: missing bridge masks" >&2; exit 2; }

python scripts/data/organize_real_synthetic_bridge_v3.py --data-dir "${OUTPUT_DIR}"
python scripts/data/augment_bridge_v3_real_lines.py \
  --data-dir "${OUTPUT_DIR}" \
  --seed "${REAL_AUG_SEED}" \
  --blur-min-radius "${REAL_BLUR_MIN_RADIUS}" \
  --blur-max-radius "${REAL_BLUR_MAX_RADIUS}" \
  --noise-min-sigma "${REAL_NOISE_MIN_SIGMA}" \
  --noise-max-sigma "${REAL_NOISE_MAX_SIGMA}" \
  --contrast-min "${REAL_CONTRAST_MIN}" \
  --contrast-max "${REAL_CONTRAST_MAX}" \
  --gamma-min "${REAL_GAMMA_MIN}" \
  --gamma-max "${REAL_GAMMA_MAX}" \
  --salt-pepper-min-prob "${REAL_SALT_PEPPER_MIN_PROB}" \
  --salt-pepper-max-prob "${REAL_SALT_PEPPER_MAX_PROB}"
python scripts/data/create_bridge_v3_category_folders.py --data-dir "${OUTPUT_DIR}"

python scripts/data/smoke_test_real_synthetic_bridge_v3.py \
  --data-dir "${OUTPUT_DIR}" \
  --expected-negatives "${NEGATIVES_PER_ANCHOR}"
python scripts/data/validate_bridge_v3_font_size.py \
  --data-dir "${OUTPUT_DIR}" \
  --min-font-size "${MIN_FONT_SIZE}"
python scripts/data/validate_bridge_v3_dense_layout.py \
  --data-dir "${OUTPUT_DIR}" \
  --min-recorded-fill "${MIN_LINE_FILL_RATIO}" \
  --min-pixel-span 0.84 \
  --expected-negatives "${NEGATIVES_PER_ANCHOR}"
python scripts/data/validate_bridge_v3_real_augmentation.py \
  --data-dir "${OUTPUT_DIR}"

echo "Bridge V3 dataset ready: ${OUTPUT_DIR}"
echo "Parallel generation workers: ${BRIDGE_BUILD_WORKERS}"
echo "Dense line policy: recorded_fill>=${MIN_LINE_FILL_RATIO}, pixel_span>=0.84"
echo "Readable font policy: preferred=${FONT_SIZE}px min=${MIN_FONT_SIZE}px max=${MAX_FONT_SIZE}px"
echo "Real training variant: combined contrast+gamma+Gaussian blur+Gaussian noise"
echo "Real stored variants: binarized, Gaussian noise, Gaussian blur, blur+noise, binarized+noise, contrast+gamma, salt+pepper"
echo "Real augmentation is appearance-only; geometric=false"
echo "Human folders: ${OUTPUT_DIR}/{real,positive,negative}/<anchor_id>/"
echo "Human index: ${OUTPUT_DIR}/README_DATASET.md"
echo "Anchor index: ${OUTPUT_DIR}/anchor_index.jsonl"
echo "Real-line scrape index: ${OUTPUT_DIR}/real_lines_index.jsonl"

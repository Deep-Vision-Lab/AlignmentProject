#!/usr/bin/env bash
# Submit the full-quality training profile on the real Arabic Quran line dataset.
#
# The real loader is selected automatically from dataset_manifest.jsonl and
# returns the same image1/text1 + image2/text2 structure used by synthetic data.
# Images are resized, Otsu-binarized, polarity-corrected, converted to RGB, and
# ImageNet-normalized before entering the existing CNN/BiLSTM model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
export DATASET_TYPE="${DATASET_TYPE:-real}"
export REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest.jsonl}"

# The image-image phases require related content on both sides. Do not include
# no_shared_content as a positive pair. Medium matches are semi-positive and
# increase the real-data training set beyond the small high-match subset.
export REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
export DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"

# Real scanned-line preprocessing.
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-0}"

export JOB_ID="${JOB_ID:-real_arabic_fullquality_span3_comp_bin}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"

exec bash "${SCRIPT_DIR}/run_span_d3tw_full_quality.sh"

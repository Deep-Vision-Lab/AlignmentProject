#!/usr/bin/env bash
# Metrics-only real bbox.json evaluation: no per-pair PNG files.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate_quantitative_no_png.sh" >&2
  echo "Configure optional settings through environment variables." >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to model_best.pth or model_latest.pth.}"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
LABELS="${LABELS:-high_match,medium_match}"
N_SAMPLES="${N_SAMPLES:-2000}"
START_INDEX="${START_INDEX:-1}"
REAL_SPLIT="${REAL_SPLIT:-test}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/training-speed-optimization/quantitative_no_png/${RUN_TAG}}"

REAL_BOX_EVAL="${REAL_BOX_EVAL:-1}"
REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-0}"
REAL_BOX_IN_MASK_RULE="${REAL_BOX_IN_MASK_RULE:-center}"
REAL_BOX_MIN_COVERAGE="${REAL_BOX_MIN_COVERAGE:-0.50}"
REAL_BOX_COORDINATE_SPACE="${REAL_BOX_COORDINATE_SPACE:-original}"
REAL_BOX_BBOX_FORMAT="${REAL_BOX_BBOX_FORMAT:-auto}"
REAL_BOX_JSON="${REAL_BOX_JSON:-}"
REAL_BOX_ANNOTATIONS_ROOT="${REAL_BOX_ANNOTATIONS_ROOT:-${REAL_DATA_DIR}}"
REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES="${REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES:-0}"
REAL_BOX_JSON_COUNT="${REAL_BOX_JSON_COUNT:-deferred-to-slurm-job}"

[[ -d "${REAL_DATA_DIR}" ]] || {
  echo "ERROR: real dataset directory not found: ${REAL_DATA_DIR}" >&2
  exit 2
}
[[ -f "${ARABIC_MANIFEST}" ]] || {
  echo "ERROR: manifest not found: ${ARABIC_MANIFEST}" >&2
  exit 2
}
[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: N_SAMPLES must be a positive integer." >&2
  exit 2
}
[[ "${START_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: START_INDEX must be a non-negative integer." >&2
  exit 2
}

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_quant_no_png_${RUN_TAG}}"

print_config() {
  printf '%s\n' \
    "Quantitative real bbox.json evaluation (no PNG)" \
    "  branch       = $(git branch --show-current)" \
    "  run          = ${RUN_TAG}" \
    "  checkpoint   = ${WEIGHTS}" \
    "  split        = ${REAL_SPLIT}" \
    "  labels       = ${LABELS}" \
    "  sample cap   = ${N_SAMPLES} total combined" \
    "  save PNG     = no" \
    "  box scoring  = ${REAL_BOX_EVAL}" \
    "  bbox source  = ${REAL_BOX_JSON:-${REAL_BOX_ANNOTATIONS_ROOT}}" \
    "  bbox name    = bbox.json" \
    "  bbox files   = ${REAL_BOX_JSON_COUNT}" \
    "  results root = ${RESULTS_ROOT}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  print_config
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --tasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

BOX_SOURCE="${REAL_BOX_JSON:-${REAL_BOX_ANNOTATIONS_ROOT}}"
[[ -e "${BOX_SOURCE}" ]] || {
  echo "ERROR: bbox.json source does not exist: ${BOX_SOURCE}" >&2
  exit 2
}
if [[ -f "${BOX_SOURCE}" ]]; then
  if [[ "$(basename "${BOX_SOURCE}")" != "bbox.json" && "${REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES}" != "1" ]]; then
    echo "ERROR: REAL_BOX_JSON must point to bbox.json, not $(basename "${BOX_SOURCE}")." >&2
    exit 2
  fi
  REAL_BOX_JSON_COUNT=1
else
  if [[ "${REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES}" == "1" ]]; then
    REAL_BOX_JSON_COUNT="$(find "${BOX_SOURCE}" -type f \
      \( -iname 'bbox.json' -o -iname 'bboxes.json' -o -iname 'bounding_boxes.json' \) \
      -print 2>/dev/null | wc -l | tr -d ' ')"
  else
    REAL_BOX_JSON_COUNT="$(find "${BOX_SOURCE}" -type f -iname 'bbox.json' \
      -print 2>/dev/null | wc -l | tr -d ' ')"
  fi
fi
export REAL_BOX_JSON_COUNT
if [[ "${REAL_BOX_JSON_COUNT}" -eq 0 ]]; then
  echo "ERROR: no canonical bbox.json annotation files were found under ${BOX_SOURCE}" >&2
  echo "The evaluator intentionally ignores unrelated debug/bboxes.json files." >&2
  exit 2
fi

export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-${TARGET_INK_HEIGHT_RATIO}}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_EVAL_BALANCED="${REAL_EVAL_BALANCED:-1}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export SW_MIN_INK="${SW_MIN_INK:-0.02}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"
export REAL_BOX_EVAL REAL_REQUIRE_BOX_ANNOTATIONS REAL_BOX_IN_MASK_RULE
export REAL_BOX_MIN_COVERAGE REAL_BOX_COORDINATE_SPACE REAL_BOX_BBOX_FORMAT
export REAL_BOX_JSON REAL_BOX_ANNOTATIONS_ROOT REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES

mkdir -p "${RESULTS_ROOT}"
print_config

python -m Evaluation.eval_img_align_sw_no_png \
  --weights "${WEIGHTS}" \
  --device cuda \
  --data-dir "${REAL_DATA_DIR}" \
  --arabic-manifest "${ARABIC_MANIFEST}" \
  --dataset-type real \
  --batch \
  --real-split "${REAL_SPLIT}" \
  --real-labels "${LABELS}" \
  --real-text-key "${REAL_TEXT_KEY}" \
  --real-min-text-score "${REAL_MIN_TEXT_SCORE}" \
  --start-index "${START_INDEX}" \
  --n-samples "${N_SAMPLES}" \
  --feature "${FEATURE}" \
  --score-mode "${SCORE_MODE}" \
  --score-clip "${SCORE_CLIP}" \
  --threshold "${THRESHOLD}" \
  --gap "${GAP}" \
  --heatmap-source "${HEATMAP_SOURCE}" \
  --no-heatmap \
  --no-annotate-heatmap-values \
  --no-save-binarized-images \
  --output-dir "${RESULTS_ROOT}"

printf '%s\n' \
  "Evaluation complete." \
  "  samples = ${RESULTS_ROOT}/samples.csv" \
  "  summary = ${RESULTS_ROOT}/summary.json" \
  "  PNGs    = disabled"

#!/usr/bin/env bash
# Canonical real-data evaluation using bbox.json quantitative annotations.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate.sh" >&2
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

MODEL_TAG="${MODEL_TAG:-$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
LABELS="${LABELS:-high_match,medium_match}"
N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"
REAL_SPLIT="${REAL_SPLIT:-test}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
# Real SCORE_MODE=auto resolves to mutual-z.  A threshold of zero is the natural
# neutral boundary for centered/z-normalized scores; 0.45 is retained only for
# explicitly raw cosine scoring.
if [[ -z "${THRESHOLD+x}" ]]; then
  case "${SCORE_MODE,,}" in
    raw) THRESHOLD=0.45 ;;
    *) THRESHOLD=0.0 ;;
  esac
fi
GAP="${GAP:--0.30}"
# Show the matrix that Smith-Waterman actually consumes.  The accumulated DP
# matrix is zero-clamped by local alignment and can therefore be entirely zero
# when no positive path exists, hiding useful diagnostic structure.
HEATMAP_SOURCE="${HEATMAP_SOURCE:-match-score}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/${MODEL_TAG}/Real_Experiments/${RUN_TAG}}"

# Object-level quantitative localization from bbox.json.
REAL_BOX_EVAL="${REAL_BOX_EVAL:-1}"
REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-0}"
REAL_BOX_IN_MASK_RULE="${REAL_BOX_IN_MASK_RULE:-center}"
REAL_BOX_MIN_COVERAGE="${REAL_BOX_MIN_COVERAGE:-0.50}"
REAL_BOX_COORDINATE_SPACE="${REAL_BOX_COORDINATE_SPACE:-original}"
REAL_BOX_BBOX_FORMAT="${REAL_BOX_BBOX_FORMAT:-auto}"
# REAL_BOX_JSON may point to one global bbox.json or a directory containing
# per-page/per-side bbox.json files. The root fallback is the real dataset.
REAL_BOX_JSON="${REAL_BOX_JSON:-}"
REAL_BOX_ANNOTATIONS_ROOT="${REAL_BOX_ANNOTATIONS_ROOT:-${REAL_DATA_DIR}}"
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
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_${MODEL_TAG}_${RUN_TAG}}"

print_config() {
  printf '%s\n' \
    "Canonical real bbox.json evaluation" \
    "  branch       = $(git branch --show-current)" \
    "  model        = ${MODEL_TAG}" \
    "  run          = ${RUN_TAG}" \
    "  checkpoint   = ${WEIGHTS}" \
    "  split        = ${REAL_SPLIT}" \
    "  labels       = ${LABELS}" \
    "  samples      = ${N_SAMPLES}" \
    "  feature      = ${FEATURE}" \
    "  score mode   = ${SCORE_MODE}" \
    "  threshold    = ${THRESHOLD}" \
    "  gap          = ${GAP}" \
    "  heatmap      = ${HEATMAP_SOURCE}" \
    "  box scoring  = ${REAL_BOX_EVAL}" \
    "  box rule     = ${REAL_BOX_IN_MASK_RULE}" \
    "  bbox source  = ${REAL_BOX_JSON:-${REAL_BOX_ANNOTATIONS_ROOT}}" \
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

if [[ "${REAL_BOX_EVAL}" == "1" || "${REAL_BOX_EVAL,,}" == "true" ]]; then
  BOX_SOURCE="${REAL_BOX_JSON:-${REAL_BOX_ANNOTATIONS_ROOT}}"
  [[ -e "${BOX_SOURCE}" ]] || {
    echo "ERROR: bbox.json source does not exist: ${BOX_SOURCE}" >&2
    exit 2
  }
  if [[ -f "${BOX_SOURCE}" ]]; then
    [[ "$(basename "${BOX_SOURCE}")" =~ ^(bbox|bboxes|bounding_boxes)\.json$ ]] || {
      echo "ERROR: REAL_BOX_JSON must point to bbox.json, bboxes.json, or bounding_boxes.json" >&2
      exit 2
    }
    REAL_BOX_JSON_COUNT=1
  else
    REAL_BOX_JSON_COUNT="$(find "${BOX_SOURCE}" -type f \
      \( -iname 'bbox.json' -o -iname 'bboxes.json' -o -iname 'bounding_boxes.json' \) \
      -print 2>/dev/null | wc -l | tr -d ' ')"
  fi
  export REAL_BOX_JSON_COUNT
  if [[ "${REAL_BOX_JSON_COUNT}" -eq 0 ]]; then
    echo "ERROR: no bbox.json annotation files were found under:" >&2
    echo "  ${BOX_SOURCE}" >&2
    echo "Set REAL_BOX_JSON to the global bbox.json file or its containing directory." >&2
    exit 2
  fi
fi

# Keep evaluation preprocessing identical to training unless explicitly
# overridden. Model window/stride and backend are reconstructed from checkpoint.
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
export REAL_BOX_JSON REAL_BOX_ANNOTATIONS_ROOT

print_config
IFS=',' read -r -a LABEL_ARRAY <<< "${LABELS}"
for LABEL in "${LABEL_ARRAY[@]}"; do
  LABEL="${LABEL//[[:space:]]/}"
  [[ -n "${LABEL}" ]] || continue
  OUTPUT_DIR="${RESULTS_ROOT}/${LABEL}"
  mkdir -p "${OUTPUT_DIR}"

  printf '%s\n' \
    "Evaluating label=${LABEL}" \
    "  output=${OUTPUT_DIR}"

  python -m Evaluation.eval_img_align_sw \
    --weights "${WEIGHTS}" \
    --device cuda \
    --data-dir "${REAL_DATA_DIR}" \
    --arabic-manifest "${ARABIC_MANIFEST}" \
    --dataset-type real \
    --batch \
    --real-split "${REAL_SPLIT}" \
    --real-labels "${LABEL}" \
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
    --no-save-binarized-images \
    --output-dir "${OUTPUT_DIR}"
done

#!/usr/bin/env bash
# Quantitative real-data evaluation without dense masks.
# Run from the login node:
#   WEIGHTS=Weights/<run>/model_best.pth bash Evaluation/evaluate_quantitative.sh
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate_quantitative.sh" >&2
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
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }

MODEL_TAG="${MODEL_TAG:-$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
REAL_SPLIT="${REAL_SPLIT:-test}"
LABELS="${LABELS:-high_match,medium_match}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
CROP_LINES="${CROP_LINES:-80}"
CROPS_PER_LINE="${CROPS_PER_LINE:-3}"
CROP_FRACTIONS="${CROP_FRACTIONS:-0.20,0.35,0.50}"
DEGRADATIONS="${DEGRADATIONS:-none,blur,contrast,noise,morphology}"
RETRIEVAL_QUERIES="${RETRIEVAL_QUERIES:-80}"
RETRIEVAL_POOL_SIZE="${RETRIEVAL_POOL_SIZE:-20}"
RANKING_SCORE="${RANKING_SCORE:-normalized_sw}"
INTERVAL_MANIFEST="${INTERVAL_MANIFEST:-}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/Results/Evaluation/${MODEL_TAG}/Real_Quantitative/${RUN_TAG}}"

[[ -d "${REAL_DATA_DIR}" ]] || { echo "ERROR: real data directory not found: ${REAL_DATA_DIR}" >&2; exit 2; }
[[ -f "${ARABIC_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${ARABIC_MANIFEST}" >&2; exit 2; }
if [[ -n "${INTERVAL_MANIFEST}" && ! -f "${INTERVAL_MANIFEST}" ]]; then
  echo "ERROR: interval manifest not found: ${INTERVAL_MANIFEST}" >&2
  exit 2
fi
for variable in CROP_LINES CROPS_PER_LINE RETRIEVAL_QUERIES RETRIEVAL_POOL_SIZE; do
  value="${!variable}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${variable} must be a positive integer, got ${value}." >&2
    exit 2
  }
done

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-6}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-quant_${MODEL_TAG}_${RUN_TAG}}"

print_config() {
  printf '%s\n' \
    "Real quantitative evaluation" \
    "  branch          = $(git branch --show-current)" \
    "  model           = ${MODEL_TAG}" \
    "  run             = ${RUN_TAG}" \
    "  checkpoint      = ${WEIGHTS}" \
    "  split/labels    = ${REAL_SPLIT} / ${LABELS}" \
    "  crop examples   = ${CROP_LINES} lines x ${CROPS_PER_LINE}" \
    "  retrieval       = ${RETRIEVAL_QUERIES} queries x pool ${RETRIEVAL_POOL_SIZE}" \
    "  ranking score   = ${RANKING_SCORE}" \
    "  interval labels = ${INTERVAL_MANIFEST:-none}" \
    "  results         = ${RESULTS_DIR}"
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
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
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

print_config
mkdir -p "${RESULTS_DIR}"

COMMAND=(
  python -m Evaluation.quantitative_real
  --weights "${WEIGHTS}"
  --output-dir "${RESULTS_DIR}"
  --device cuda
  --real-data-dir "${REAL_DATA_DIR}"
  --arabic-manifest "${ARABIC_MANIFEST}"
  --real-split "${REAL_SPLIT}"
  --labels "${LABELS}"
  --real-text-key "${REAL_TEXT_KEY}"
  --real-min-text-score "${REAL_MIN_TEXT_SCORE}"
  --split-seed "${SPLIT_SEED}"
  --seed "${EVAL_SEED}"
  --feature "${FEATURE}"
  --score-mode "${SCORE_MODE}"
  --score-clip "${SCORE_CLIP}"
  --threshold "${THRESHOLD}"
  --gap "${GAP}"
  --crop-lines "${CROP_LINES}"
  --crops-per-line "${CROPS_PER_LINE}"
  --crop-fractions "${CROP_FRACTIONS}"
  --degradations "${DEGRADATIONS}"
  --retrieval-queries "${RETRIEVAL_QUERIES}"
  --retrieval-pool-size "${RETRIEVAL_POOL_SIZE}"
  --ranking-score "${RANKING_SCORE}"
)
if [[ -n "${INTERVAL_MANIFEST}" ]]; then
  COMMAND+=(--interval-manifest "${INTERVAL_MANIFEST}")
fi

"${COMMAND[@]}"

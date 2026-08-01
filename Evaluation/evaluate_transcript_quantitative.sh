#!/usr/bin/env bash
# Transcript-supervised real-data evaluation without masks or spatial labels.
# Run from the login node:
#   WEIGHTS=Weights/<run>/model_best.pth bash Evaluation/evaluate_transcript_quantitative.sh
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate_transcript_quantitative.sh" >&2
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
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/${MODEL_TAG}/Transcript_Quantitative/${RUN_TAG}}"

POSITIVE_OVERLAP="${POSITIVE_OVERLAP:-0.50}"
NEGATIVE_OVERLAP="${NEGATIVE_OVERLAP:-0.10}"
MIN_SHARED_WORDS="${MIN_SHARED_WORDS:-2}"
MAX_VALID_PAIRS="${MAX_VALID_PAIRS:-0}"
MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-0}"
RETRIEVAL_QUERIES="${RETRIEVAL_QUERIES:-100}"
RETRIEVAL_POOL_SIZE="${RETRIEVAL_POOL_SIZE:-20}"
WORD_PAIRS="${WORD_PAIRS:-100}"
WORD_MIN_SUPPORT="${WORD_MIN_SUPPORT:-1}"
FEATURE="${FEATURE:-contextual}"
WORD_FEATURE="${WORD_FEATURE:-local}"
RANKING_SCORE="${RANKING_SCORE:-coverage_sw}"
MIN_PATH_STEPS="${MIN_PATH_STEPS:-5}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
EVAL_SEED="${EVAL_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"

[[ -f "${ARABIC_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${ARABIC_MANIFEST}" >&2; exit 2; }

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-40G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-transcript_${MODEL_TAG}_${RUN_TAG}}"

print_config() {
  printf '%s\n' \
    "Transcript-supervised real evaluation" \
    "  branch          = $(git branch --show-current)" \
    "  model           = ${MODEL_TAG}" \
    "  checkpoint      = ${WEIGHTS}" \
    "  transcript key  = ${REAL_TEXT_KEY}" \
    "  positive overlap= ${POSITIVE_OVERLAP}" \
    "  negative overlap= ${NEGATIVE_OVERLAP}" \
    "  retrieval       = ${RETRIEVAL_QUERIES} x ${RETRIEVAL_POOL_SIZE}" \
    "  word pairs      = ${WORD_PAIRS}" \
    "  results         = ${RESULTS_ROOT}"
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
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export SW_MIN_INK="${SW_MIN_INK:-0.02}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

mkdir -p "${RESULTS_ROOT}"
print_config
python -m Evaluation.transcript_quantitative \
  --weights "${WEIGHTS}" \
  --output-dir "${RESULTS_ROOT}" \
  --device cuda \
  --real-data-dir "${REAL_DATA_DIR}" \
  --arabic-manifest "${ARABIC_MANIFEST}" \
  --real-text-key "${REAL_TEXT_KEY}" \
  --split-seed "${SPLIT_SEED}" \
  --seed "${EVAL_SEED}" \
  --positive-overlap "${POSITIVE_OVERLAP}" \
  --negative-overlap "${NEGATIVE_OVERLAP}" \
  --min-shared-words "${MIN_SHARED_WORDS}" \
  --max-valid-pairs "${MAX_VALID_PAIRS}" \
  --max-test-pairs "${MAX_TEST_PAIRS}" \
  --retrieval-queries "${RETRIEVAL_QUERIES}" \
  --retrieval-pool-size "${RETRIEVAL_POOL_SIZE}" \
  --word-pairs "${WORD_PAIRS}" \
  --word-min-support "${WORD_MIN_SUPPORT}" \
  --feature "${FEATURE}" \
  --word-feature "${WORD_FEATURE}" \
  --ranking-score "${RANKING_SCORE}" \
  --min-path-steps "${MIN_PATH_STEPS}" \
  --score-mode "${SCORE_MODE}" \
  --score-clip "${SCORE_CLIP}" \
  --threshold "${THRESHOLD}" \
  --gap "${GAP}"

#!/usr/bin/env bash
# Run the same component-aware Needleman-Wunsch evaluator used for the
# augmented synthetic dataset, but load pairs from ArabicDatasetRealAug10K.
# No ground-truth mask or bbox annotation is used by this script.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate_nw_real_augmented.sh" >&2
  echo "Optional overrides: REAL_DATA_DIR, TEST_MANIFEST, N_SAMPLES, LABELS, RESULTS_ROOT." >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to the trained model_latest.pth/model_best.pth checkpoint.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

NW_ENTRYPOINT="${PROJECT_DIR}/Evaluation/eval_img_align_nw.py"
[[ -f "${NW_ENTRYPOINT}" ]] || {
  echo "ERROR: NW evaluator is not available: ${NW_ENTRYPOINT}" >&2
  exit 2
}

REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDatasetRealAug10K}"
REAL_DATA_DIR="$(readlink -f "${REAL_DATA_DIR}")"
TEST_MANIFEST="${TEST_MANIFEST:-${REAL_DATA_DIR}/test_manifest.jsonl}"
TEST_MANIFEST="$(readlink -f "${TEST_MANIFEST}")"
[[ -d "${REAL_DATA_DIR}" ]] || {
  echo "ERROR: real augmented dataset not found: ${REAL_DATA_DIR}" >&2
  exit 2
}
[[ -f "${TEST_MANIFEST}" ]] || {
  echo "ERROR: test manifest not found: ${TEST_MANIFEST}" >&2
  exit 2
}

TEST_ROWS="$(grep -cve '^[[:space:]]*$' "${TEST_MANIFEST}" || true)"
[[ "${TEST_ROWS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: test manifest contains no rows: ${TEST_MANIFEST}" >&2
  exit 2
}

N_SAMPLES="${N_SAMPLES:-10}"
START_INDEX="${START_INDEX:-1}"
LABELS="${LABELS:-all}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/NW/RealAugmented/${RUN_TAG}}"

[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: N_SAMPLES must be a positive integer." >&2
  exit 2
}
[[ "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: START_INDEX must be a positive integer." >&2
  exit 2
}
if (( START_INDEX > TEST_ROWS )); then
  echo "ERROR: START_INDEX=${START_INDEX} exceeds test rows=${TEST_ROWS}." >&2
  exit 2
fi
AVAILABLE=$((TEST_ROWS - START_INDEX + 1))
if (( N_SAMPLES > AVAILABLE )); then
  N_SAMPLES="${AVAILABLE}"
fi

# Real images are normalized before feature extraction, but the NW algorithm,
# score construction, traceback, supported-component extraction, and plots are
# the same ones used for augmented synthetic evaluation.
LINE_HEIGHT="${LINE_HEIGHT:-128}"
LINE_WIDTH="${LINE_WIDTH:-1024}"
TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-${TARGET_INK_HEIGHT_RATIO}}"
ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
REAL_BINARIZE="${REAL_BINARIZE:-1}"
REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
SW_INK_AWARE="${SW_INK_AWARE:-1}"
SW_MIN_INK="${SW_MIN_INK:-0.02}"
SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

# Same supported-component interpretation used by augmented synthetic NW.
NW_COMPONENT_MAX_COMPONENTS="${NW_COMPONENT_MAX_COMPONENTS:-3}"
NW_COMPONENT_MIN_MATCHES="${NW_COMPONENT_MIN_MATCHES:-7}"
NW_COMPONENT_MIN_SPAN_WINDOWS="${NW_COMPONENT_MIN_SPAN_WINDOWS:-7}"
NW_COMPONENT_MIN_SPAN_FRACTION="${NW_COMPONENT_MIN_SPAN_FRACTION:-0.13}"
NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--1000000.0}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-nw_real_aug_${RUN_TAG}}"

export PROJECT_DIR WEIGHTS REAL_DATA_DIR TEST_MANIFEST TEST_ROWS
export N_SAMPLES START_INDEX LABELS FEATURE SCORE_MODE SCORE_CLIP THRESHOLD GAP
export HEATMAP_SOURCE RUN_TAG RESULTS_ROOT
export LINE_HEIGHT LINE_WIDTH TARGET_INK_HEIGHT_RATIO ZERO_SHOT_TARGET_INK_HEIGHT_RATIO
export ZERO_SHOT_PREPROCESS ZERO_SHOT_PRESERVE_ASPECT ZERO_SHOT_FOREGROUND_CROP
export ZERO_SHOT_SOURCE_GEOMETRY REAL_BINARIZE REAL_BINARIZE_METHOD
export REAL_BINARIZE_AUTO_INVERT REAL_BINARIZE_AUTOCONTRAST
export SW_INK_AWARE SW_MIN_INK SW_BLANK_BLANK_SCORE SW_BLANK_INK_SCORE
export NW_COMPONENT_MAX_COMPONENTS NW_COMPONENT_MIN_MATCHES
export NW_COMPONENT_MIN_SPAN_WINDOWS NW_COMPONENT_MIN_SPAN_FRACTION
export NW_COMPONENT_WEAK_GLOBAL_SCORE
export CONDA_ENV PARTITION GPU_RESOURCE CPUS_PER_TASK MEMORY TIME_LIMIT
export MAIL_USER EVAL_JOB_NAME
set +a

print_config() {
  printf '%s\n' \
    "Component-aware NW evaluation on real augmented data" \
    "  branch=$(git branch --show-current)" \
    "  checkpoint=${WEIGHTS}" \
    "  dataset=${REAL_DATA_DIR}" \
    "  manifest=${TEST_MANIFEST}" \
    "  split=explicit test_manifest.jsonl (no re-split)" \
    "  manifest rows=${TEST_ROWS}" \
    "  start=${START_INDEX}" \
    "  samples=${N_SAMPLES}" \
    "  labels=${LABELS}" \
    "  algorithm=Needleman-Wunsch, same component path as augmented synthetic" \
    "  max components=${NW_COMPONENT_MAX_COMPONENTS}" \
    "  ground-truth masks=not used" \
    "  bbox annotations=not used" \
    "  results=${RESULTS_ROOT}" \
    "  GPU=${GPU_RESOURCE}:1" \
    "  time limit=${TIME_LIMIT}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  print_config
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

resolve_env_prefix() {
  local candidate
  for candidate in \
    "${CONDA_PREFIX:-}" \
    "${HOME}/.conda/envs/${CONDA_ENV}" \
    "${HOME}/miniconda3/envs/${CONDA_ENV}" \
    "${HOME}/anaconda3/envs/${CONDA_ENV}"; do
    [[ -n "${candidate}" ]] || continue
    [[ -x "${candidate}/bin/python" ]] || continue
    if "${candidate}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
import numpy
from PIL import Image
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ENV_PREFIX="$(resolve_env_prefix)" || {
  echo "ERROR: could not find usable conda environment '${CONDA_ENV}'." >&2
  exit 2
}
TRAIN_PYTHON="${ENV_PREFIX}/bin/python"
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

mkdir -p "${RESULTS_ROOT}"
print_config
printf '%s\n' "  python=${TRAIN_PYTHON}"
"${TRAIN_PYTHON}" -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
nvidia-smi -L || true

"${TRAIN_PYTHON}" -m Evaluation.eval_img_align_nw \
  --weights "${WEIGHTS}" \
  --device cuda \
  --data-dir "${REAL_DATA_DIR}" \
  --arabic-manifest "${TEST_MANIFEST}" \
  --dataset-type real \
  --batch \
  --real-split all \
  --real-labels "${LABELS}" \
  --real-text-key text_original_path \
  --real-min-text-score 0.0 \
  --start-index "${START_INDEX}" \
  --n-samples "${N_SAMPLES}" \
  --feature "${FEATURE}" \
  --score-mode "${SCORE_MODE}" \
  --score-clip "${SCORE_CLIP}" \
  --threshold "${THRESHOLD}" \
  --gap "${GAP}" \
  --heatmap-source "${HEATMAP_SOURCE}" \
  --no-annotate-heatmap-values \
  --no-save-binarized-images \
  --output-dir "${RESULTS_ROOT}"

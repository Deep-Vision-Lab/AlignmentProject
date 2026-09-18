#!/usr/bin/env bash
# The single public evaluation launcher for the current restoration alignment model.
#
# Usage from the login node:
#   WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
#   EVAL_MODE=all \
#   bash Evaluation/evaluate.sh
#
# EVAL_MODE: qualitative | quantitative | all
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> EVAL_MODE=qualitative|quantitative|all bash Evaluation/evaluate.sh" >&2
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

: "${WEIGHTS:?Set WEIGHTS to the trained checkpoint (for Point 3-6 use Weights/res18_tinyvit_point2/model_latest.pth).}"
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }

EVAL_MODE="${EVAL_MODE:-all}"
case "${EVAL_MODE}" in
  qualitative|quantitative|all) ;;
  *) echo "ERROR: EVAL_MODE must be qualitative, quantitative, or all" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")")}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
REAL_SPLIT="${REAL_SPLIT:-test}"
LABELS="${LABELS:-high_match,medium_match}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-raw}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.0}"
GAP="${GAP:--0.30}"
EVAL_SEED="${EVAL_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"

# Qualitative settings.
N_SAMPLES="${N_SAMPLES:-50}"
START_INDEX="${START_INDEX:-1}"

# Quantitative external-target settings.
CROP_LINES="${CROP_LINES:-200}"
CROPS_PER_LINE="${CROPS_PER_LINE:-5}"
CROP_FRACTIONS="${CROP_FRACTIONS:-0.10,0.20,0.30,0.40,0.50}"
CROP_DEGRADATIONS="${CROP_DEGRADATIONS:-blur,contrast,noise}"
RETRIEVAL_QUERIES="${RETRIEVAL_QUERIES:-100}"
RETRIEVAL_POOL_SIZE="${RETRIEVAL_POOL_SIZE:-20}"
CALIBRATION_QUERIES="${CALIBRATION_QUERIES:-40}"
RANKING_SCORE="${RANKING_SCORE:-normalized_sw}"
INTERVAL_MANIFEST="${INTERVAL_MANIFEST:-}"

# Label-free diagnostics.
CYCLE_PAIRS="${CYCLE_PAIRS:-80}"
ROBUSTNESS_PAIRS="${ROBUSTNESS_PAIRS:-30}"
ROBUSTNESS_MODES="${ROBUSTNESS_MODES:-blur,contrast,brightness,noise,horizontal_scale,vertical_shift,erosion,dilation}"
MIN_INK="${MIN_INK:-0.02}"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/Restoration/${RUN_TAG}}"
QUALITATIVE_DIR="${QUALITATIVE_DIR:-${RESULTS_ROOT}/Qualitative}"
QUANTITATIVE_DIR="${QUANTITATIVE_DIR:-${RESULTS_ROOT}/Quantitative}"

[[ -d "${REAL_DATA_DIR}" ]] || {
  echo "ERROR: real dataset directory not found: ${REAL_DATA_DIR}" >&2
  exit 2
}
[[ -f "${ARABIC_MANIFEST}" ]] || {
  echo "ERROR: real dataset manifest not found: ${ARABIC_MANIFEST}" >&2
  exit 2
}
if [[ -n "${INTERVAL_MANIFEST}" && ! -f "${INTERVAL_MANIFEST}" ]]; then
  echo "ERROR: sparse interval manifest not found: ${INTERVAL_MANIFEST}" >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-48G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-restoration_eval_${EVAL_MODE}}"

print_config() {
  printf '%s\n' \
    "AlignmentProject evaluation" \
    "  branch              = $(git branch --show-current)" \
    "  mode                = ${EVAL_MODE}" \
    "  checkpoint          = ${WEIGHTS}" \
    "  real data           = ${REAL_DATA_DIR}" \
    "  split / labels      = ${REAL_SPLIT} / ${LABELS}" \
    "  feature             = ${FEATURE} (restoration contextual = trained fused output)" \
    "  model input         = original RGB, full-image 1024x128 resize, no binarization" \
    "  checkpoint geometry = expected physical windows 128x32, stride 16" \
    "  results             = ${RESULTS_ROOT}"
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Current experiment policy: preserve original RGB pixels for evaluation.
# ZERO_SHOT_PREPROCESS=0 gives a deterministic full-image resize without
# foreground crop, aspect-ratio padding, autocontrast, or binarization.
export LINE_HEIGHT=128
export LINE_WIDTH=1024
export ZERO_SHOT_PREPROCESS=0
export ZERO_SHOT_PRESERVE_ASPECT=0
export ZERO_SHOT_FOREGROUND_CROP=0
export REAL_BINARIZE=0
export SYNTHETIC_BINARIZE=0
export REAL_BINARIZE_AUTOCONTRAST=0
export REAL_BINARIZE_AUTO_INVERT=0
export REAL_EVAL_BALANCED=1
export SW_INK_AWARE=1
export SW_MIN_INK="${MIN_INK}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

# Slurm already exposes the allocated physical GPU as a job-local CUDA device.
# Never remap CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS / SLURM_STEP_GPUS.
echo "GPU environment (preserved from Slurm):"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
nvidia-smi -L || true

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable inside allocated Slurm job")
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile \
  Evaluation/_eval_utils.py \
  Evaluation/eval_img_align_sw.py \
  Evaluation/sw_runner.py \
  Evaluation/quantitative_real.py \
  Evaluation/quantitative_diagnostics.py

print_config
mkdir -p "${RESULTS_ROOT}"

if [[ "${EVAL_MODE}" == "qualitative" || "${EVAL_MODE}" == "all" ]]; then
  IFS=',' read -r -a LABEL_ARRAY <<< "${LABELS}"
  for LABEL in "${LABEL_ARRAY[@]}"; do
    LABEL="${LABEL//[[:space:]]/}"
    [[ -n "${LABEL}" ]] || continue
    OUTPUT_DIR="${QUALITATIVE_DIR}/${LABEL}"
    mkdir -p "${OUTPUT_DIR}"
    python -u -m Evaluation.eval_img_align_sw \
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
      --split-seed "${SPLIT_SEED}" \
      --start-index "${START_INDEX}" \
      --n-samples "${N_SAMPLES}" \
      --feature "${FEATURE}" \
      --score-mode "${SCORE_MODE}" \
      --score-clip "${SCORE_CLIP}" \
      --threshold "${THRESHOLD}" \
      --gap "${GAP}" \
      --heatmap-source dp-score \
      --no-save-binarized-images \
      --output-dir "${OUTPUT_DIR}"
  done
fi

if [[ "${EVAL_MODE}" == "quantitative" || "${EVAL_MODE}" == "all" ]]; then
  mkdir -p "${QUANTITATIVE_DIR}"
  COMMAND=(
    python -u -m Evaluation.quantitative_real
    --weights "${WEIGHTS}"
    --output-dir "${QUANTITATIVE_DIR}"
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
    --degradations "${CROP_DEGRADATIONS}"
    --retrieval-queries "${RETRIEVAL_QUERIES}"
    --retrieval-pool-size "${RETRIEVAL_POOL_SIZE}"
    --calibration-queries "${CALIBRATION_QUERIES}"
    --ranking-score "${RANKING_SCORE}"
    --cycle-pairs "${CYCLE_PAIRS}"
    --robustness-pairs "${ROBUSTNESS_PAIRS}"
    --robustness-modes "${ROBUSTNESS_MODES}"
    --min-ink "${MIN_INK}"
  )
  if [[ -n "${INTERVAL_MANIFEST}" ]]; then
    COMMAND+=(--interval-manifest "${INTERVAL_MANIFEST}")
  fi
  "${COMMAND[@]}"
fi

echo "Evaluation complete: ${RESULTS_ROOT}"

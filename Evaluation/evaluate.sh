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

RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")")_$(date +%Y%m%d_%H%M%S)}"
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
AUTO_CALIBRATE_SW="${AUTO_CALIBRATE_SW:-1}"
SW_CALIBRATION_LINES="${SW_CALIBRATION_LINES:-20}"
SW_CALIBRATION_THRESHOLDS="${SW_CALIBRATION_THRESHOLDS:-0.0,0.15,0.30,0.45}"
SW_CALIBRATION_GAPS="${SW_CALIBRATION_GAPS:--0.15,-0.30,-0.45}"
INTERVAL_MANIFEST="${INTERVAL_MANIFEST:-}"

# Label-free real-data diagnostics.
CYCLE_PAIRS="${CYCLE_PAIRS:-80}"
ROBUSTNESS_PAIRS="${ROBUSTNESS_PAIRS:-30}"
ROBUSTNESS_MODES="${ROBUSTNESS_MODES:-blur,contrast,brightness,noise,horizontal_scale,vertical_shift,erosion,dilation}"
MIN_INK="${MIN_INK:-0.02}"

# Six-point checklist continuation.
SYNTHETIC_DATASET="${SYNTHETIC_DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
RUN_POINT3="${RUN_POINT3:-1}"
POINT3_SAMPLES="${POINT3_SAMPLES:-10}"
RUN_POINT45="${RUN_POINT45:-1}"
POINT45_SAMPLES="${POINT45_SAMPLES:-100}"
POINT45_MIN_WINDOWS="${POINT45_MIN_WINDOWS:-5}"
POINT45_MIN_IOU="${POINT45_MIN_IOU:-0.50}"
RUN_POINT6="${RUN_POINT6:-1}"
POINT6_LINE_PAIRS="${POINT6_LINE_PAIRS:-80}"
POINT6_QUERIES="${POINT6_QUERIES:-100}"
POINT6_TOP_K="${POINT6_TOP_K:-5}"
POINT6_VISUALIZE="${POINT6_VISUALIZE:-20}"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/Restoration/${RUN_TAG}}"
QUALITATIVE_DIR="${QUALITATIVE_DIR:-${RESULTS_ROOT}/Qualitative}"
QUANTITATIVE_DIR="${QUANTITATIVE_DIR:-${RESULTS_ROOT}/Quantitative}"
POINT3_DIR="${POINT3_DIR:-${RESULTS_ROOT}/Point3_TrainingPaths}"
POINT45_ROOT="${POINT45_ROOT:-${RESULTS_ROOT}/Point45_SpatialCorrectness}"
POINT6_DIR="${POINT6_DIR:-${RESULTS_ROOT}/Point6_WindowNeighbors}"

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
if [[ ("${EVAL_MODE}" == "quantitative" || "${EVAL_MODE}" == "all") && ! -d "${SYNTHETIC_DATASET}" ]]; then
  echo "ERROR: synthetic dataset for Points 3-5 not found: ${SYNTHETIC_DATASET}" >&2
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
    "  model input         = channels and normalization resolved from checkpoint" \
    "  geometry            = deterministic checkpoint training preprocessing" \
    "  checkpoint windows  = size, stride, packing and validity resolved from checkpoint" \
    "  checklist           = P3:${RUN_POINT3} P4-5:${RUN_POINT45} P6:${RUN_POINT6}" \
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

# Each Python entrypoint resolves geometry, color, normalization and mask flags
# from model_config after its imports. Shell environment is not the contract.
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
  Evaluation/checkpoint_contract.py \
  Evaluation/_eval_utils.py \
  Evaluation/eval_img_align_sw.py \
  Evaluation/sw_runner.py \
  Evaluation/quantitative_real.py \
  Evaluation/quantitative_diagnostics.py \
  Evaluation/eval_point3_training_paths.py \
  Evaluation/eval_point2.py \
  Evaluation/point2_runtime.py \
  Evaluation/point3_spatial_metrics.py \
  Evaluation/eval_point6_window_neighbors.py


# Exercise one complete visual forward under inference_mode before the expensive
# evaluation starts. This specifically catches the PyTorch-2.0 odd-head native
# MHA incompatibility and checkpoint reconstruction problems immediately.
WEIGHTS="${WEIGHTS}" python - <<'PY'
import os
import json
from Evaluation._eval_utils import load_evaluation_models
from Evaluation.checkpoint_contract import evaluation_metadata, visual_preflight

models = load_evaluation_models(
    os.environ["WEIGHTS"], device="cuda", load_text_model=False
)
print("Evaluation contract:", json.dumps(evaluation_metadata(models, os.environ["WEIGHTS"]), sort_keys=True))
print("Visual forward preflight: OK", json.dumps(visual_preflight(models), sort_keys=True))
PY

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
    --image-preprocessing training
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
    --auto-calibrate-sw "${AUTO_CALIBRATE_SW}"
    --sw-calibration-lines "${SW_CALIBRATION_LINES}"
    --sw-calibration-thresholds "${SW_CALIBRATION_THRESHOLDS}"
    --sw-calibration-gaps="${SW_CALIBRATION_GAPS}"
    --cycle-pairs "${CYCLE_PAIRS}"
    --robustness-pairs "${ROBUSTNESS_PAIRS}"
    --robustness-modes "${ROBUSTNESS_MODES}"
    --min-ink "${MIN_INK}"
  )
  if [[ -n "${INTERVAL_MANIFEST}" ]]; then
    COMMAND+=(--interval-manifest "${INTERVAL_MANIFEST}")
  fi
  "${COMMAND[@]}"

  if [[ "${RUN_POINT3}" == "1" ]]; then
    mkdir -p "${POINT3_DIR}"
    python -u -m Evaluation.eval_point3_training_paths \
      --dataset "${SYNTHETIC_DATASET}" \
      --weights "${WEIGHTS}" \
      --output-dir "${POINT3_DIR}" \
      --split test \
      --training-samples 6000 \
      --split-seed "${SPLIT_SEED}" \
      --n-samples "${POINT3_SAMPLES}" \
      --device cuda \
      --image-preprocessing training
  fi

  if [[ "${RUN_POINT45}" == "1" ]]; then
    POINT45_EVAL="${POINT45_ROOT}/fused"
    POINT45_METRICS="${POINT45_ROOT}/metrics"
    mkdir -p "${POINT45_ROOT}"
    python -u -m Evaluation.eval_point2 \
      --point2-representation fused \
      --dataset "${SYNTHETIC_DATASET}" \
      --weights "${WEIGHTS}" \
      --branch restoration \
      --alignment-unit window \
      --word-support-floor 0.0 \
      --min-aligned-windows "${POINT45_MIN_WINDOWS}" \
      --image-preprocessing training \
      --split test \
      --training-samples 6000 \
      --split-seed "${SPLIT_SEED}" \
      --n-samples "${POINT45_SAMPLES}" \
      --start-index 1 \
      --device cuda \
      --score-mode raw \
      --threshold 0.0 \
      --gap -0.30 \
      --output-dir "${POINT45_EVAL}"

    python -u Evaluation/point3_spatial_metrics.py \
      --eval-root "${POINT45_EVAL}" \
      --output-dir "${POINT45_METRICS}" \
      --min-consecutive-windows "${POINT45_MIN_WINDOWS}" \
      --min-region-iou "${POINT45_MIN_IOU}"
  fi

  if [[ "${RUN_POINT6}" == "1" ]]; then
    mkdir -p "${POINT6_DIR}"
    python -u -m Evaluation.eval_point6_window_neighbors \
      --dataset "${REAL_DATA_DIR}" \
      --weights "${WEIGHTS}" \
      --output-dir "${POINT6_DIR}" \
      --split "${REAL_SPLIT}" \
      --split-seed "${SPLIT_SEED}" \
      --labels "${LABELS}" \
      --line-pairs "${POINT6_LINE_PAIRS}" \
      --queries "${POINT6_QUERIES}" \
      --top-k "${POINT6_TOP_K}" \
      --visualize-queries "${POINT6_VISUALIZE}" \
      --min-ink "${MIN_INK}" \
      --seed "${EVAL_SEED}" \
      --device cuda \
      --image-preprocessing training
  fi
fi

echo "Evaluation complete: ${RESULTS_ROOT}"

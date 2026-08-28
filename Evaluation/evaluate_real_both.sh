#!/usr/bin/env bash
# Run Smith-Waterman and Needleman-Wunsch on the same real-style dataset split.
#
# The feature/scoring evidence is identical for both algorithms.  Only the DP
# constraint differs:
#   SW: local traceback, maximum accumulated DP cell -> zero boundary.
#   NW: global traceback, terminal (N,M) DP boundary -> origin (0,0).
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to the checkpoint to evaluate.}"
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
LABELS="${LABELS:-all}"
N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-mutual-z}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
if [[ -z "${THRESHOLD+x}" ]]; then
  case "${SCORE_MODE,,}" in
    raw) THRESHOLD=0.45 ;;
    *) THRESHOLD=0.0 ;;
  esac
fi
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-match-score}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/${MODEL_TAG}/Real_Experiments/${RUN_TAG}/SW_vs_NW}"

[[ -d "${REAL_DATA_DIR}" ]] || { echo "ERROR: dataset not found: ${REAL_DATA_DIR}" >&2; exit 2; }
[[ -f "${ARABIC_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${ARABIC_MANIFEST}" >&2; exit 2; }
[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: N_SAMPLES must be positive." >&2; exit 2; }
[[ "${START_INDEX}" =~ ^[0-9]+$ ]] || { echo "ERROR: START_INDEX must be non-negative." >&2; exit 2; }

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_${MODEL_TAG}_sw_nw}"

# Real-line preprocessing: Otsu identifies foreground; side whitespace is
# physically removed; the cropped line is resized directly instead of being
# placed back on a 1024-pixel white canvas.
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-0}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-0}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_EVAL_BALANCED="${REAL_EVAL_BALANCED:-1}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export SW_MIN_INK="${SW_MIN_INK:-0.02}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

# Shared post-trace interpretation.  These values do NOT alter either DP table.
# They only decide which diagonal matches on the algorithm's own traceback are
# rendered/scored as aligned components.  One or two missing/noisy trace cells
# may be bridged; three or more open a real hole between regions.
export TRACE_COMPONENTS="${TRACE_COMPONENTS:-1}"
export TRACE_COMPONENT_SUPPORT_FLOOR="${TRACE_COMPONENT_SUPPORT_FLOOR:-0.0}"
export TRACE_COMPONENT_MAX_BRIDGE_STEPS="${TRACE_COMPONENT_MAX_BRIDGE_STEPS:-2}"
export TRACE_COMPONENT_MAX_WINDOW_GAP="${TRACE_COMPONENT_MAX_WINDOW_GAP:-2}"
export TRACE_COMPONENT_MIN_MATCHES="${TRACE_COMPONENT_MIN_MATCHES:-3}"

# Save the underlying 63x63 evidence so pushed results can be analyzed without
# reverse-engineering values from a PNG heatmap.
export SAVE_HEATMAP_CSV="${SAVE_HEATMAP_CSV:-1}"
export ANNOTATE_HEATMAP_VALUES="${ANNOTATE_HEATMAP_VALUES:-0}"

print_config() {
  printf '%s\n' \
    "Real-style ViT SW + NW evaluation" \
    "  branch       = $(git branch --show-current 2>/dev/null || true)" \
    "  checkpoint   = ${WEIGHTS}" \
    "  dataset      = ${REAL_DATA_DIR}" \
    "  manifest     = ${ARABIC_MANIFEST}" \
    "  split        = ${REAL_SPLIT}" \
    "  labels       = ${LABELS}" \
    "  samples      = ${N_SAMPLES}" \
    "  feature      = ${FEATURE}" \
    "  score mode   = ${SCORE_MODE}" \
    "  threshold    = ${THRESHOLD}" \
    "  gap          = ${GAP}" \
    "  heatmap      = ${HEATMAP_SOURCE}" \
    "  component support floor = ${TRACE_COMPONENT_SUPPORT_FLOOR}" \
    "  component bridge steps  = ${TRACE_COMPONENT_MAX_BRIDGE_STEPS}" \
    "  component window gap    = ${TRACE_COMPONENT_MAX_WINDOW_GAP}" \
    "  component min matches   = ${TRACE_COMPONENT_MIN_MATCHES}" \
    "  save matrix CSV         = ${SAVE_HEATMAP_CSV}" \
    "  SW trace      = maximum DP -> zero" \
    "  NW trace      = terminal (N,M) -> origin (0,0)" \
    "  output       = ${RESULTS_ROOT}"
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
print_config

COMMON_ARGS=(
  --weights "${WEIGHTS}"
  --device cuda
  --data-dir "${REAL_DATA_DIR}"
  --arabic-manifest "${ARABIC_MANIFEST}"
  --dataset-type real
  --batch
  --real-split "${REAL_SPLIT}"
  --real-labels "${LABELS}"
  --real-text-key "${REAL_TEXT_KEY}"
  --real-min-text-score "${REAL_MIN_TEXT_SCORE}"
  --split-seed "${DATASET_SPLIT_SEED}"
  --start-index "${START_INDEX}"
  --n-samples "${N_SAMPLES}"
  --feature "${FEATURE}"
  --score-mode "${SCORE_MODE}"
  --score-clip "${SCORE_CLIP}"
  --threshold "${THRESHOLD}"
  --gap "${GAP}"
  --heatmap-source "${HEATMAP_SOURCE}"
)

mkdir -p "${RESULTS_ROOT}/SW" "${RESULTS_ROOT}/NW"

echo "=== Smith-Waterman: traceback maximum DP -> zero ==="
python -m Evaluation.eval_img_align_sw \
  "${COMMON_ARGS[@]}" \
  --no-save-binarized-images \
  --output-dir "${RESULTS_ROOT}/SW"

echo "=== Needleman-Wunsch: traceback terminal (N,M) -> origin (0,0) ==="
python -m Evaluation.eval_img_align_nw_real \
  "${COMMON_ARGS[@]}" \
  --output-dir "${RESULTS_ROOT}/NW"

printf '%s\n' \
  "SW + NW evaluation finished." \
  "  SW results = ${RESULTS_ROOT}/SW" \
  "  NW results = ${RESULTS_ROOT}/NW" \
  "  Each result folder also contains matrices/ and evidence/."

#!/usr/bin/env bash
# Small fixed qualitative real evaluation with saved heatmaps/tracebacks.
set -euo pipefail
set -a
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out
: "${WEIGHTS:?Set WEIGHTS to the checkpoint to inspect.}"
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: missing ${WEIGHTS}" >&2; exit 2; }

MODEL_TAG="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
RUN_TAG="${RUN_TAG:-qualitative_${MODEL_TAG}}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
LABELS="${LABELS:-high_match,medium_match,low_match,no_shared_content}"
N_PER_LABEL="${N_PER_LABEL:-4}"
REAL_SPLIT="${REAL_SPLIT:-test}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/ResearchPipeline/${RUN_TAG}}"
THRESHOLD="${THRESHOLD:-0.50}"
GAP="${GAP:--0.30}"
FEATURE="${FEATURE:-local}"
SCORE_MODE="${SCORE_MODE:-raw}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  DEP_ARGS=(); [[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS+=(--dependency="${DEPENDENCY}")
  sbatch --job-name="qual_${RUN_TAG}" --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME_LIMIT}" \
    --mail-type=ALL --mail-user="${MAIL_USER}" "${DEP_ARGS[@]}" --export=ALL "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BOX_EVAL=0 REAL_REQUIRE_BOX_ANNOTATIONS=0
export ZERO_SHOT_PREPROCESS=1 ZERO_SHOT_PRESERVE_ASPECT=1 ZERO_SHOT_FOREGROUND_CROP=1 ZERO_SHOT_SOURCE_GEOMETRY=1
mkdir -p "${RESULTS_ROOT}"

IFS=',' read -r -a LABEL_ARRAY <<< "${LABELS}"
for LABEL in "${LABEL_ARRAY[@]}"; do
  LABEL="${LABEL//[[:space:]]/}"
  [[ -n "${LABEL}" ]] || continue
  OUT="${RESULTS_ROOT}/${LABEL}"
  mkdir -p "${OUT}"
  echo "=== QUALITATIVE ${RUN_TAG}: ${LABEL} ==="
  python -m Evaluation.eval_img_align_sw \
    --weights "${WEIGHTS}" --device cuda \
    --data-dir "${REAL_DATA_DIR}" --arabic-manifest "${ARABIC_MANIFEST}" \
    --dataset-type real --batch --real-split "${REAL_SPLIT}" --real-labels "${LABEL}" \
    --start-index 1 --n-samples "${N_PER_LABEL}" \
    --feature "${FEATURE}" --score-mode "${SCORE_MODE}" --threshold "${THRESHOLD}" \
    --gap "${GAP}" --heatmap-source cosine --no-save-binarized-images \
    --output-dir "${OUT}"
done

echo "QUALITATIVE_RESULTS=${RESULTS_ROOT}"

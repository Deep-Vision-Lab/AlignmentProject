#!/usr/bin/env bash
#SBATCH --job-name=yelda_eval_hierarchy
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=out/%x_%J.out

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/manucripts_align/bin/python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/yelda_cnn_hierarchy/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
REPRESENTATIONS="${REPRESENTATIONS:-primary local}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
CHECKPOINT_NAME="$(basename "$(dirname "${WEIGHTS}")")"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/Results/Evaluation/Yelda/${CHECKPOINT_NAME}/$(basename "${DATASET}")/${EVAL_SPLIT}/${RUN_TAG}}"

cd "${PROJECT_DIR}"
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${WEIGHTS}" ]] || { echo "Checkpoint not available yet: ${WEIGHTS}" >&2; exit 2; }
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export EVALUATION_SOURCE_WEIGHTS="${WEIGHTS}"

echo "Evaluation commit: $(git rev-parse --short HEAD)"
echo "Checkpoint: ${WEIGHTS}"
echo "Dataset: ${DATASET}; split: ${EVAL_SPLIT}; results: ${RESULTS_DIR}"
# Pin a stable on-disk checkpoint for every representation, even while training
# updates model_best/model_latest. Fail if it changes during the copy.
mkdir -p "${RESULTS_DIR}"
SNAPSHOT="${RESULTS_DIR}/evaluation_checkpoint.pth"
"${PYTHON_BIN}" - "${WEIGHTS}" "${SNAPSHOT}" <<'PY'
import os
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
try:
    with source.open('rb') as src, target.open('xb') as dst:
        before = os.fstat(src.fileno())
        shutil.copyfileobj(src, dst)
        after = os.fstat(src.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        target.unlink()
        raise SystemExit('Checkpoint changed during copy. Retry after checkpoint saving finishes.')
except FileExistsError:
    raise SystemExit('This evaluation run already exists. Choose a new RUN_TAG or RESULTS_DIR.')
PY

for representation in ${REPRESENTATIONS}; do
  "${PYTHON_BIN}" -m Evaluation.eval_yelda \
    --dataset "${DATASET}" --weights "${SNAPSHOT}" \
    --branch auto --representation "${representation}" \
    --split "${EVAL_SPLIT}" --training-samples "${TRAINING_SAMPLES:-6000}" \
    --split-seed "${SPLIT_SEED:-42}" --n-samples "${N_SAMPLES:-100}" \
    --start-index "${START_INDEX:-1}" --device cuda \
    --score-mode "${SCORE_MODE:-raw}" --threshold "${THRESHOLD:-0.45}" \
    --gap "${GAP:--0.30}" --min-ink "${MIN_INK:-0.02}" \
    --output-dir "${RESULTS_DIR}/${representation}"
done

#!/usr/bin/env bash
# Component-aware Needleman-Wunsch sampling on the canonical untouched real test split.
# Reconstructs the same pair-id-safe 60/20/20 split used by real training.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> [N_SAMPLES=<n>] bash Evaluation/evaluate_nw_real.sh" >&2
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
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }

REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
REAL_DATA_DIR="$(readlink -f "${REAL_DATA_DIR}")"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
SOURCE_MANIFEST="$(readlink -f "${SOURCE_MANIFEST}")"
[[ -f "${SOURCE_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${SOURCE_MANIFEST}" >&2; exit 2; }

N_SAMPLES="${N_SAMPLES:-10}"
START_INDEX="${START_INDEX:-1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/NW/Real/${RUN_TAG}}"
TEST_MANIFEST="${RESULTS_ROOT}/canonical_test_manifest.jsonl"

[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: N_SAMPLES must be positive." >&2; exit 2; }
[[ "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: START_INDEX must be positive." >&2; exit 2; }

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
EVAL_JOB_NAME="${EVAL_JOB_NAME:-nw_real_${RUN_TAG}}"

export PROJECT_DIR WEIGHTS REAL_DATA_DIR SOURCE_MANIFEST TEST_MANIFEST RESULTS_ROOT
export N_SAMPLES START_INDEX SPLIT_SEED FEATURE SCORE_MODE SCORE_CLIP THRESHOLD GAP HEATMAP_SOURCE RUN_TAG
export LINE_HEIGHT LINE_WIDTH TARGET_INK_HEIGHT_RATIO ZERO_SHOT_TARGET_INK_HEIGHT_RATIO
export ZERO_SHOT_PREPROCESS ZERO_SHOT_PRESERVE_ASPECT ZERO_SHOT_FOREGROUND_CROP ZERO_SHOT_SOURCE_GEOMETRY
export REAL_BINARIZE REAL_BINARIZE_METHOD REAL_BINARIZE_AUTO_INVERT REAL_BINARIZE_AUTOCONTRAST
export SW_INK_AWARE SW_MIN_INK SW_BLANK_BLANK_SCORE SW_BLANK_INK_SCORE
export NW_COMPONENT_MAX_COMPONENTS NW_COMPONENT_MIN_MATCHES NW_COMPONENT_MIN_SPAN_WINDOWS
export NW_COMPONENT_MIN_SPAN_FRACTION NW_COMPONENT_WEAK_GLOBAL_SCORE
export CONDA_ENV PARTITION GPU_RESOURCE CPUS_PER_TASK MEMORY TIME_LIMIT MAIL_USER EVAL_JOB_NAME
set +a

print_config() {
  printf '%s\n' \
    "Component-aware NW evaluation on canonical real test data" \
    "  branch=$(git branch --show-current)" \
    "  checkpoint=${WEIGHTS}" \
    "  dataset=${REAL_DATA_DIR}" \
    "  source manifest=${SOURCE_MANIFEST}" \
    "  split=training-compatible pair-id-safe test" \
    "  split seed=${SPLIT_SEED}" \
    "  start=${START_INDEX}" \
    "  samples=${N_SAMPLES}" \
    "  algorithm=Needleman-Wunsch" \
    "  results=${RESULTS_ROOT}" \
    "  GPU=${GPU_RESOURCE}:1"
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
mkdir -p "${RESULTS_ROOT}"

# Reproduce DataLoader._group_split_real_dataset exactly for high/medium positives.
python - "${SOURCE_MANIFEST}" "${TEST_MANIFEST}" "${SPLIT_SEED}" <<'PY'
import json, random, sys
from collections import OrderedDict
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
allowed = {"high_match", "medium_match"}
rows = []
with open(src, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("label_type", "")) in allowed:
            rows.append((line.rstrip("\n"), row))
groups = OrderedDict()
for i, (_line, row) in enumerate(rows):
    pair_id = str(row.get("pair_id", f"sample_{i}"))
    groups.setdefault(pair_id, []).append(i)
ids = list(groups)
random.Random(seed).shuffle(ids)
train_target = int(0.6 * len(rows))
valid_target = int(0.2 * len(rows))
train, valid, test = [], [], []
for gid in ids:
    indices = groups[gid]
    if len(train) < train_target:
        train.extend(indices)
    elif len(valid) < valid_target:
        valid.extend(indices)
    else:
        test.extend(indices)
if not train or not valid or not test:
    raise SystemExit("ERROR: canonical group split produced an empty split")
with open(dst, "w", encoding="utf-8") as out:
    for i in test:
        out.write(rows[i][0] + "\n")
print(f"canonical real split: all={len(rows)} train={len(train)} valid={len(valid)} test={len(test)} seed={seed}")
PY

TEST_ROWS="$(grep -cve '^[[:space:]]*$' "${TEST_MANIFEST}" || true)"
if (( START_INDEX > TEST_ROWS )); then
  echo "ERROR: START_INDEX=${START_INDEX} exceeds test rows=${TEST_ROWS}." >&2
  exit 2
fi
AVAILABLE=$((TEST_ROWS - START_INDEX + 1))
if (( N_SAMPLES > AVAILABLE )); then N_SAMPLES="${AVAILABLE}"; fi
export N_SAMPLES TEST_ROWS

print_config
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
nvidia-smi -L || true

python -m Evaluation.eval_img_align_nw \
  --weights "${WEIGHTS}" \
  --device cuda \
  --data-dir "${REAL_DATA_DIR}" \
  --arabic-manifest "${TEST_MANIFEST}" \
  --dataset-type real \
  --batch \
  --real-split all \
  --real-labels all \
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

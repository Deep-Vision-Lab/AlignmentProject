#!/usr/bin/env bash
# Evaluate the training-speed CNN+BiLSTM checkpoint with global Needleman-Wunsch.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: [WEIGHTS=<checkpoint>] bash Evaluation/evaluate_synthetic_fixed63_nw.sh" >&2
  echo "Optional: RUN_ID, DATA_DIR, N_SAMPLES, TEST_START, RESULTS_ROOT." >&2
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

RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"
DATA_DIR="${DATA_DIR:-${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset63}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
N_SAMPLES="${N_SAMPLES:-20}"
TEST_START="${TEST_START:-1}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-raw}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/training_speed_optimization/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/test_start_${TEST_START}_count_${N_SAMPLES}}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${OUTPUT_DIR}/selected_test_pairs.jsonl}"
SELECTED_INDICES="${SELECTED_INDICES:-${OUTPUT_DIR}/selected_test_indices.json}"

for name in NUM_SAMPLES N_SAMPLES TEST_START DATASET_SPLIT_SEED; do
  value="${!name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: ${name} must be a non-negative integer." >&2
    exit 2
  }
done
(( NUM_SAMPLES > 0 && N_SAMPLES > 0 && TEST_START > 0 )) || {
  echo "ERROR: NUM_SAMPLES, N_SAMPLES, and TEST_START must be greater than zero." >&2
  exit 2
}

DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" && -d "${DATA_DIR}/masks" ]] || {
  echo "ERROR: synthetic dataset must contain images/, texts/, and masks/: ${DATA_DIR}" >&2
  exit 2
}

if [[ -z "${WEIGHTS:-}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/Weights/${RUN_ID}/model_best.pth" \
    "${PROJECT_DIR}/Weights/${RUN_ID}/model_latest.pth" \
    "${PROJECT_DIR}/Weights/${RUN_ID}/checkpoint_latest.pth"; do
    if [[ -f "${candidate}" ]]; then
      WEIGHTS="${candidate}"
      break
    fi
  done
fi
: "${WEIGHTS:?Set WEIGHTS, or place the checkpoint under Weights/${RUN_ID}.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_trainopt_nw63}"

has_gpu_allocation() {
  local name value
  for name in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS SLURM_JOB_GPUS SLURM_GPU_INDEX; do
    value="${!name:-}"
    if [[ -n "${value}" && "${value}" != "NoDevFiles" && "${value}" != "(null)" ]]; then
      return 0
    fi
  done
  return 1
}

print_config() {
  printf '%s\n' \
    "Training-speed fixed-63 synthetic Needleman-Wunsch evaluation" \
    "  branch       = $(git branch --show-current 2>/dev/null || true)" \
    "  checkpoint   = ${WEIGHTS}" \
    "  dataset      = ${DATA_DIR}" \
    "  split        = exact synthetic test split" \
    "  split seed   = ${DATASET_SPLIT_SEED}" \
    "  test start   = ${TEST_START}" \
    "  samples      = ${N_SAMPLES}" \
    "  feature      = ${FEATURE}" \
    "  score mode   = ${SCORE_MODE}" \
    "  threshold    = ${THRESHOLD}" \
    "  gap          = ${GAP}" \
    "  output       = ${OUTPUT_DIR}"
}

if ! has_gpu_allocation; then
  print_config
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Detected a CPU-only Slurm context; submitting a separate GPU evaluation job."
  fi
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --error="${PROJECT_DIR}/out/%x_%J.err" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",WEIGHTS="${WEIGHTS}",DATA_DIR="${DATA_DIR}",RUN_ID="${RUN_ID}" \
    "${SCRIPT_PATH}"
  exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load anaconda || true
fi

resolve_python() {
  local conda_base=""
  local candidate=""
  local -a candidates=()

  [[ -n "${CONDA_ENV_PYTHON:-}" ]] && candidates+=("${CONDA_ENV_PYTHON}")
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=(
    "${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
    "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
    "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python"
  )
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    [[ -n "${conda_base}" ]] && candidates+=("${conda_base}/envs/${CONDA_ENV}/bin/python")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import torch' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "ERROR: could not find a Python executable with PyTorch for ${CONDA_ENV}." >&2
  echo "Checked candidates:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  echo "Set CONDA_ENV_PYTHON to the working environment's bin/python path." >&2
  return 1
}

EVAL_PYTHON="$(resolve_python)"
export CONDA_ENV_PYTHON="${EVAL_PYTHON}"
if ! "${EVAL_PYTHON}" - <<'PY'
import sys
import torch
print(f"Evaluation Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated CUDA GPU")
PY
then
  echo "ERROR: ${EVAL_PYTHON} cannot use the allocated CUDA GPU." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
"${EVAL_PYTHON}" - \
  "${DATA_DIR}" \
  "${NUM_SAMPLES}" \
  "${DATASET_SPLIT_SEED}" \
  "${TEST_START}" \
  "${N_SAMPLES}" \
  "${PAIR_MANIFEST}" \
  "${SELECTED_INDICES}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

root = Path(sys.argv[1]).resolve()
total = int(sys.argv[2])
seed = int(sys.argv[3])
test_start = int(sys.argv[4])
count = int(sys.argv[5])
manifest_path = Path(sys.argv[6])
indices_path = Path(sys.argv[7])

train_size = int(0.6 * total)
valid_size = int(0.2 * total)
permutation = torch.randperm(
    total,
    generator=torch.Generator().manual_seed(seed),
).tolist()
test_zero_based = permutation[train_size + valid_size :]
start = test_start - 1
selected_zero_based = test_zero_based[start : start + count]
if len(selected_zero_based) != count:
    raise SystemExit(
        f"Requested {count} test samples from position {test_start}, "
        f"but only {len(selected_zero_based)} are available."
    )

records = []
selected_dataset_indices = []
for output_index, zero_based in enumerate(selected_zero_based, start=1):
    dataset_index = int(zero_based) + 1
    image1 = root / "images" / f"img1_{dataset_index}.png"
    image2 = root / "images" / f"img2_{dataset_index}.png"
    text1 = root / "texts" / f"text1_{dataset_index}.txt"
    text2 = root / "texts" / f"text2_{dataset_index}.txt"
    mask1 = root / "masks" / f"mask1_{dataset_index}.png"
    mask2 = root / "masks" / f"mask2_{dataset_index}.png"
    missing = [
        str(path)
        for path in (image1, image2, text1, text2, mask1, mask2)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit("Missing synthetic sample files: " + ", ".join(missing))
    lengths = (
        len(text1.read_text(encoding="utf-8").strip()),
        len(text2.read_text(encoding="utf-8").strip()),
    )
    if lengths != (63, 63):
        raise SystemExit(
            f"Dataset index {dataset_index} has transcript lengths {lengths}, expected (63, 63)."
        )
    records.append(
        {
            "index": output_index,
            "pair_id": f"synthetic_dataset_index_{dataset_index}",
            "label_type": "synthetic_test",
            "dataset_index": dataset_index,
            "image1": str(image1),
            "image2": str(image2),
        }
    )
    selected_dataset_indices.append(dataset_index)

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
indices_path.write_text(
    json.dumps(
        {
            "split": "test",
            "split_seed": seed,
            "total_samples": total,
            "test_start": test_start,
            "selected_dataset_indices": selected_dataset_indices,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(
    f"Selected {len(records)} exact held-out test pairs; "
    f"dataset indices={selected_dataset_indices}"
)
PY

export SYNTHETIC_MANUSCRIPT_AUGMENT=0
export REAL_AUGMENT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

print_config
"${EVAL_PYTHON}" -m Evaluation.eval_img_align_nw \
  --weights "${WEIGHTS}" \
  --device cuda \
  --data-dir "${DATA_DIR}" \
  --dataset-type synthetic \
  --pair-manifest "${PAIR_MANIFEST}" \
  --batch \
  --start-index 1 \
  --n-samples "${N_SAMPLES}" \
  --feature "${FEATURE}" \
  --score-mode "${SCORE_MODE}" \
  --score-clip "${SCORE_CLIP}" \
  --threshold "${THRESHOLD}" \
  --gap "${GAP}" \
  --heatmap-source "${HEATMAP_SOURCE}" \
  --no-save-binarized-images \
  --output-dir "${OUTPUT_DIR}"

printf '%s\n' \
  "Synthetic Needleman-Wunsch evaluation finished." \
  "  sample images = ${OUTPUT_DIR}/pair_*.png" \
  "  metrics CSV   = ${OUTPUT_DIR}/samples.csv" \
  "  summary       = ${OUTPUT_DIR}/summary.json" \
  "  test indices  = ${SELECTED_INDICES}"

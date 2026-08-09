#!/usr/bin/env bash
# Train a genuine ViT on the validated 27,000-pair fixed-63 Arabic dataset.
# This ViT-branch wrapper is intentionally strict: it refuses to submit unless
# model_backend.py resolves to ViT, uses the branch-aware runtime, and defaults
# to a clean ViT-specific output folder with a 3-day Slurm limit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="${DATA_DIR:-${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset63}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
EXPECTED_TEXT_CHARS="${EXPECTED_TEXT_CHARS:-63}"
JOB_ID="${JOB_ID:-vit_augmented_fixed63_27k_v2}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
EXPECTED_MODEL_BACKEND=vit

[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: NUM_SAMPLES must be a positive integer." >&2
  exit 2
}
[[ "${EXPECTED_TEXT_CHARS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: EXPECTED_TEXT_CHARS must be a positive integer." >&2
  exit 2
}

DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: fixed-63 dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}

python - "${DATA_DIR}/texts" "${NUM_SAMPLES}" "${EXPECTED_TEXT_CHARS}" <<'PY'
from pathlib import Path
import sys

texts_dir = Path(sys.argv[1])
num_samples = int(sys.argv[2])
expected = int(sys.argv[3])
missing = []
invalid = []
for index in range(1, num_samples + 1):
    for side in (1, 2):
        path = texts_dir / f"text{side}_{index}.txt"
        if not path.is_file():
            missing.append(path.name)
            if len(missing) >= 10:
                break
            continue
        length = len(path.read_text(encoding="utf-8").strip())
        if length != expected:
            invalid.append((path.name, length))
            if len(invalid) >= 10:
                break
    if len(missing) >= 10 or len(invalid) >= 10:
        break
if missing:
    raise SystemExit("Missing transcript files: " + ", ".join(missing))
if invalid:
    details = ", ".join(f"{name}={length}" for name, length in invalid)
    raise SystemExit(
        f"Expected every transcript to contain exactly {expected} characters; {details}"
    )
print(
    f"Fixed-length dataset validation passed: {2 * num_samples} transcripts, "
    f"each exactly {expected} characters."
)
PY

DATASET_TYPE=synthetic
SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-2}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"
SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${MAX_TEXT_SPAN_CHARS}}"
SPAN_INCLUDE_SPACE_CONTEXT=0
SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
ALLOW_UNSAFE_SPAN_CONFIG=0

[[ "${MAX_TEXT_SPAN_CHARS}" =~ ^[12]$ ]] || {
  echo "ERROR: MAX_TEXT_SPAN_CHARS must be 1 or 2; got ${MAX_TEXT_SPAN_CHARS}." >&2
  exit 2
}
[[ "${MAX_TEXT_TOKEN_CHARS}" =~ ^[12]$ ]] || {
  echo "ERROR: MAX_TEXT_TOKEN_CHARS must be 1 or 2; got ${MAX_TEXT_TOKEN_CHARS}." >&2
  exit 2
}
[[ "${MAX_WINDOWS_PER_SPAN}" =~ ^[1-3]$ ]] || {
  echo "ERROR: MAX_WINDOWS_PER_SPAN must be between 1 and 3; got ${MAX_WINDOWS_PER_SPAN}." >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
if [[ "${MODEL_BACKEND}" != "vit" ]]; then
  echo "ERROR: this launcher is ViT-only, but model_backend.py resolved to '${MODEL_BACKEND}'." >&2
  echo "Switch to agent/use-vit-encoder before submitting." >&2
  exit 2
fi

export PROJECT_DIR DATA_DIR NUM_SAMPLES EXPECTED_TEXT_CHARS JOB_ID TIME_LIMIT
export DATASET_TYPE SYNTHETIC_MANUSCRIPT_AUGMENT REAL_AUGMENT
export WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE
export MAX_TEXT_SPAN_CHARS MAX_TEXT_TOKEN_CHARS MAX_WINDOWS_PER_SPAN
export SPAN_MAX_CORE_CHARS_CAP SPAN_CONNECTED_MAX_UNITS_PER_SPAN
export SPAN_INCLUDE_SPACE_CONTEXT SPAN_ALLOW_CHARACTER_SPACE_SURFACES ALLOW_UNSAFE_SPAN_CONFIG
export MODEL_BACKEND EXPECTED_MODEL_BACKEND

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

if [[ -n "${SLURM_JOB_ID:-}" ]] && ! has_gpu_allocation; then
  echo "Detected CPU-only Slurm context ${SLURM_JOB_ID}; submitting the GPU batch job instead of starting torchrun locally."
  unset SLURM_JOB_ID SLURM_STEP_ID SLURM_STEP_GPUS SLURM_JOB_GPUS \
    SLURM_GPU_INDEX CUDA_VISIBLE_DEVICES
fi

printf '%s\n' \
  "ViT Fixed63 pretraining wrapper" \
  "  backend=${MODEL_BACKEND}" \
  "  dataset=${DATA_DIR}" \
  "  samples=${NUM_SAMPLES}" \
  "  output=${PROJECT_DIR}/Weights/${JOB_ID}" \
  "  time limit=${TIME_LIMIT}" \
  "  next stage=real augmented fine-tuning"

BRANCH_AWARE_LAUNCHER="${PROJECT_DIR}/scripts/train/run_branch_aware_synthetic_27k.sh"
[[ -f "${BRANCH_AWARE_LAUNCHER}" ]] || {
  echo "ERROR: missing branch-aware synthetic launcher: ${BRANCH_AWARE_LAUNCHER}" >&2
  exit 2
}

exec bash "${BRANCH_AWARE_LAUNCHER}"

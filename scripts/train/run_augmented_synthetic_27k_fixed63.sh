#!/usr/bin/env bash
# Train the stroke-aware branch on the validated fixed-63 synthetic dataset.
# This wrapper keeps the underlying GPU SBATCH settings unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="${DATA_DIR:-${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset63}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
EXPECTED_TEXT_CHARS="${EXPECTED_TEXT_CHARS:-63}"

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

root = Path(sys.argv[1])
count = int(sys.argv[2])
expected = int(sys.argv[3])
for index in range(1, count + 1):
    for side in (1, 2):
        path = root / f"text{side}_{index}.txt"
        if not path.is_file():
            raise SystemExit(f"Missing transcript: {path}")
        length = len(path.read_text(encoding="utf-8").strip())
        if length != expected:
            raise SystemExit(
                f"Invalid transcript length: {path.name}={length}, expected {expected}"
            )
print(f"Validated {2 * count} transcripts at exactly {expected} characters.")
PY

DATASET_TYPE=synthetic
SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${MAX_TEXT_SPAN_CHARS}}"

export DATA_DIR NUM_SAMPLES EXPECTED_TEXT_CHARS
export DATASET_TYPE SYNTHETIC_MANUSCRIPT_AUGMENT REAL_AUGMENT
export WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE
export MAX_TEXT_SPAN_CHARS SPAN_MAX_CORE_CHARS_CAP SPAN_CONNECTED_MAX_UNITS_PER_SPAN

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

# CPU interactive jobs also define SLURM_JOB_ID. Clear only stale allocation
# markers so the existing launcher submits the real GPU batch job.
if [[ -n "${SLURM_JOB_ID:-}" ]] && ! has_gpu_allocation; then
  echo "Detected CPU-only Slurm context ${SLURM_JOB_ID}; submitting the GPU batch job instead of starting torchrun locally."
  unset SLURM_JOB_ID SLURM_STEP_ID SLURM_STEP_GPUS SLURM_JOB_GPUS \
    SLURM_GPU_INDEX CUDA_VISIBLE_DEVICES
fi

exec bash "${PROJECT_DIR}/scripts/train/run_augmented_synthetic_27k.sh"

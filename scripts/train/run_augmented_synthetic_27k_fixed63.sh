#!/usr/bin/env bash
# Train the current branch on the validated 27,000-pair fixed-63 Arabic dataset.
# This wrapper does not change the underlying Slurm resource configuration.
set -euo pipefail
set -a

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

# The images already contain the generated noise and visual augmentation.
DATASET_TYPE=synthetic
SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0

# Preserve the established 63-window visual sequence.
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"

# Keep multi-character span support available for spaces and connected units.
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"
SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${MAX_TEXT_SPAN_CHARS}}"

export DATA_DIR NUM_SAMPLES EXPECTED_TEXT_CHARS
export DATASET_TYPE SYNTHETIC_MANUSCRIPT_AUGMENT REAL_AUGMENT
export WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE
export MAX_TEXT_SPAN_CHARS SPAN_MAX_CORE_CHARS_CAP SPAN_CONNECTED_MAX_UNITS_PER_SPAN

exec bash "${PROJECT_DIR}/scripts/train/run_augmented_synthetic_27k.sh"

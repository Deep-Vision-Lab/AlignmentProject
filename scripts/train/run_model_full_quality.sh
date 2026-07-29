#!/usr/bin/env bash
# Canonical full-quality launcher for the active branch model.
#
# Both canonical branches use this exact script. It reuses the proven optimized
# SLURM/NCCL/torchrun launcher and changes only the Python entry point to
# scripts/train/train_model.py. The selected visual model comes from the single
# branch-specific file model_backend.py.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_model_full_quality.sh" >&2
  echo "Override configuration through environment variables." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_LAUNCHER="${PROJECT_DIR}/scripts/train/run_span_d3tw_full_quality.sh"

[[ -f "${BASE_LAUNCHER}" ]] || {
  echo "ERROR: base launcher not found: ${BASE_LAUNCHER}" >&2
  exit 1
}
[[ -f "${PROJECT_DIR}/scripts/train/train_model.py" ]] || {
  echo "ERROR: canonical trainer not found: ${PROJECT_DIR}/scripts/train/train_model.py" >&2
  exit 1
}
[[ -f "${PROJECT_DIR}/model_backend.py" ]] || {
  echo "ERROR: model backend not found: ${PROJECT_DIR}/model_backend.py" >&2
  exit 1
}

export PROJECT_DIR
export NUM_GPUS="${NUM_GPUS:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"

# One geometry for synthetic/real train, validation, test, and evaluation.
# TARGET_INK_HEIGHT_RATIO=0.72 means roughly 92 ink pixels on a 128-pixel line.
export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export SYNTHETIC_BINARIZE="${SYNTHETIC_BINARIZE:-1}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"

TMP_LAUNCHER="$(mktemp "${TMPDIR:-/tmp}/alignment_model_launcher.XXXXXX.sh")"
cleanup() {
  rm -f "${TMP_LAUNCHER}"
}
trap cleanup EXIT

python3 - "${BASE_LAUNCHER}" "${TMP_LAUNCHER}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text(encoding="utf-8")
needle = "  train.py\n"
count = text.count(needle)
if count != 1:
    raise SystemExit(
        f"Expected exactly one TRAIN_ARGS entry {needle!r} in {source}, found {count}."
    )
text = text.replace(needle, "  scripts/train/train_model.py\n", 1)
text = text.replace(
    "Generic full-quality training launcher",
    "Canonical branch-model full-quality training launcher",
    1,
)
target.write_text(text, encoding="utf-8")
PY
chmod +x "${TMP_LAUNCHER}"

bash "${TMP_LAUNCHER}"

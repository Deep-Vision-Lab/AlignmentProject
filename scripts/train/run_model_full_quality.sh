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

# This wrapper must be run with bash from the login node. It then submits the
# generated launcher with the requested Slurm resources. Calling sbatch directly
# on this wrapper skips that submission layer and can start two ranks on one GPU.
if [[ -n "${SLURM_JOB_ID:-}" && "${ALIGNMENT_CANONICAL_SUBMISSION:-0}" != "1" ]]; then
  echo "ERROR: do not run 'sbatch scripts/train/run_model_full_quality.sh'." >&2
  echo "Run it with bash from the login node; the launcher submits itself offline." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_LAUNCHER="${PROJECT_DIR}/scripts/train/run_span_d3tw_full_quality.sh"
RANK_WRAPPER="${PROJECT_DIR}/scripts/train/run_rank_isolated.sh"

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
[[ -f "${RANK_WRAPPER}" ]] || {
  echo "ERROR: rank GPU wrapper is missing: ${RANK_WRAPPER}" >&2
  exit 1
}

export PROJECT_DIR
export ALIGNMENT_CANONICAL_SUBMISSION=1
export NUM_GPUS="${NUM_GPUS:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"

# Validated span semantics used by training_optimizations.py. Keep these defaults
# here so the older generic launcher cannot reintroduce its obsolete unsafe
# values. Explicit smaller values remain supported; unsafe ablations must opt in
# with ALLOW_UNSAFE_SPAN_CONFIG=1.
export MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-2}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"
export SPAN_INCLUDE_SPACE_CONTEXT="${SPAN_INCLUDE_SPACE_CONTEXT:-0}"
export SPAN_ALLOW_CHARACTER_SPACE_SURFACES="${SPAN_ALLOW_CHARACTER_SPACE_SURFACES:-0}"

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

entry_needle = "  train.py\n"
if text.count(entry_needle) != 1:
    raise SystemExit(
        f"Expected exactly one TRAIN_ARGS entry {entry_needle!r} in {source}."
    )
text = text.replace(entry_needle, "  scripts/train/train_model.py\n", 1)

old_torchrun = (
    "  exec torchrun \\\n"
    "    --standalone \\\n"
    "    --nnodes=1 \\\n"
    "    --nproc_per_node=\"${NUM_GPUS}\" \\\n"
    "    --max_restarts=0 \\\n"
    "    \"${TRAIN_ARGS[@]}\"\n"
)
new_torchrun = (
    "  exec torchrun \\\n"
    "    --standalone \\\n"
    "    --nnodes=1 \\\n"
    "    --nproc_per_node=\"${NUM_GPUS}\" \\\n"
    "    --max_restarts=0 \\\n"
    "    --no_python \\\n"
    "    bash \\\n"
    "    \"${PROJECT_DIR}/scripts/train/run_rank_isolated.sh\" \\\n"
    "    \"${TRAIN_ARGS[@]}\"\n"
)
if text.count(old_torchrun) != 1:
    raise SystemExit(f"Expected exactly one canonical torchrun block in {source}.")
text = text.replace(old_torchrun, new_torchrun, 1)
text = text.replace(
    "Generic full-quality training launcher",
    "Canonical branch-model full-quality training launcher",
    1,
)
target.write_text(text, encoding="utf-8")
PY
chmod +x "${TMP_LAUNCHER}"

bash "${TMP_LAUNCHER}"

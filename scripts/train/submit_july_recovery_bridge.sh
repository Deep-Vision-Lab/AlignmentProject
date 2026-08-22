#!/usr/bin/env bash
# Safe submission wrapper for run_july_recovery_bridge.sh.
#
# Recurrent cluster/runtime hazards are handled here BEFORE Slurm is asked for GPUs:
# 1) pin PROJECT_DIR because Slurm executes a copied spool script;
# 2) resolve one complete offline AraBERT snapshot;
# 3) derive Span-DTW image-window capacity from the exact recovery geometry so
#    feasibility filtering cannot retain samples that the active lattice cannot fit.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
export PROJECT_DIR="${ROOT}"

ARABIC_TEXT_MODEL_ID="${ARABIC_TEXT_MODEL_ID:-${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}}"
HF_RESOLUTION="$(python - "${ROOT}" "${ARABIC_TEXT_MODEL_ID}" <<'PY'
import sys
from hf_offline_runtime import resolve_hf_model_snapshot
root, model_id = sys.argv[1:3]
resolved = resolve_hf_model_snapshot(model_id, project_dir=root)
print(str(resolved.cache_root))
print(str(resolved.snapshot_path))
PY
)"
HF_HOME="$(printf '%s\n' "${HF_RESOLUTION}" | sed -n '1p')"
ARABIC_TEXT_MODEL_RESOLVED_PATH="$(printf '%s\n' "${HF_RESOLUTION}" | sed -n '2p')"
[[ -n "${HF_HOME}" && -n "${ARABIC_TEXT_MODEL_RESOLVED_PATH}" ]] || {
  echo "ERROR: offline AraBERT resolver returned an empty path." >&2
  exit 2
}

export ARABIC_TEXT_MODEL_ID
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_ID}"
export ARABIC_TEXT_MODEL_RESOLVED_PATH
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE || true

echo "hf_preflight_ok model_id=${ARABIC_TEXT_MODEL_ID}"
echo "hf_preflight_ok snapshot=${ARABIC_TEXT_MODEL_RESOLVED_PATH}"
echo "hf_preflight_ok cache_root=${HF_HOME}"

# Historical recovery geometry: 1024px line, 32px windows, stride ratio 0.5.
# Derive the exact lattice capacity instead of inheriting the old bridge default
# of 125 windows from the stride-8 profile.
RECOVERY_LINE_WIDTH="${LINE_WIDTH:-1024}"
RECOVERY_WINDOW_SIZE="${WINDOW_SIZE:-32}"
RECOVERY_STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
REAL_MAX_ALIGNMENT_WINDOWS="$(python - "${RECOVERY_LINE_WIDTH}" "${RECOVERY_WINDOW_SIZE}" "${RECOVERY_STRIDE_RATIO}" <<'PY'
import sys
line_width = int(sys.argv[1])
window_size = int(sys.argv[2])
stride_ratio = float(sys.argv[3])
if line_width <= 0 or window_size <= 0 or stride_ratio <= 0:
    raise SystemExit("ERROR: line width, window size, and stride ratio must be positive")
if window_size > line_width:
    raise SystemExit(
        f"ERROR: window_size={window_size} exceeds line_width={line_width}"
    )
stride = max(1, int(window_size * stride_ratio))
windows = ((line_width - window_size) // stride) + 1
print(windows)
PY
)"
export LINE_WIDTH="${RECOVERY_LINE_WIDTH}"
export WINDOW_SIZE="${RECOVERY_WINDOW_SIZE}"
export STRIDE_RATIO="${RECOVERY_STRIDE_RATIO}"
export REAL_MAX_ALIGNMENT_WINDOWS

echo "span_feasibility_preflight line_width=${LINE_WIDTH} window=${WINDOW_SIZE} stride_ratio=${STRIDE_RATIO} max_image_windows=${REAL_MAX_ALIGNMENT_WINDOWS} max_span_chars=${MAX_TEXT_SPAN_CHARS:-2}"

exec bash "${ROOT}/scripts/train/run_july_recovery_bridge.sh"

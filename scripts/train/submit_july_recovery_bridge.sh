#!/usr/bin/env bash
# Safe submission wrapper for run_july_recovery_bridge.sh.
#
# Two recurrent cluster hazards are handled here BEFORE Slurm is asked for GPUs:
# 1) Slurm executes a copied spool script, so PROJECT_DIR must be pinned to the
#    real checkout rather than inferred from BASH_SOURCE on the compute node.
# 2) Hugging Face is offline on compute nodes, so resolve one complete AraBERT
#    snapshot (config + weights + tokenizer) and force Transformers to load that
#    exact local directory instead of relying on ambiguous cache lookup.
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
export ARABIC_TEXT_MODEL_RESOLVED_PATH
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_RESOLVED_PATH}"
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE || true

echo "hf_preflight_ok model_id=${ARABIC_TEXT_MODEL_ID}"
echo "hf_preflight_ok snapshot=${ARABIC_TEXT_MODEL_RESOLVED_PATH}"
echo "hf_preflight_ok cache_root=${HF_HOME}"

exec bash "${ROOT}/scripts/train/run_july_recovery_bridge.sh"

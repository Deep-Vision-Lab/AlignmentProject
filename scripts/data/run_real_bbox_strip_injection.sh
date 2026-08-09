#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${CONDA_ENV_PYTHON:-python}"

exec "${PYTHON_BIN}" "${ROOT}/scripts/data/run_real_bbox_strip_injection.py" "$@"

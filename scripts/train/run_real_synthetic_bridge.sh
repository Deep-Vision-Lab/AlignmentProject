#!/usr/bin/env bash
# Public bridge-training command. Architecture is selected by model_backend.py.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
export JOB_ID="${JOB_ID:-$(git -C "${PROJECT_DIR}" branch --show-current | tr '/' '-')-real-synthetic-bridge-v2}"
# V2 positives intentionally contain unrelated distractor regions. Do not apply the
# generic whole-image positive sequence-ranking objective to them; the bridge-specific
# text loss ranks only the actual shared islands. The mask is exposed for a separate
# future mask-aware ablation rather than mixing another new loss into this first test.
export USE_SEQUENCE_ALIGNMENT_RANKING="${USE_SEQUENCE_ALIGNMENT_RANKING:-0}"
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.0}"
exec bash "${PROJECT_DIR}/scripts/train/run_branch_real_synthetic_bridge.sh"

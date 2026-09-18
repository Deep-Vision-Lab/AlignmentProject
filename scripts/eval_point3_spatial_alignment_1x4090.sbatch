#!/usr/bin/env bash
# Deprecated internal launcher.
#
# Astra's evaluation protocol uses one public entrypoint only:
#   WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
#   EVAL_MODE=quantitative \
#   bash Evaluation/evaluate.sh
#
# This file is intentionally kept only so old notes/commands fail loudly rather
# than running the obsolete full-image-resize Point-3 path.
set -euo pipefail

cat >&2 <<'EOF'
ERROR: scripts/eval_point3_spatial_alignment_1x4090.sbatch is deprecated.

Use the single public evaluation launcher instead:

  WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
  EVAL_MODE=quantitative \
  bash Evaluation/evaluate.sh

Evaluation/evaluate.sh now runs Points 3-6 with the checkpoint training
geometry, validation-only SW calibration, source-coordinate spatial scoring,
whole-line/empty controls, and the Point-6 local/context/fused neighbor test.
EOF
exit 2

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Running restoration recommendation point 08: 08_reconstruction_contrastive"
exec "$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_08_reconstruction_and_negative_margin_losses_both_train"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Running restoration recommendation point 10: 10_dtw_recompute"
exec "$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_10_dtw_path_is_recomputed_from_current_cost_matrix"

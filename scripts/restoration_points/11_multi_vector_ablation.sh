#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Running restoration recommendation point 11: 11_multi_vector_ablation"
exec "$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_11_multiple_spatial_vectors_per_window_are_available_as_ablation"

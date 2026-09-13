#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Running restoration recommendation point 04: 04_preserve_fine_detail"
exec "$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_04_encoder_preserves_horizontal_spatial_detail"

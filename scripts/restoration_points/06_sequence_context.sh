#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Running restoration recommendation point 06: 06_sequence_context"
exec "$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_06_transformer_context_changes_a_window_when_neighbor_changes"

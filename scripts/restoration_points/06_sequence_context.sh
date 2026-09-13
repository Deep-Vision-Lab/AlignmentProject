#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$ROOT/Results/Diagnostics/restoration_points/point_06"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Restoration recommendation point 06"
echo "Results: $OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_06_transformer_context_changes_a_window_when_neighbor_changes" | tee "$OUT_DIR/test_result.txt"
"$PYTHON_BIN" tools/restoration_point_visualize.py --point 6 | tee "$OUT_DIR/visual_result.txt"

echo
echo "PASS point 06"
echo "Inspect the files yourself in:"
echo "  $OUT_DIR"
echo "Start with:"
echo "  $OUT_DIR/summary.txt"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$ROOT/Results/Diagnostics/restoration_points/point_11"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Restoration recommendation point 11"
echo "Results: $OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_11_multiple_spatial_vectors_per_window_are_available_as_ablation" | tee "$OUT_DIR/test_result.txt"
"$PYTHON_BIN" tools/restoration_point_visualize.py --point 11 | tee "$OUT_DIR/visual_result.txt"

echo
echo "PASS point 11"
echo "Inspect the files yourself in:"
echo "  $OUT_DIR"
echo "Start with:"
echo "  $OUT_DIR/summary.txt"

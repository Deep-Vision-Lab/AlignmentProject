#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$ROOT/Results/Diagnostics/restoration_points/point_13"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Restoration recommendation point 13"
echo "Results: $OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" -m pytest -q "tests/test_restoration_recommendation_points.py::test_point_13_primary_evaluation_representation_is_image_only_fused_vector" | tee "$OUT_DIR/test_result.txt"
"$PYTHON_BIN" tools/restoration_point_visualize.py --point 13 | tee "$OUT_DIR/visual_result.txt"

echo
echo "PASS point 13"
echo "Inspect the files yourself in:"
echo "  $OUT_DIR"
echo "Start with:"
echo "  $OUT_DIR/summary.txt"

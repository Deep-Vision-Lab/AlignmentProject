#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$ROOT/Results/Diagnostics/restoration_points/point_12"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Restoration recommendation point 12"
echo "Results: $OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" tools/restoration_tiny_overfit.py | tee "$OUT_DIR/test_result.txt"

echo
echo "Inspect the result yourself in:"
echo "  $OUT_DIR"
echo "Start with:"
echo "  $OUT_DIR/summary.txt"

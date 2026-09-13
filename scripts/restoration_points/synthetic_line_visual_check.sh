#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="$ROOT/Results/Diagnostics/restoration_points/synthetic_line"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Real synthetic-line restoration preprocessing check"
echo "No flags required"
echo "Results: $OUT_DIR"
echo "============================================================"

"$PYTHON_BIN" tools/restoration_synthetic_line_visual.py | tee "$OUT_DIR/run_log.txt"

echo
echo "Open these files:"
echo "  $OUT_DIR/01_original_synthetic_line.png"
echo "  $OUT_DIR/02_detected_crop_on_real_line.png"
echo "  $OUT_DIR/03_cropped_resized_padded_line.png"
echo "  $OUT_DIR/04_window_boundaries.png"
echo "  $OUT_DIR/05_windows_physical_left_to_right.png"
echo "  $OUT_DIR/06_windows_arabic_logical_right_to_left.png"
echo "  $OUT_DIR/summary.txt"

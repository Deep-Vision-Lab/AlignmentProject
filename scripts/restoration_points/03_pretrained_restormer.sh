#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

RESTORMER_ARCH="$ROOT/third_party/Restormer/basicsr/models/archs/restormer_arch.py"
RESTORMER_WEIGHTS="$ROOT/Weights/Pretrained/Restormer/real_denoising.pth"

if [[ ! -f "$RESTORMER_ARCH" || ! -f "$RESTORMER_WEIGHTS" ]]; then
  echo "ERROR: Point 03 requires the official Restormer source and pretrained weights." >&2
  echo >&2
  echo "Run this first:" >&2
  echo "  bash scripts/setup_restormer_pretrained.sh" >&2
  echo >&2
  echo "Expected source:" >&2
  echo "  $RESTORMER_ARCH" >&2
  echo "Expected weights:" >&2
  echo "  $RESTORMER_WEIGHTS" >&2
  exit 2
fi

echo "Running restoration point 03 on ACTUAL synthetic data"
echo "Restormer source : $RESTORMER_ARCH"
echo "Restormer weights: $RESTORMER_WEIGHTS"
exec "$PYTHON_BIN" tools/restoration_synthetic_point.py --point 3

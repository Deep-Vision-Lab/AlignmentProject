#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Running restoration point 04 on ACTUAL synthetic data"
exec "$PYTHON_BIN" tools/restoration_synthetic_point.py --point 4

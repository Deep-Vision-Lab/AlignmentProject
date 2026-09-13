#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
for script in scripts/restoration_points/[0-1][0-9]_*.sh; do
  [[ "$(basename "$script")" == "run_all.sh" ]] && continue
  echo "============================================================"
  echo "$script"
  bash "$script"
done

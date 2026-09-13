#!/usr/bin/env bash
# Run all 13 restoration recommendation checks on the same actual synthetic pair.
# Continue after failures so one bad stage does not hide later evidence.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="$ROOT/Results/Diagnostics/restoration_points"
STATUS="$OUT/synthetic_all_status.tsv"
mkdir -p "$OUT"
printf "point\tstatus\texit_code\n" > "$STATUS"

failed=0
for point in $(seq -w 1 13); do
  script="$(find scripts/restoration_points -maxdepth 1 -type f -name "${point}_*.sh" | head -1)"
  if [[ -z "$script" ]]; then
    printf "%s\tMISSING\t127\n" "$point" >> "$STATUS"
    echo "[MISSING] point $point"
    failed=$((failed + 1))
    continue
  fi

  echo
  echo "======================================================================"
  echo "SYNTHETIC POINT $point: $script"
  echo "======================================================================"

  bash "$script"
  code=$?
  if [[ "$code" -eq 0 ]]; then
    printf "%s\tPASS\t0\n" "$point" >> "$STATUS"
  else
    printf "%s\tFAIL\t%s\n" "$point" "$code" >> "$STATUS"
    failed=$((failed + 1))
  fi
done

echo
echo "======================================================================"
echo "ALL SYNTHETIC POINTS FINISHED"
echo "status file: $STATUS"
echo "failed points: $failed"
echo "======================================================================"

# Keep the overall command successful so the user can inspect every produced
# diagnostic even when one point intentionally identifies a failure.
exit 0

#!/usr/bin/env bash
# Run every restoration diagnostic on ACTUAL synthetic data and keep going after failures.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -d "$ROOT/DataSet/Synthetic63/images" ]]; then
  DATASET="$ROOT/DataSet/Synthetic63"
elif [[ -d "$ROOT/DataSet/Synthetic_Arabic/images" ]]; then
  DATASET="$ROOT/DataSet/Synthetic_Arabic"
else
  DATASET="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
for p in sorted(Path("DataSet").glob("Synthetic*/images")):
    if p.is_dir():
        print(p.parent.resolve())
        break
PY
)"
fi

if [[ -z "${DATASET:-}" || ! -d "$DATASET/images" ]]; then
  echo "ERROR: No synthetic dataset with an images/ directory was found." >&2
  exit 2
fi

INDEX="$("$PYTHON_BIN" - "$DATASET" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
def complete(i):
    images = root / "images"
    texts = root / "texts"
    suffixes = (".png",".jpg",".jpeg",".tif",".tiff")
    has1 = any((images / f"img1_{i}{s}").is_file() for s in suffixes)
    has2 = any((images / f"img2_{i}{s}").is_file() for s in suffixes)
    return has1 and has2 and (texts/f"text1_{i}.txt").is_file() and (texts/f"text2_{i}.txt").is_file()
if complete(132):
    print(132)
    raise SystemExit
indices = []
for path in (root/"images").iterdir():
    match = re.match(r"img1_(\d+)\.", path.name)
    if match:
        indices.append(int(match.group(1)))
for i in sorted(set(indices)):
    if complete(i):
        print(i)
        raise SystemExit
raise SystemExit("No complete img1/img2/text1/text2 synthetic pair found")
PY
)" || exit 2

OUT="$ROOT/Results/Diagnostics/restoration_points/synthetic_suite/index_$INDEX"
rm -rf "$OUT"
mkdir -p "$OUT"
STATUS_FILE="$OUT/statuses.tsv"
printf "step\tstatus\texit_code\n" > "$STATUS_FILE"

DEVICE="$("$PYTHON_BIN" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"

echo "======================================================================"
echo "FULL SYNTHETIC RESTORATION DIAGNOSTIC SUITE"
echo "dataset : $DATASET"
echo "index   : $INDEX"
echo "device  : $DEVICE"
echo "output  : $OUT"
echo "======================================================================"

record_status() {
  local name="$1"
  local code="$2"
  local status="PASS"
  if [[ "$code" -ne 0 ]]; then
    status="FAIL"
  fi
  printf "%s\t%s\t%s\n" "$name" "$status" "$code" >> "$STATUS_FILE"
  echo "[$status] $name (exit=$code)"
}

run_logged() {
  local name="$1"
  shift
  local log="$OUT/$name.log"
  echo
  echo "----------------------------------------------------------------------"
  echo "RUN: $name"
  echo "----------------------------------------------------------------------"
  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  record_status "$name" "$code"
  return 0
}

export SYNTHETIC_DIAG_DATASET="$DATASET"
export SYNTHETIC_DIAG_INDEX="$INDEX"
run_logged "00_real_synthetic_preprocessing"   bash scripts/restoration_points/synthetic_line_visual_check.sh

PREP_SOURCE="$ROOT/Results/Diagnostics/restoration_points/synthetic_line"
if [[ -d "$PREP_SOURCE" ]]; then
  cp -a "$PREP_SOURCE" "$OUT/preprocessing"
fi

for script in   01_crop_geometry.sh   02_original_window_target.sh   03_pretrained_restormer.sh   04_preserve_fine_detail.sh   05_feature_dependency.sh   06_sequence_context.sh   07_local_context_fusion.sh   08_reconstruction_contrastive.sh   09_dtw_transitions.sh   10_dtw_recompute.sh   11_multi_vector_ablation.sh   12_small_overfit.sh   13_image_only_evaluation.sh
do
  point="${script%%_*}"
  run_logged "structural_point_$point"     bash "$ROOT/scripts/restoration_points/$script"
done

run_logged "real_synthetic_tiny_overfit"   "$PYTHON_BIN" tools/restoration_synthetic_tiny_overfit.py     --dataset "$DATASET"     --index "$INDEX"     --output-dir "$OUT/real_tiny_overfit"

STAGE_A="$ROOT/Weights/restore_rgb_pretrain_s16/model_best.pth"
STAGE_B="$ROOT/Weights/restore_fused_rgb_dtw_s16/model_best.pth"
LEGACY="$ROOT/Weights/vit_restore_dtw_s16/model_best.pth"

diagnose_modern_checkpoint() {
  local label="$1"
  local weights="$2"

  if [[ ! -f "$weights" ]]; then
    printf "%s\tSKIP\t0\n" "$label-checkpoint-missing" >> "$STATUS_FILE"
    echo "[SKIP] $label checkpoint missing: $weights"
    return 0
  fi

  for side in 1 2; do
    run_logged "$label-side$side"       "$PYTHON_BIN" -u -m Evaluation.analyze_restoration_line         --weights "$weights"         --dataset "$DATASET"         --index "$INDEX"         --side "$side"         --device "$DEVICE"         --image-preprocessing training         --output-dir "$OUT/$label-side$side"
  done

  run_logged "$label-pair"     "$PYTHON_BIN" -u tools/restoration_synthetic_pair_diagnostic.py       --dataset "$DATASET"       --weights "$weights"       --index "$INDEX"       --device "$DEVICE"       --output-dir "$OUT/$label-pair"
}

diagnose_modern_checkpoint "stage_a" "$STAGE_A"
diagnose_modern_checkpoint "stage_b" "$STAGE_B"

if [[ -f "$LEGACY" ]]; then
  for side in 1 2; do
    run_logged "legacy-side$side"       "$PYTHON_BIN" -u -m Evaluation.analyze_restoration_line         --weights "$LEGACY"         --dataset "$DATASET"         --index "$INDEX"         --side "$side"         --device "$DEVICE"         --image-preprocessing training         --output-dir "$OUT/legacy-side$side"
  done
else
  printf "legacy-checkpoint-missing\tSKIP\t0\n" >> "$STATUS_FILE"
  echo "[SKIP] legacy checkpoint missing: $LEGACY"
fi

for label in stage_a stage_b; do
  for side in 1 2; do
    if [[ -d "$OUT/$label-side$side" ]]; then
      mv "$OUT/$label-side$side" "$OUT/${label}_side$side"
    fi
  done
  if [[ -d "$OUT/$label-pair" ]]; then
    mv "$OUT/$label-pair" "$OUT/${label}_pair"
  fi
done
for side in 1 2; do
  if [[ -d "$OUT/legacy-side$side" ]]; then
    mv "$OUT/legacy-side$side" "$OUT/legacy_side$side"
  fi
done

run_logged "99_build_diagnosis"   "$PYTHON_BIN" tools/restoration_synthetic_suite_report.py --root "$OUT"

echo
echo "======================================================================"
echo "SYNTHETIC DIAGNOSTIC SUITE FINISHED"
echo "======================================================================"
echo "Main report:"
echo "  $OUT/DIAGNOSIS.md"
echo
echo "Status of every test:"
echo "  $OUT/statuses.tsv"
echo
echo "Most useful visual folders:"
echo "  $OUT/preprocessing/"
echo "  $OUT/real_tiny_overfit/"
echo "  $OUT/stage_a_side1/"
echo "  $OUT/stage_b_side1/"
echo "  $OUT/stage_b_pair/"
echo "  $OUT/legacy_side1/"
echo
echo "Read DIAGNOSIS.md first, then inspect the PNGs for the first stage it flags."

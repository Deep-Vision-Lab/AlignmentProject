#!/usr/bin/env bash
# Download and verify the official Restormer code + pretrained RGB checkpoint.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESTORMER_DIR="${RESTORMER_DIR:-$ROOT/third_party/Restormer}"
WEIGHT_DIR="${WEIGHT_DIR:-$ROOT/Weights/Pretrained/Restormer}"
WEIGHT_FILE="$WEIGHT_DIR/real_denoising.pth"
RESTORMER_REPO="https://github.com/swz30/Restormer.git"
WEIGHT_URL="https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"

echo "======================================================================"
echo "SETUP OFFICIAL RESTORMER"
echo "code    : $RESTORMER_DIR"
echo "weights : $WEIGHT_FILE"
echo "======================================================================"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required." >&2
  exit 2
fi

if [[ -f "$RESTORMER_DIR/basicsr/models/archs/restormer_arch.py" ]]; then
  echo "[OK] Restormer source already exists."
else
  if [[ -e "$RESTORMER_DIR" ]]; then
    echo "ERROR: $RESTORMER_DIR exists but is not a valid Restormer checkout." >&2
    echo "Move/remove that directory first; this script will not delete it." >&2
    exit 3
  fi
  mkdir -p "$(dirname "$RESTORMER_DIR")"
  echo "[DOWNLOAD] Cloning official Restormer source..."
  git clone --depth 1 "$RESTORMER_REPO" "$RESTORMER_DIR"
fi

mkdir -p "$WEIGHT_DIR"

download_weight() {
  local tmp="$WEIGHT_FILE.part"
  rm -f "$tmp"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2       "$WEIGHT_URL"       -o "$tmp"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 -O "$tmp" "$WEIGHT_URL"
  else
    echo "ERROR: curl or wget is required to download the checkpoint." >&2
    exit 4
  fi
  mv "$tmp" "$WEIGHT_FILE"
}

if [[ -f "$WEIGHT_FILE" ]]; then
  size="$(wc -c < "$WEIGHT_FILE" | tr -d ' ')"
  if [[ "$size" -ge 90000000 ]]; then
    echo "[OK] Pretrained checkpoint already exists ($size bytes)."
  else
    echo "[WARN] Existing checkpoint looks incomplete ($size bytes); re-downloading."
    download_weight
  fi
else
  echo "[DOWNLOAD] Downloading official Restormer real-denoising checkpoint..."
  download_weight
fi

echo
echo "[VERIFY] Checking Python dependencies and strict checkpoint compatibility..."

"$PYTHON_BIN" - "$RESTORMER_DIR" "$WEIGHT_FILE" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
weights = Path(sys.argv[2]).resolve()

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: PyTorch is unavailable: {exc}")

try:
    import einops  # noqa: F401
except Exception:
    raise SystemExit(
        "ERROR: Restormer requires einops. Install it in this environment with:\n"
        "  python -m pip install einops"
    )

sys.path.insert(0, str(repo))
from basicsr.models.archs.restormer_arch import Restormer

model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="BiasFree",
    dual_pixel_task=False,
)

payload = torch.load(weights, map_location="cpu", weights_only=False)
if not isinstance(payload, dict):
    raise SystemExit("ERROR: unexpected checkpoint payload type")

state = payload.get("params", payload.get("state_dict", payload))
if not isinstance(state, dict):
    raise SystemExit("ERROR: no state dictionary found in checkpoint")

cleaned = {
    str(k).removeprefix("module."): v
    for k, v in state.items()
    if torch.is_tensor(v)
}

model.load_state_dict(cleaned, strict=True)
params = sum(p.numel() for p in model.parameters())

print("[PASS] Official Restormer checkpoint loaded strictly.")
print(f"       architecture file : {repo / 'basicsr/models/archs/restormer_arch.py'}")
print(f"       checkpoint        : {weights}")
print(f"       parameter count   : {params:,}")
print("       input/output      : RGB 3-channel -> RGB 3-channel")
print("       LayerNorm         : BiasFree")
PY

echo
echo "======================================================================"
echo "RESTORMER SETUP COMPLETE"
echo "======================================================================"
echo "Now run:"
echo "  bash scripts/restoration_points/03_pretrained_restormer.sh"

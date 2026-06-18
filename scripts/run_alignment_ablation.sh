#!/usr/bin/env bash
# run_alignment_ablation.sh
# =========================
# Runs five training experiments that isolate the contribution of each
# component of the Arabic line-alignment model.
#
# Each experiment writes to a unique JOB_ID subdirectory under Weights/.
# Set CHECKPOINT_BASE if you want to warm-start from an existing checkpoint;
# leave it empty to train from scratch.
#
# Usage:
#   bash scripts/run_alignment_ablation.sh [--checkpoint <path>] [--epochs <N>]
#
# Environment overrides (all optional):
#   CUDA_VISIBLE_DEVICES  — GPU(s) to use (default: 0)
#   DATA_DIR              — dataset root (default: DataSet/Synthetic_Arabic_100000)
#   OUTPUT_DIR            — figure output dir (default: paper_figures/outputs_ablation)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CHECKPOINT_BASE=""
EPOCHS=30
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic_100000}"
OUTPUT_DIR="${OUTPUT_DIR:-paper_figures/outputs_ablation}"
DEVICE="cuda"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT_BASE="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2";          shift 2 ;;
        --device)     DEVICE="$2";          shift 2 ;;
        --output_dir) OUTPUT_DIR="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TRAIN="python train.py"
COMMON="--epochs $EPOCHS --device $DEVICE"
[[ -n "$CHECKPOINT_BASE" ]] && COMMON="$COMMON --checkpoint $CHECKPOINT_BASE"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Arabic Alignment Ablation Suite"
echo "  epochs=$EPOCHS  device=$DEVICE"
echo "  output_dir=$OUTPUT_DIR"
echo "============================================================"

# ── Experiment 1: Baseline — margin loss, fasttext embedder, mixed negatives ──
echo ""
echo "[1/5] Baseline: margin loss + fasttext + mixed negatives"
JOB_ID="ablation_01_baseline"
TEXT_EMBEDDER=fasttext \
NEGATIVE_MODE=mixed \
LOSS_TYPE=margin \
    $TRAIN $COMMON --job_id "$JOB_ID"

# ── Experiment 2: Orthogonal char embedder ─────────────────────────────────
echo ""
echo "[2/5] Orthogonal char embedder (no linguistic bias)"
JOB_ID="ablation_02_orthogonal_char"
TEXT_EMBEDDER=orthogonal_char \
NEGATIVE_MODE=length_controlled \
LOSS_TYPE=margin \
    $TRAIN $COMMON --job_id "$JOB_ID" --text_embedder orthogonal_char

# ── Experiment 3: Length-controlled negatives ──────────────────────────────
echo ""
echo "[3/5] Length-controlled negatives (no length bias in DTW)"
JOB_ID="ablation_03_length_ctrl_neg"
TEXT_EMBEDDER=fasttext \
NEGATIVE_MODE=length_controlled \
LOSS_TYPE=margin \
    $TRAIN $COMMON --job_id "$JOB_ID" --negative_mode length_controlled

# ── Experiment 4: InfoNCE loss ─────────────────────────────────────────────
echo ""
echo "[4/5] InfoNCE loss (cross-entropy over DTW costs)"
JOB_ID="ablation_04_infonce"
TEXT_EMBEDDER=fasttext \
NEGATIVE_MODE=length_controlled \
LOSS_TYPE=infonce \
    $TRAIN $COMMON --job_id "$JOB_ID" --negative_mode length_controlled --loss_type infonce

# ── Experiment 5: Full system — hybrid loss + orthogonal + dot-confusion ──
echo ""
echo "[5/5] Full system: hybrid loss + orthogonal_char + dot_confusion negatives"
JOB_ID="ablation_05_full_system"
TEXT_EMBEDDER=orthogonal_char \
NEGATIVE_MODE=dot_confusion \
LOSS_TYPE=hybrid \
    $TRAIN $COMMON --job_id "$JOB_ID" \
    --text_embedder orthogonal_char \
    --negative_mode dot_confusion \
    --loss_type hybrid

# ── Generate comparison figures ────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Generating diagnostic figures for each ablation checkpoint"
echo "============================================================"

for SUFFIX in \
    "ablation_01_baseline" \
    "ablation_02_orthogonal_char" \
    "ablation_03_length_ctrl_neg" \
    "ablation_04_infonce" \
    "ablation_05_full_system"
do
    CKPT="Weights/$SUFFIX/model_latest.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo "  [skip] $CKPT not found"
        continue
    fi
    FIGDIR="$OUTPUT_DIR/$SUFFIX"
    mkdir -p "$FIGDIR"
    echo "  Figures for $SUFFIX → $FIGDIR"
    python paper_figures/fig04_d3tw_path_overlay.py \
        --checkpoint "$CKPT" --output_dir "$FIGDIR" --device "$DEVICE" || true
    python paper_figures/fig05_pos_neg_costs.py \
        --checkpoint "$CKPT" --output_dir "$FIGDIR" --device "$DEVICE" \
        --num_samples 10 --negative_mode length_controlled || true
done

echo ""
echo "============================================================"
echo " Ablation complete.  Outputs: $OUTPUT_DIR"
echo "============================================================"

"""
generate_all_figures.py
========================
Master script — generates a curated subset of paper figures in one command.

Always generated (no additional checkpoints required):
  fig01  Architecture diagram
  fig02  Sliding-window decomposition

Requires --checkpoint:
  fig03  Similarity before vs after training
  fig04  D3TW alignment path overlay
  fig05  Positive vs negative DTW cost distribution
  fig08  Hard negative case study
  fig09  Embedding space (t-SNE)
  fig10  Image-only alignment (if --sample_idx_b is provided)
  fig11  Success / partial / failure cases

Optional:
  fig06  Ablation study          (requires --ablation_csv)
  fig07  CNN vs CNN+BiLSTM       (requires --checkpoint_bilstm; --checkpoint_cnn_only optional)

Usage
-----
python paper_figures/generate_all_figures.py \\
    --checkpoint  Weights/localRun/model_latest.pth \\
    --data_dir    DataSet/Synthetic_Arabic_100000 \\
    --output_dir  paper_figures/outputs \\
    --device      cuda \\
    --sample_idx  0 \\
    --num_samples 20
"""
import argparse
import os
import subprocess
import sys
import traceback

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

import torch


def _run(label, func, *args, **kwargs):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    try:
        func(*args, **kwargs)
        print(f"  ✓ {label} done")
    except FileNotFoundError as e:
        print(f"  ✗ {label} SKIPPED — {e}")
    except Exception:
        print(f"  ✗ {label} FAILED:")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--checkpoint",       default="",
                        help="Trained model weights (.pth). Required for most figures.")
    parser.add_argument("--output_dir",       default="paper_figures/outputs")
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    # Sentence indices (into built-in pool — no dataset needed)
    parser.add_argument("--sentence_idx",     type=int, default=0,
                        help="Primary sentence index (0-9)")
    parser.add_argument("--sentence_idx_b",   type=int, default=1,
                        help="Second sentence for image-only alignment (fig10)")
    parser.add_argument("--sentence_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Sentence indices for fig11 success/failure grid")
    # Evaluation settings
    parser.add_argument("--num_samples",      type=int, default=10,
                        help="Sentences for cost distribution (fig05) and embedding space (fig09)")
    parser.add_argument("--num_negatives",    type=int, default=5)
    # Optional extras
    parser.add_argument("--ablation_csv",     default="",
                        help="Path to ablation CSV/JSON for fig06")
    parser.add_argument("--ablation_metric",  default="Top-1 Retrieval")
    parser.add_argument("--checkpoint_cnn_only", default="",
                        help="CNN-only checkpoint for fig07 (optional)")
    parser.add_argument("--embedding_method", default="tsne",
                        choices=["tsne", "umap", "pca"])
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Import figure modules ──────────────────────────────────────────────────
    from fig01_architecture        import draw_architecture
    from fig02_sliding_windows     import draw_sliding_windows
    if args.checkpoint:
        from fig03_similarity_before_after import draw_before_after
        from fig04_d3tw_path_overlay       import draw_path_overlay
        from fig05_pos_neg_costs           import draw_cost_distribution
        from fig08_hard_negative_case      import draw_hard_negative_case
        from fig09_embedding_space         import draw_embedding_space
        from fig10_image_only_alignment    import draw_image_only_alignment
        from fig11_success_failure_cases   import draw_success_failure

    # ── fig01: architecture (no checkpoint needed) ───────────────────────────
    _run("fig01 — Architecture diagram",
         draw_architecture, args.output_dir)

    # ── fig02: sliding windows ────────────────────────────────────────────────
    _run("fig02 — Sliding-window decomposition",
         draw_sliding_windows, args.sentence_idx, args.output_dir)

    if not args.checkpoint:
        print("\n  No --checkpoint provided — skipping figures 03–11.")
        print(f"\n{'='*60}")
        print(f"  Outputs in: {args.output_dir}")
        print(f"{'='*60}")
        return

    # ── fig03: before vs after ─────────────────────────────────────────────────
    _run("fig03 — Similarity before/after training",
         draw_before_after,
         args.checkpoint, args.sentence_idx,
         args.output_dir, args.device)

    # ── fig04: D3TW path overlay ───────────────────────────────────────────────
    _run("fig04 — D3TW alignment path",
         draw_path_overlay,
         args.checkpoint, args.sentence_idx,
         args.output_dir, args.device)

    # ── fig05: pos vs neg costs ────────────────────────────────────────────────
    _run("fig05 — Positive vs negative cost distribution",
         draw_cost_distribution,
         args.checkpoint,
         args.num_samples, args.num_negatives,
         args.output_dir, args.device)

    # ── fig06: ablation (optional) ─────────────────────────────────────────────
    if args.ablation_csv:
        from fig06_ablation_plot import draw_ablation
        _run("fig06 — Ablation study",
             draw_ablation,
             args.ablation_csv, args.ablation_metric, args.output_dir)
    else:
        print("\n  fig06 SKIPPED — provide --ablation_csv to generate.")

    # ── fig07: CNN vs BiLSTM (optional) ───────────────────────────────────────
    from fig07_cnn_vs_bilstm import draw_cnn_vs_bilstm
    _run("fig07 — CNN-only vs CNN+BiLSTM",
         draw_cnn_vs_bilstm,
         args.checkpoint_cnn_only or None,
         args.checkpoint,
         args.sentence_idx,
         args.output_dir, args.device)

    # ── fig08: hard negative case study ───────────────────────────────────────
    _run("fig08 — Hard negative case study",
         draw_hard_negative_case,
         args.checkpoint, args.sentence_idx,
         args.num_negatives, args.output_dir, args.device)

    # ── fig09: embedding space ─────────────────────────────────────────────────
    _run("fig09 — Embedding space (t-SNE)",
         draw_embedding_space,
         args.checkpoint,
         args.num_samples, args.output_dir,
         args.device, args.embedding_method)

    # ── fig10: image-only alignment ────────────────────────────────────────────
    _run("fig10 — Image-only alignment",
         draw_image_only_alignment,
         args.checkpoint,
         0,           # pair_idx into FIG10_SENTENCE_PAIRS
         args.output_dir, args.device)

    # ── fig11: success / failure cases ────────────────────────────────────────
    n = len(args.sentence_indices)
    labels = (["success", "partial", "failure"] + ["unknown"] * n)[:n]
    notes  = (["Good monotonic alignment",
                "Minor ambiguity near dots",
                "Failure due to degradation"] + [""] * n)[:n]
    _run("fig11 — Success/partial/failure grid",
         draw_success_failure,
         args.checkpoint,
         args.sentence_indices, labels, notes,
         args.output_dir, args.device)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
fig11_success_failure_cases.py
================================
Grid of qualitative alignment examples:
  Each row = one case (success / partial / failure)
  Columns: line image | similarity heatmap | D3TW path overlay | note

Output:
  fig11_success_failure_cases.pdf
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from utils.model_loading import load_image_model, load_text_embedder, load_char_bank_if_available
from utils.sample_data import make_sample, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_windows, pad_text
from utils.similarity import (compute_image_embeddings, compute_text_embeddings,
                               compute_text_image_similarity)
from utils.alignment import compute_path_quality
from utils.char_pooling import (
    compute_d3tw_char_pool_for_sample, char_pool_predictions, group_records,
)
from alignment_pooling import hard_d3tw_path_from_similarity
from utils.plotting import setup_paper_style, save_figure, plot_similarity_with_path, attach_window_strip, arabic_label


# Row-label colours
_CASE_COLORS = {
    "success": "#1a9641",
    "partial": "#fdae61",
    "failure": "#d7191c",
}


def _get_sim(model, text_embedder, img_tensor, text, device):
    img_emb = compute_image_embeddings(model, img_tensor, device)
    txt_emb = compute_text_embeddings(text_embedder, text)
    sim     = compute_text_image_similarity(txt_emb, img_emb)
    S = sim.shape[1]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
    return sim.detach().cpu().numpy()


def draw_success_failure(checkpoint, sentence_indices, case_labels,
                          notes, output_dir, device):
    setup_paper_style()

    model         = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)
    text_embedder = load_text_embedder(device)

    n_cases = len(sentence_indices)
    n_cols  = 4   # image | heatmap | heatmap+path | grouping/top-k

    col_w   = [3.0, 5.5, 5.5, 4.5]
    fig_w   = sum(col_w) + 1.5
    row_h   = 4.8
    fig_h   = n_cases * row_h + 2.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_cases, n_cols, figure=fig,
        width_ratios=col_w,
        hspace=0.45, wspace=0.30,
    )

    _heatmap_axes = []   # [(ax_sim, ax_path, wins), ...] – populated in loop
    case_json = []
    bank_emb, char_to_idx, idx_to_char = load_char_bank_if_available(checkpoint, device)

    for row, (si, label, note) in enumerate(
            zip(sentence_indices, case_labels, notes)):
        pil_img, text  = make_sample(si, transform=False)
        img_tensor, _  = make_sample(si, transform=True)

        text_padded = pad_text(text)
        sim_np = _get_sim(model, text_embedder, img_tensor, text_padded, device)
        path   = hard_d3tw_path_from_similarity(torch.tensor(sim_np))
        pq     = compute_path_quality(sim_np, path)
        chars  = list(text_padded)
        color  = _CASE_COLORS.get(label.lower(), "#555555")
        img_emb_full = compute_image_embeddings(model, img_tensor, device)
        txt_emb_full = compute_text_embeddings(text_embedder, text_padded)
        pool_result = compute_d3tw_char_pool_for_sample(
            visual_emb=img_emb_full,
            text_emb=txt_emb_full,
            transcript_chars=chars,
            detach_assignment=True,
        )
        predictions, pred_stats = char_pool_predictions(
            pool_result["pooled_visual"], chars, bank_emb, char_to_idx, idx_to_char,
            valid_mask=pool_result["valid_mask"], topk=5,
        )
        records = group_records(chars, pool_result["groups"], predictions)

        # Col 0: line image ────────────────────────────────────────────────────
        ax_img = fig.add_subplot(gs[row, 0])
        from PIL import Image as _PIL
        ax_img.imshow(np.array(pil_img.resize((708, 128))), aspect="auto")
        ax_img.axis("off")
        ax_img.set_title(
            f"[{label.upper()}]  Sentence {si}\n"
            f'"{arabic_label(text)}"',
            fontsize=7.5, color=color, pad=3,
        )
        # Coloured left border
        ax_img.axvline(x=0, color=color, linewidth=5, solid_capstyle="butt")

        # Col 1: similarity heatmap ────────────────────────────────────────────
        ax_sim = fig.add_subplot(gs[row, 1])
        T_s, S_s = sim_np.shape
        v_lo, v_hi = sim_np.min(), sim_np.max()
        span = max(v_hi - v_lo, 1e-8)
        im = ax_sim.imshow(sim_np, aspect="auto", cmap="viridis",
                           interpolation="nearest")
        im.set_rasterized(True)
        y_labels = ["_" if ch == " " else arabic_label(ch) for ch in chars]
        show_vals = max(T_s, S_s) <= 64
        fsize = max(6, min(11, int(320 / max(T_s, S_s))))

        ax_sim.set_title("Similarity matrix", fontsize=9, pad=3)
        ax_sim.set_ylabel("Chars", fontsize=8)
        ax_sim.set_yticks(range(T_s))
        ax_sim.set_yticklabels(y_labels, fontsize=7)
        ax_sim.tick_params(labelsize=6)
        # Omit per-panel colorbars to keep multi-case PDFs compact.
        # Dense per-cell vector gridlines make multi-sample PDFs very large.
        ax_sim.set_ylim(T_s - 0.5 + 0.2, -0.5 - 0.2)
        if show_vals:
            for r in range(T_s):
                for c in range(S_s):
                    norm = (sim_np[r, c] - v_lo) / span
                    clr  = "white" if norm < 0.6 else "#111111"
                    ax_sim.text(c, r, f"{sim_np[r,c]:.2f}", ha="center", va="center",
                                fontsize=fsize, color=clr, fontweight="bold")

        # Col 2: heatmap + path ────────────────────────────────────────────────
        ax_path = fig.add_subplot(gs[row, 2])
        im2 = ax_path.imshow(sim_np, aspect="auto", cmap="viridis",
                              interpolation="nearest")
        im2.set_rasterized(True)
        if path:
            rows_p = [p[0] for p in path]
            cols_p = [p[1] for p in path]
            ax_path.plot(cols_p, rows_p, color="red", linewidth=1.5, alpha=0.85)
        ax_path.set_title(
            f"Path overlay  |  gap={pq['path_gap']:+.3f}\n"
            f"Note: {note[:35]}",
            fontsize=9, pad=3,
        )
        ax_path.set_yticks(range(T_s))
        ax_path.set_yticklabels(y_labels, fontsize=7)
        ax_path.tick_params(labelsize=6)
        # Omit per-panel colorbars to keep multi-case PDFs compact.
        # Dense per-cell vector gridlines make multi-sample PDFs very large.
        ax_path.set_ylim(T_s - 0.5 + 0.2, -0.5 - 0.2)
        if show_vals:
            for r in range(T_s):
                for c in range(S_s):
                    norm = (sim_np[r, c] - v_lo) / span
                    clr  = "white" if norm < 0.6 else "#111111"
                    ax_path.text(c, r, f"{sim_np[r,c]:.2f}", ha="center", va="center",
                                 fontsize=fsize, color=clr, fontweight="bold")

        # Col 3: character-window grouping and char-pool predictions ─────────
        ax_group = fig.add_subplot(gs[row, 3])
        ax_group.axis("off")
        summary_lines = []
        for rec in records[:18]:
            ch = "sp" if rec["char"] == " " else rec["char"]
            pred = rec["top1_pred"]
            pred = "n/a" if pred is None else ("sp" if pred == " " else pred)
            ok = "" if rec["correct"] is None else ("✓" if rec["correct"] else "✗")
            summary_lines.append(
                f"{rec['char_index']:02d} {ch}: {rec['assigned_windows']} | {pred} {ok}"
            )
        ax_group.text(
            0.0, 1.0,
            "\n".join(summary_lines),
            transform=ax_group.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontfamily="monospace",
        )
        ax_group.set_title(
            f"Groups + top-k\nlabel={label}, top1={pred_stats['char_pool_top1']}",
            fontsize=8,
        )
        case_json.append({
            "sample_idx": int(si),
            "label": label,
            "note": note,
            "text": text,
            "path_quality": pq,
            "char_pool_stats": pred_stats,
            "groups": records,
        })

        # Store axes + windows for strip attachment after layout
        _heatmap_axes.append((ax_sim, ax_path, get_fig_windows(si)))

    fig.suptitle("Qualitative Success and Failure Cases",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.05)
    for ax_s, ax_p, wins in _heatmap_axes:
        attach_window_strip(ax_s, wins)
        attach_window_strip(ax_p, wins)
    save_figure(fig, output_dir, "fig11_success_failure_cases")
    plt.close(fig)
    print(f"  [fig11] computed diagnostics for {len(case_json)} cases (not saved; PDF-only output).")


def main():
    parser = argparse.ArgumentParser(
        description="Success/partial/failure case grid (fig11)."
    )
    parser.add_argument("--checkpoint",         required=True)
    parser.add_argument("--sentence_indices",   type=int, nargs="+", default=[0, 1, 2],
                        help="Indices into the built-in Arabic sentence pool")
    parser.add_argument("--case_labels",        nargs="+",
                        default=["success", "partial", "failure"])
    parser.add_argument("--notes",              nargs="+",
                        default=["Good monotonic alignment",
                                 "Minor ambiguity near dots",
                                 "Failure due to degradation"])
    parser.add_argument("--output_dir",         default="paper_figures/outputs")
    parser.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    n = len(args.sentence_indices)
    labels = (args.case_labels + ["unknown"] * n)[:n]
    notes  = (args.notes       + [""]        * n)[:n]

    os.chdir(_PROJ_ROOT)
    draw_success_failure(args.checkpoint,
                         args.sentence_indices, labels, notes,
                         args.output_dir, args.device)


if __name__ == "__main__":
    main()

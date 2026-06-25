"""
fig08_hard_negative_case.py
============================
Shows one image line alongside a ranked table of:
  - positive transcript  (✓ mark, lowest cost expected)
  - hard negative transcripts (cropped / dropped / shuffled / random)

Each row shows: rank | transcript | type | DTW cost | ✓ marker

Output:
  fig08_hard_negative_case_sample_{idx}.pdf
  fig08_hard_negative_case_sample_{idx}.json
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

from utils.model_loading import (
    load_image_model, load_text_embedder, load_ctc_components, load_char_bank_if_available,
)
from utils.sample_data import make_sample, ARABIC_SENTENCES, FIG_STRIDE, FIG_NUM_PATCHES, pad_text
from utils.similarity import (compute_image_embeddings, compute_text_embeddings,
                               compute_text_image_similarity, compute_dtw_cost)
from utils.char_pooling import compute_d3tw_char_pool_for_sample, char_pool_predictions
from ctc_utils import compute_ctc_cost
from utils.negatives import generate_hard_negatives
from utils.plotting import setup_paper_style, save_figure, arabic_label


def _get_cost(text_embedder, img_emb, text, device,
              cost_type="d3tw", ctc_head=None, ctc_vocab=None):
    try:
        if cost_type == "ctc":
            return compute_ctc_cost(img_emb, text, ctc_head, ctc_vocab, device)
        txt_emb = compute_text_embeddings(text_embedder, text)
        sim     = compute_text_image_similarity(txt_emb, img_emb)
        S = sim.shape[1]
        if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
            subfeat = S // FIG_NUM_PATCHES
            sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
        return compute_dtw_cost(sim, device="cpu")
    except Exception as e:
        print(f"    [fig08] cost failed: {e}")
        return float("nan")


def draw_hard_negative_case(checkpoint, sentence_idx, num_negatives,
                             output_dir, device, negative_mode="length_controlled",
                             cost_type="d3tw", include_char_pool_stats=False):
    setup_paper_style()

    pil_img, text  = make_sample(sentence_idx, transform=False)
    img_tensor, _  = make_sample(sentence_idx, transform=True)

    cost_type = cost_type.lower()
    if cost_type == "ctc":
        model, ctc_head, ctc_vocab = load_ctc_components(checkpoint, device)
        text_embedder = None
    else:
        model = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)
        text_embedder = load_text_embedder(device)
        ctc_head = None
        ctc_vocab = None
    img_emb       = compute_image_embeddings(model, img_tensor, device)

    all_texts = list(ARABIC_SENTENCES)
    negs = generate_hard_negatives(text, all_texts=all_texts,
                                   k=num_negatives, negative_mode=negative_mode)

    rows = []
    pos_cost = _get_cost(text_embedder, img_emb, pad_text(text), device,
                         cost_type=cost_type, ctc_head=ctc_head, ctc_vocab=ctc_vocab)
    char_pool_summary = {}
    if include_char_pool_stats and cost_type == "d3tw":
        try:
            chars = list(pad_text(text))
            txt_emb = compute_text_embeddings(text_embedder, pad_text(text))
            pool_result = compute_d3tw_char_pool_for_sample(
                visual_emb=img_emb,
                text_emb=txt_emb,
                transcript_chars=chars,
                detach_assignment=True,
            )
            bank_emb, char_to_idx, idx_to_char = load_char_bank_if_available(checkpoint, device)
            _, pred_stats = char_pool_predictions(
                pool_result["pooled_visual"], chars, bank_emb, char_to_idx, idx_to_char,
                valid_mask=pool_result["valid_mask"], topk=5,
            )
            counts = pool_result["counts"].detach().cpu().float().numpy()
            char_pool_summary = {
                "char_pool_top1": pred_stats["char_pool_top1"],
                "char_pool_top5": pred_stats["char_pool_top5"],
                "mean_windows_per_char": float(np.mean(counts)),
                "min_windows_per_char": float(np.min(counts)),
                "max_windows_per_char": float(np.max(counts)),
            }
        except Exception as exc:
            char_pool_summary = {"char_pool_warning": str(exc)}
    rows.append({"text": text,       "type": "positive", "cost": pos_cost, "is_pos": True})
    for neg_text, neg_type in negs:
        nc = _get_cost(text_embedder, img_emb, pad_text(neg_text), device,
                       cost_type=cost_type, ctc_head=ctc_head, ctc_vocab=ctc_vocab)
        rows.append({"text": neg_text, "type": neg_type, "cost": nc, "is_pos": False})

    # Rank by cost (lower = better alignment)
    valid = [r for r in rows if np.isfinite(r["cost"])]
    valid.sort(key=lambda r: r["cost"])
    for rank, r in enumerate(valid, 1):
        r["rank"] = rank

    # ── Figure layout ──────────────────────────────────────────────────────────
    n_rows = len(valid)
    fig_h  = 2.5 + n_rows * 0.40
    fig    = plt.figure(figsize=(13, fig_h))
    gs     = GridSpec(2, 1, figure=fig,
                      height_ratios=[2.0, n_rows * 0.40],
                      hspace=0.05)

    # Top panel: image
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(np.array(pil_img.resize((708, 128))), aspect="auto")
    pos_is_top = (next((r["rank"] for r in valid if r["is_pos"]), None) == 1)
    case_label = "Hard Negative Case Study" if pos_is_top else "Hard Negative Case Study (Failure)"
    ax_img.set_title(
        f"Sentence {sentence_idx} — {case_label}\n"
        f"\"{arabic_label(text)}\"  |  neg_mode={negative_mode}  |  cost={cost_type}",
        fontsize=11, pad=6,
    )
    ax_img.axis("off")

    # Bottom panel: table
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    col_labels = ["Rank", "Transcript", "Type", f"{cost_type.upper()} Cost", "Match"]
    table_data = []
    row_colors = []
    GREEN = "#d4efdf"
    RED   = "#fadbd8"
    WHITE = "#ffffff"

    for r in valid:
        short = arabic_label(r["text"])
        check = "✓ positive" if r["is_pos"] else ""
        table_data.append([
            str(r["rank"]),
            short,
            r["type"],
            f"{r['cost']:.4f}",
            check,
        ])
        row_colors.append([GREEN if r["is_pos"] else (RED if r["rank"] == 1 and not r["is_pos"] else WHITE)] * 5)

    # Fixed column widths: [rank, transcript, type, cost, check]
    col_widths = [0.06, 0.46, 0.16, 0.12, 0.14]

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c7bb6")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        # Right-align the Arabic transcript column
        if col == 1 and row > 0:
            cell._text.set_ha("right")

    pos_rank = next((r["rank"] for r in valid if r["is_pos"]), None)
    ax_tbl.set_title(
        f"Positive rank: {pos_rank}/{len(valid)}  |  "
        f"Positive cost: {pos_cost:.4f}",
        fontsize=9, pad=4,
    )

    plt.tight_layout()
    save_figure(fig, output_dir, f"fig08_hard_negative_case_s{sentence_idx}")
    plt.close(fig)

    if char_pool_summary:
        print(f"  [fig08] char-pool stats: {char_pool_summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Hard negative case study figure (fig08)."
    )
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--sentence_idx",  type=int, default=None,
                        help="Index into the built-in Arabic sentence pool")
    parser.add_argument("--sentence_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Multiple built-in sentence indices to render.")
    parser.add_argument("--num_negatives",  type=int, default=5)
    parser.add_argument("--negative_mode",  default="length_controlled",
                        choices=["mixed", "length_controlled", "dot_confusion",
                                 "same_length_random", "shuffle_only"])
    parser.add_argument("--cost_type", default="d3tw", choices=["d3tw", "ctc"])
    parser.add_argument("--include_char_pool_stats", action="store_true")
    parser.add_argument("--output_dir",    default="paper_figures/outputs")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    indices = [args.sentence_idx] if args.sentence_idx is not None else args.sentence_indices
    for idx in indices:
        draw_hard_negative_case(
            args.checkpoint,
            idx,
            args.num_negatives,
            args.output_dir,
            args.device,
            negative_mode=args.negative_mode,
            cost_type=args.cost_type,
            include_char_pool_stats=args.include_char_pool_stats,
        )


if __name__ == "__main__":
    main()

"""Main D3TW-guided character-pooling architecture figure."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from NormalizeFuncs import normalize_func
from utils.char_pooling import (
    compute_d3tw_char_pool_for_sample,
    char_pool_predictions,
    group_records,
)
from utils.model_loading import (
    load_char_bank_if_available,
    load_image_model,
    load_sample,
    load_text_embedder,
)


def _display_char(char):
    return "sp" if char == " " else char


def _load_text_embeddings(text, chars, device, checkpoint):
    bank_emb, char_to_idx, idx_to_char = load_char_bank_if_available(checkpoint, device)
    if bank_emb is not None and char_to_idx is not None and all(ch in char_to_idx for ch in chars):
        ids = torch.tensor([char_to_idx[ch] for ch in chars], device=device, dtype=torch.long)
        return bank_emb[ids], bank_emb, char_to_idx, idx_to_char

    print("  [fig13] Warning: char bank unavailable/incomplete; falling back to text embedder.")
    text_embedder = load_text_embedder(device)
    with torch.no_grad():
        text_emb = normalize_func(text_embedder(text).to(device))
    return text_emb, bank_emb, char_to_idx, idx_to_char


def _window_patches(image_np, ws, stride, num_visual):
    raw_windows = max(1, (image_np.shape[1] - ws) // stride + 1)
    subfeatures = max(1, num_visual // raw_windows)
    patches = []
    for raw_idx in range(raw_windows):
        x = image_np.shape[1] - ws - raw_idx * stride
        patches.append((raw_idx, max(0, x), min(image_np.shape[1], x + ws)))
    return patches, raw_windows, subfeatures


def generate_figure(
    checkpoint,
    data_dir,
    sample_idx,
    output_dir,
    device,
    window_overlap_mode="custom",
    show_topk=5,
):
    del window_overlap_mode  # Model stride comes from the checkpoint config.
    device = torch.device(device)
    model = load_image_model(checkpoint, device)
    image, transcript = load_sample(data_dir, sample_idx, transform=True)

    chars = list(" " + transcript + " ")
    text_for_embedding = "".join(chars)

    with torch.no_grad():
        visual_emb = normalize_func(model(image.unsqueeze(0).to(device)).squeeze(0).float())
        text_emb, bank_emb, char_to_idx, idx_to_char = _load_text_embeddings(
            text_for_embedding, chars, device, checkpoint
        )
        result = compute_d3tw_char_pool_for_sample(
            visual_emb=visual_emb,
            text_emb=text_emb,
            transcript_chars=chars,
            detach_assignment=True,
        )
        predictions, pred_stats = char_pool_predictions(
            pooled_visual=result["pooled_visual"],
            transcript_chars=chars,
            char_bank_embeddings=bank_emb,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            valid_mask=result["valid_mask"],
            topk=show_topk,
        )

    sim = result["sim"].detach().cpu().numpy()
    assignment = result["assignment"].detach().cpu().numpy()
    pooled = result["pooled_visual"].detach().cpu().numpy()
    path = result["path"]
    groups = result["groups"]
    records = group_records(chars, groups, predictions)

    image_np = image.permute(1, 2, 0).cpu().numpy()
    ws = int(getattr(model, "window_size", 16))
    stride = int(getattr(model, "stride", ws))
    patches, raw_windows, subfeatures = _window_patches(image_np, ws, stride, visual_emb.shape[0])

    fig = plt.figure(figsize=(34, 24), constrained_layout=False)
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=[2.2, 1.2, 4.0, 3.0, 4.0],
        width_ratios=[1.25, 1.0],
        hspace=0.42,
        wspace=0.18,
    )

    ax_img = fig.add_subplot(grid[0, :])
    ax_img.imshow(image_np, aspect="auto")
    ax_img.set_title(
        f"Arabic line image with selected visual windows — sample {sample_idx} "
        f"(window={ws}, stride={stride})",
        fontsize=16,
    )
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(chars), 1)))
    shown_raw = sorted(set(np.linspace(0, raw_windows - 1, min(24, raw_windows), dtype=int)))
    for raw_idx, left, right in patches:
        if raw_idx not in shown_raw:
            continue
        assigned_chars = sorted({
            j for j, group in enumerate(groups)
            if any((visual_idx // subfeatures) == raw_idx for visual_idx in group)
        })
        color = colors[assigned_chars[0] % len(colors)] if assigned_chars else (0, 0, 0, 1)
        ax_img.axvspan(left, right, color=color, alpha=0.22)
        label = "\n".join(_display_char(chars[j]) for j in assigned_chars) or "-"
        ax_img.text(
            (left + right) / 2,
            6,
            label,
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            linespacing=0.82,
            bbox=dict(facecolor="black", alpha=0.68, pad=1.5),
        )
    ax_img.axis("off")

    ax_pipe = fig.add_subplot(grid[1, :])
    ax_pipe.axis("off")
    ax_pipe.text(
        0.5,
        0.5,
        "V[S,D] + E[T,D]  →  S = E @ Vᵀ  →  D3TW path  →  assignment A[T,S]  "
        "→  pooled M = normalized(A) @ V  →  character InfoNCE",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#eef3fb", ec="#4c72b0"),
    )

    ax_sim = fig.add_subplot(grid[2, 0])
    im = ax_sim.imshow(sim, aspect="auto", origin="upper", cmap="viridis")
    if path:
        rows = [j for j, _ in path]
        cols = [i for _, i in path]
        ax_sim.plot(cols, rows, color="red", linewidth=1.6, label="hard D3TW path")
        ax_sim.legend(loc="upper right", fontsize=9)
    ax_sim.set_title("Pre-pooling similarity matrix S[j,i] = E[j] · V[i]", fontsize=13)
    ax_sim.set_xlabel("visual window/subfeature index i")
    ax_sim.set_ylabel("transcript character index j")
    ax_sim.set_yticks(range(len(chars)))
    ax_sim.set_yticklabels([_display_char(ch) for ch in chars], fontsize=8)
    fig.colorbar(im, ax=ax_sim, fraction=0.025, pad=0.01)

    ax_assign = fig.add_subplot(grid[2, 1])
    im_a = ax_assign.imshow(assignment, aspect="auto", origin="upper", cmap="Greys")
    ax_assign.set_title("D3TW assignment matrix A [T,S]", fontsize=13)
    ax_assign.set_xlabel("visual index i")
    ax_assign.set_ylabel("character index j")
    ax_assign.set_yticks(range(len(chars)))
    ax_assign.set_yticklabels([_display_char(ch) for ch in chars], fontsize=8)
    fig.colorbar(im_a, ax=ax_assign, fraction=0.035, pad=0.01)

    ax_pool = fig.add_subplot(grid[3, :])
    im_p = ax_pool.imshow(pooled, aspect="auto", origin="upper", cmap="coolwarm")
    ax_pool.set_title(
        "Pooled character vectors M = normalized(A) @ V "
        f"(shape={pooled.shape[0]}×{pooled.shape[1]})",
        fontsize=13,
    )
    ax_pool.set_xlabel("embedding dimension")
    ax_pool.set_ylabel("character index")
    ax_pool.set_yticks(range(len(chars)))
    ax_pool.set_yticklabels([_display_char(ch) for ch in chars], fontsize=8)
    fig.colorbar(im_p, ax=ax_pool, fraction=0.015, pad=0.01)

    ax_table = fig.add_subplot(grid[4, :])
    ax_table.axis("off")
    rows = []
    for record in records[: min(42, len(records))]:
        rows.append([
            record["char_index"],
            _display_char(record["char"]),
            ",".join(map(str, record["assigned_windows"])),
            record["num_windows"],
            _display_char(record["top1_pred"]) if record["top1_pred"] is not None else "n/a",
            " ".join(_display_char(ch) for ch in record["top5_pred"])
            if record["top5_pred"] else "n/a",
            "" if record["correct"] is None else ("✓" if record["correct"] else "✗"),
        ])
    table = ax_table.table(
        cellText=rows,
        colLabels=["char index", "char", "assigned windows", "#win", "top1", "top5", "ok"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax_table.set_title(
        "Character-window groups from D3TW assignment"
        + (f" | top1={pred_stats['char_pool_top1']:.3f}, top5={pred_stats['char_pool_top5']:.3f}"
           if pred_stats["char_pool_top1"] is not None else " | char bank unavailable"),
        fontsize=13,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig13_d3tw_character_pooling_sample_{sample_idx}"
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_overlap_mode", default="custom")
    parser.add_argument("--show_topk", type=int, default=5)
    args = parser.parse_args()
    indices = [args.sample_idx] if args.sample_idx is not None else args.sample_indices
    for idx in indices:
        generate_figure(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            sample_idx=idx,
            output_dir=args.output_dir,
            device=args.device,
            window_overlap_mode=args.window_overlap_mode,
            show_topk=args.show_topk,
        )


if __name__ == "__main__":
    main()

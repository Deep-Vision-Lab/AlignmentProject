"""Main D3TW-guided character-pooling architecture figure."""

import argparse
import os
import sys
from textwrap import fill

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
    token_pool_predictions,
    unit_group_records,
)
from utils.model_loading import (
    load_char_bank_if_available,
    load_image_model,
    load_ngram_tokenizer_if_available,
    load_sample,
    load_text_embedder,
    load_token_bank_if_available,
)
from token_bank import build_adjacent_pair_visuals
from token_embedding_bank import encode_text_units


def _display_char(char):
    return "sp" if char == " " else char


def _display_token(token, max_len=18):
    text = str(token)
    if text == " ":
        return "sp"
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _join_predictions(values, max_items=4, max_len=18):
    shown = [_display_token(value, max_len=max_len) for value in list(values)[:max_items]]
    if len(values) > max_items:
        shown.append("...")
    return "\n".join(shown) if shown else "n/a"


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


def _window_unit_assignments(assignment, raw_windows, subfeatures):
    """Map each image window to the transcript unit with most assigned features."""
    owners = []
    if assignment.size == 0:
        return [None] * raw_windows

    num_visual = assignment.shape[1]
    for raw_idx in range(raw_windows):
        start = raw_idx * subfeatures
        end = min(num_visual, start + subfeatures)
        if start >= num_visual:
            owners.append(None)
            continue
        scores = assignment[:, start:end].sum(axis=1)
        owner = int(np.argmax(scores))
        owners.append(owner if float(scores[owner]) > 0 else None)
    return owners


def _stable_unit_colors(units):
    """Assign one stable color per distinct letter/token, not per position."""
    palette = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    palette += list(plt.cm.Set3(np.linspace(0, 1, 12)))
    color_by_unit = {}
    next_color = 0
    for unit in units:
        if unit == " ":
            continue
        if unit not in color_by_unit:
            color_by_unit[unit] = palette[next_color % len(palette)]
            next_color += 1
    return color_by_unit


def _contiguous_runs(values):
    runs = []
    start = 0
    prev = values[0] if values else None
    for idx, value in enumerate(values[1:], start=1):
        if value != prev:
            runs.append((start, idx - 1, prev))
            start = idx
            prev = value
    if values:
        runs.append((start, len(values) - 1, prev))
    return runs


def generate_figure(
    checkpoint,
    data_dir,
    sample_idx,
    output_dir,
    device,
    window_overlap_mode="custom",
    show_topk=5,
    text_unit_type="char",
):
    del window_overlap_mode  # Model stride comes from the checkpoint config.
    device = torch.device(device)
    model = load_image_model(checkpoint, device)
    image, transcript = load_sample(data_dir, sample_idx, transform=True)

    text_padded = " " + transcript + " "
    text_unit_type = str(text_unit_type).lower()

    with torch.no_grad():
        visual_emb = normalize_func(model(image.unsqueeze(0).to(device)).squeeze(0).float())
        if text_unit_type == "ngram":
            ngram_tokenizer = load_ngram_tokenizer_if_available(checkpoint)
            if ngram_tokenizer is None:
                raise RuntimeError("fig13 --text_unit_type ngram requires ngram_vocab.json next to the checkpoint.")
            text_embedder = load_text_embedder(device)
            units, spans, text_emb = encode_text_units(
                text_padded, "ngram", text_embedder, device, ngram_tokenizer=ngram_tokenizer
            )
            bank_emb, unit_to_idx, idx_to_unit = load_token_bank_if_available(checkpoint, device)
        else:
            units = list(text_padded)
            spans = [(idx, idx + 1) for idx in range(len(units))]
            text_emb, bank_emb, unit_to_idx, idx_to_unit = _load_text_embeddings(
                text_padded, units, device, checkpoint
            )
        result = compute_d3tw_char_pool_for_sample(
            visual_emb=visual_emb,
            text_emb=text_emb,
            transcript_chars=units,
            detach_assignment=True,
        )
        bigram_rows = []
        if text_unit_type == "ngram":
            predictions, pred_stats = token_pool_predictions(
                result["pooled_visual"],
                units,
                bank_emb,
                unit_to_idx,
                idx_to_unit,
                valid_mask=result["valid_mask"],
                topk=show_topk,
            )
        else:
            predictions, pred_stats = char_pool_predictions(
                pooled_visual=result["pooled_visual"],
                transcript_chars=units,
                char_bank_embeddings=bank_emb,
                char_to_idx=unit_to_idx,
                idx_to_char=idx_to_unit,
                valid_mask=result["valid_mask"],
                topk=show_topk,
            )
            token_emb, token_to_idx, idx_to_token = load_token_bank_if_available(checkpoint, device)
            if token_emb is not None and token_to_idx is not None and idx_to_token is not None:
                pair_visuals, target_ids, pair_meta = build_adjacent_pair_visuals(
                    pooled_visual=result["pooled_visual"],
                    transcript_chars=units,
                    token_to_idx=token_to_idx,
                    fusion="mean",
                    skip_spaces=True,
                )
                if pair_visuals.numel() > 0:
                    logits = normalize_func(pair_visuals) @ normalize_func(token_emb).T
                    top_idx = logits.topk(min(show_topk, logits.shape[-1]), dim=-1).indices.cpu().tolist()
                    targets = target_ids.cpu().tolist()
                    for meta, preds, target in zip(pair_meta, top_idx, targets):
                        pred_tokens = [idx_to_token[i] for i in preds]
                        bigram_rows.append({
                            **meta,
                            "top1_pred": pred_tokens[0] if pred_tokens else None,
                            "top5_pred": pred_tokens,
                            "correct": bool(preds and preds[0] == target),
                        })

    sim = result["sim"].detach().cpu().numpy()
    assignment = result["assignment"].detach().cpu().numpy()
    pooled = result["pooled_visual"].detach().cpu().numpy()
    path = result["path"]
    groups = result["groups"]
    if text_unit_type == "ngram":
        records = unit_group_records(units, spans, groups, predictions)
    else:
        records = group_records(units, groups, predictions)

    image_np = image.permute(1, 2, 0).cpu().numpy()
    ws = int(getattr(model, "window_size", 16))
    stride = int(getattr(model, "stride", ws))
    patches, raw_windows, subfeatures = _window_patches(image_np, ws, stride, visual_emb.shape[0])

    max_table_rows = 28 if text_unit_type == "char" else 22
    table_rows_est = min(max_table_rows, len(records)) + min(8, len(bigram_rows))
    table_ratio = max(4.8, min(7.2, 2.7 + 0.16 * table_rows_est))
    fig_h = max(26, 20 + 0.28 * table_rows_est)
    fig = plt.figure(figsize=(34, fig_h), constrained_layout=False)
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=[2.4, 1.45, 4.0, 3.2, table_ratio],
        width_ratios=[1.25, 1.0],
        hspace=0.55,
        wspace=0.18,
    )

    ax_img = fig.add_subplot(grid[0, :])
    ax_img.imshow(image_np, aspect="auto")
    ax_img.set_title(
        f"Arabic line image masked by D3TW-assigned letters — sample {sample_idx} "
        f"(all {raw_windows} windows, window={ws}, stride={stride})",
        fontsize=16,
    )
    window_owners = _window_unit_assignments(assignment, raw_windows, subfeatures)
    color_by_unit = _stable_unit_colors(units)
    window_letters = [
        units[owner] if owner is not None and 0 <= owner < len(units) else None
        for owner in window_owners
    ]
    for raw_idx, left, right in patches:
        letter = window_letters[raw_idx] if raw_idx < len(window_letters) else None
        if letter is None or letter == " ":
            continue
        color = color_by_unit[letter]
        ax_img.axvspan(left, right, color=color, alpha=0.22)
        ax_img.axvline(left, color=color, alpha=0.45, linewidth=0.35)
    ax_img.axvline(image_np.shape[1], color="#222222", alpha=0.25, linewidth=0.35)

    patch_by_raw = {raw_idx: (left, right) for raw_idx, left, right in patches}
    for start, end, letter in _contiguous_runs(window_letters):
        if letter is None or letter == " ":
            continue
        left = patch_by_raw[start][0]
        right = patch_by_raw[end][1]
        ax_img.text(
            (left + right) / 2,
            6,
            _display_token(letter, max_len=4),
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.58, pad=0.9),
        )

    legend_units = [unit for unit in dict.fromkeys(units) if unit != " "]
    max_legend = min(32, len(legend_units))
    for slot, unit in enumerate(legend_units[:max_legend]):
        x = 0.012 + slot * (0.976 / max(max_legend, 1))
        ax_img.text(
            x,
            -0.16,
            _display_token(unit, max_len=4),
            transform=ax_img.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#111111",
            bbox=dict(
                boxstyle="square,pad=0.12",
                fc=color_by_unit[unit],
                ec="none",
                alpha=0.65,
            ),
            clip_on=False,
        )
    ax_img.axis("off")

    ax_pipe = fig.add_subplot(grid[1, :])
    ax_pipe.axis("off")
    ax_pipe.text(
        0.5,
        0.5,
        "V[S,D] + E[K,D]\n"
        "S = E @ V^T  ->  D3TW path  ->  assignment A[K,S]\n"
        f"pooled M = normalized(A) @ V  ->  {text_unit_type} token InfoNCE",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", fc="#eef3fb", ec="#4c72b0"),
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
    ax_sim.set_ylabel(f"transcript {text_unit_type} unit index j")
    ax_sim.set_yticks(range(len(units)))
    ax_sim.set_yticklabels([_display_char(ch) for ch in units], fontsize=8)
    fig.colorbar(im, ax=ax_sim, fraction=0.025, pad=0.01)

    ax_assign = fig.add_subplot(grid[2, 1])
    im_a = ax_assign.imshow(assignment, aspect="auto", origin="upper", cmap="Greys")
    ax_assign.set_title("D3TW assignment matrix A [T,S]", fontsize=13)
    ax_assign.set_xlabel("visual index i")
    ax_assign.set_ylabel(f"{text_unit_type} unit index j")
    ax_assign.set_yticks(range(len(units)))
    ax_assign.set_yticklabels([_display_char(ch) for ch in units], fontsize=8)
    fig.colorbar(im_a, ax=ax_assign, fraction=0.035, pad=0.01)

    ax_pool = fig.add_subplot(grid[3, :])
    im_p = ax_pool.imshow(pooled, aspect="auto", origin="upper", cmap="coolwarm")
    ax_pool.set_title(
        f"Pooled {text_unit_type} unit vectors M = normalized(A) @ V "
        f"(shape={pooled.shape[0]}×{pooled.shape[1]})",
        fontsize=13,
    )
    ax_pool.set_xlabel("embedding dimension")
    ax_pool.set_ylabel(f"{text_unit_type} unit index")
    ax_pool.set_yticks(range(len(units)))
    ax_pool.set_yticklabels([_display_char(ch) for ch in units], fontsize=8)
    fig.colorbar(im_p, ax=ax_pool, fraction=0.015, pad=0.01)

    ax_table = fig.add_subplot(grid[4, :])
    ax_table.axis("off")
    rows = []
    if text_unit_type == "ngram":
        for record in records[: min(max_table_rows, len(records))]:
            rows.append([
                record["unit_index"],
                fill(_display_token(record["token"], max_len=20), width=10),
                f"{record['span'][0]}-{record['span'][1]}",
                ",".join(map(str, record["assigned_windows"])),
                record["num_windows"],
                _display_token(record["top1_pred"]) if record["top1_pred"] is not None else "n/a",
                _join_predictions(record["top5_pred"]) if record["top5_pred"] else "n/a",
                "" if record["correct"] is None else ("yes" if record["correct"] else "no"),
            ])
        col_labels = ["unit index", "token", "span", "assigned windows", "#win", "top1", "top5", "ok"]
    else:
        for record in records[: min(max_table_rows, len(records))]:
            rows.append([
                record["char_index"],
                _display_char(record["char"]),
                ",".join(map(str, record["assigned_windows"])),
                record["num_windows"],
                _display_char(record["top1_pred"]) if record["top1_pred"] is not None else "n/a",
                _join_predictions(record["top5_pred"], max_len=6) if record["top5_pred"] else "n/a",
                "" if record["correct"] is None else ("yes" if record["correct"] else "no"),
            ])
        col_labels = ["char index", "char", "assigned windows", "#win", "top1", "top5", "ok"]
    if bigram_rows:
        rows.append(["-", "BIGRAMS", "-", "-", "-", "-", "-"])
        for idx, record in enumerate(bigram_rows[:8]):
            rows.append([
                f"p{idx}",
                _display_token(record["token"]),
                f"{record['start_char_index']}-{record['end_char_index']}",
                "2 chars",
                _display_token(record["top1_pred"]) if record["top1_pred"] else "n/a",
                _join_predictions(record["top5_pred"]) if record["top5_pred"] else "n/a",
                "yes" if record["correct"] else "no",
            ])
    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7 if len(rows) > 26 else 8)
    table.scale(1.0, 1.08 if len(rows) > 26 else 1.18)
    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass
    ax_table.set_title(
        f"{text_unit_type} text-unit/window groups from D3TW assignment"
        + (
            f" | top1={pred_stats.get('char_pool_top1', pred_stats.get('token_pool_top1')):.3f}, "
            f"top5={pred_stats.get('char_pool_top5', pred_stats.get('token_pool_top5')):.3f}"
            if pred_stats.get("char_pool_top1", pred_stats.get("token_pool_top1")) is not None
            else " | bank unavailable"
        ),
        fontsize=13,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig13_d3tw_character_pooling_sample_{sample_idx}"
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {stem}.pdf/.png in {output_dir}")


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
    parser.add_argument("--text_unit_type", default="char", choices=["char", "ngram"])
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
            text_unit_type=args.text_unit_type,
        )


if __name__ == "__main__":
    main()

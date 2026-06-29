"""Main D3TW-guided character-pooling architecture figure."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
from token_embedding_bank import embed_token_with_fallback, encode_text_units


def _display_char(char):
    return "sp" if char == " " else char


def _display_unit(unit, max_len=9):
    text = str(unit)
    if text == " ":
        return "sp"
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _stable_unit_colors(units):
    palette = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    palette += list(plt.cm.Set3(np.linspace(0, 1, 12)))
    colors = {}
    for unit in units:
        if unit not in colors:
            colors[unit] = palette[len(colors) % len(palette)]
    return colors


def _word_spans(text):
    words = []
    start = None
    for idx, ch in enumerate(text):
        if ch == " ":
            if start is not None:
                words.append({"start": start, "end": idx, "text": text[start:idx]})
                start = None
        elif start is None:
            start = idx
    if start is not None:
        words.append({"start": start, "end": len(text), "text": text[start:]})
    return words


def _word_space_units(text):
    units = []
    spans = []
    idx = 0
    while idx < len(text):
        ch = text[idx]
        start = idx
        if ch == " ":
            while idx < len(text) and text[idx] == " ":
                idx += 1
            units.append(" ")
            spans.append((start, idx))
        else:
            while idx < len(text) and text[idx] != " ":
                idx += 1
            units.append(text[start:idx])
            spans.append((start, idx))
    return units, spans


def _encode_word_units(text, text_embedder, device):
    units, spans = _word_space_units(text)
    if not units:
        return units, spans, torch.empty((0, 0), device=device)

    text_embedder.eval()
    vectors = []
    with torch.no_grad():
        for unit in units:
            vec = embed_token_with_fallback(text_embedder, unit, device)
            vectors.append(vec.to(device=device, dtype=torch.float32))
        embeddings = normalize_func(torch.stack(vectors, dim=0))
    return units, spans, embeddings


def _encode_bigram_units(text, text_embedder, device, include_space_units=True):
    units = []
    spans = []
    idx = 0
    while idx < len(text):
        start = idx
        if text[idx] == " ":
            while idx < len(text) and text[idx] == " ":
                idx += 1
            if include_space_units:
                units.append(" ")
                spans.append((start, idx))
            continue

        while idx < len(text) and text[idx] != " ":
            idx += 1
        word_start, word_end = start, idx
        if word_end - word_start == 1:
            units.append(text[word_start:word_end])
            spans.append((word_start, word_end))
            continue
        for char_idx in range(word_start, word_end - 1):
            units.append(text[char_idx:char_idx + 2])
            spans.append((char_idx, char_idx + 2))
    if not units:
        return units, spans, torch.empty((0, 0), device=device)

    text_embedder.eval()
    vectors = []
    with torch.no_grad():
        for unit in units:
            vec = embed_token_with_fallback(text_embedder, unit, device)
            vectors.append(vec.to(device=device, dtype=torch.float32))
        embeddings = normalize_func(torch.stack(vectors, dim=0))
    return units, spans, embeddings


def _unit_word_map(spans, words):
    mapping = []
    for span in spans:
        unit_start, unit_end = int(span[0]), int(span[1])
        best_idx = None
        best_overlap = 0
        for word_idx, word in enumerate(words):
            overlap = max(0, min(unit_end, word["end"]) - max(unit_start, word["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = word_idx
        if best_idx is None:
            # Space units sit between word spans. Attach the space before a
            # word to that following word so the word mask starts from the
            # first space-owned window before its letters.
            for word_idx, word in enumerate(words):
                if word["start"] >= unit_end:
                    best_idx = word_idx
                    break
            if best_idx is None and words:
                best_idx = len(words) - 1
        mapping.append(best_idx)
    return mapping


def _window_unit_assignments(assignment, sim, raw_windows, subfeatures):
    """
    Assign every raw sliding window to one text unit.

    Hard D3TW can leave a raw window with no assigned subfeature.  For those
    cases, use the strongest mean similarity over the window's subfeatures so
    the top image has no unmasked gaps.
    """
    owners = []
    num_visual = assignment.shape[1] if assignment.size else sim.shape[1]
    for raw_idx in range(raw_windows):
        start = raw_idx * subfeatures
        end = min(num_visual, start + subfeatures)
        if start >= num_visual:
            owners.append(None)
            continue

        assign_scores = assignment[:, start:end].sum(axis=1) if assignment.size else np.array([])
        if assign_scores.size and float(assign_scores.max()) > 0:
            owners.append(int(np.argmax(assign_scores)))
            continue

        sim_scores = sim[:, start:end].mean(axis=1)
        owners.append(int(np.argmax(sim_scores)))
    return owners


def _window_word_assignments(unit_owners, unit_to_word, assignment, sim, raw_windows, subfeatures):
    word_owners = []
    num_visual = assignment.shape[1] if assignment.size else sim.shape[1]
    for raw_idx, unit_owner in enumerate(unit_owners):
        if unit_owner is not None and unit_owner < len(unit_to_word):
            word_owner = unit_to_word[unit_owner]
            word_owners.append(word_owner)
            continue

        start = raw_idx * subfeatures
        end = min(num_visual, start + subfeatures)
        if start >= num_visual:
            word_owners.append(None)
            continue

        sim_scores = sim[:, start:end].mean(axis=1)
        for unit_idx in np.argsort(sim_scores)[::-1]:
            unit_idx = int(unit_idx)
            if unit_idx < len(unit_to_word) and unit_to_word[unit_idx] is not None:
                word_owners.append(unit_to_word[unit_idx])
                break
        else:
            word_owners.append(None)
    return word_owners


def _word_runs(owners, words, patches):
    runs = []
    if not patches:
        return runs

    start_idx = 0
    prev_owner = owners[0] if owners else None
    for raw_idx in range(1, len(patches)):
        owner = owners[raw_idx] if raw_idx < len(owners) else None
        if owner != prev_owner:
            runs.append((start_idx, raw_idx - 1, prev_owner))
            start_idx = raw_idx
            prev_owner = owner
    runs.append((start_idx, len(patches) - 1, prev_owner))

    labeled_runs = []
    for start, end, owner in runs:
        word = words[owner]["text"] if owner is not None and owner < len(words) else None
        run_patches = patches[start:end + 1]
        left = min(patch[1] for patch in run_patches)
        right = max(patch[2] for patch in run_patches)
        labeled_runs.append((left, right, word))
    return labeled_runs


def _display_word_letters(word, max_len=18):
    text = " ".join(str(word))
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _draw_space_row_guides(ax, units, n_cols, color="#ff2f92"):
    """Draw full-width guide lines on DTW rows that represent spaces."""
    if not units or n_cols <= 0:
        return
    x0, x1 = -0.5, n_cols - 0.5
    for row_idx, unit in enumerate(units):
        if unit != " ":
            continue
        ax.axhspan(row_idx - 0.5, row_idx + 0.5, color=color, alpha=0.08, zorder=3)
        ax.hlines(
            row_idx,
            x0,
            x1,
            colors=color,
            linestyles="-",
            linewidth=1.4,
            alpha=0.95,
            zorder=4,
        )


def _draw_word_row_bands(ax, units, spans, words, word_colors, n_cols):
    """Tint each non-space text-unit row by the word it belongs to."""
    if not units or not words or n_cols <= 0:
        return
    unit_to_word = _unit_word_map(spans, words)
    x0, x1 = -0.5, n_cols - 0.5
    for row_idx, word_idx in enumerate(unit_to_word):
        if row_idx >= len(units) or units[row_idx] == " ":
            continue
        if word_idx is None or word_idx >= len(words):
            continue
        word = words[word_idx]["text"]
        color = word_colors.get(word, (0.15, 0.15, 0.15, 1.0))
        ax.axhspan(
            row_idx - 0.5,
            row_idx + 0.5,
            xmin=0,
            xmax=1,
            color=color,
            alpha=0.075,
            zorder=3,
        )
        ax.hlines(
            [row_idx - 0.5, row_idx + 0.5],
            x0,
            x1,
            colors=[color, color],
            linewidth=0.55,
            alpha=0.5,
            zorder=4,
        )


def _draw_window_word_rectangles(
    ax,
    window_word_owners,
    words,
    word_colors,
    n_rows,
    n_cols,
    subfeatures,
):
    """Color each visual-window block in the heatmap by its assigned word."""
    if not window_word_owners or not words or n_rows <= 0 or n_cols <= 0:
        return

    subfeatures = max(1, int(subfeatures))
    y0 = -0.5
    height = n_rows
    for raw_idx, word_idx in enumerate(window_word_owners):
        start_col = raw_idx * subfeatures
        if start_col >= n_cols:
            break
        end_col = min(n_cols, start_col + subfeatures)
        if word_idx is None or word_idx >= len(words):
            continue

        word = words[word_idx]["text"]
        color = word_colors.get(word, (0.15, 0.15, 0.15, 1.0))
        left = start_col - 0.5
        width = max(0.0, end_col - start_col)
        ax.add_patch(Rectangle(
            (left, y0),
            width,
            height,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.18,
            zorder=3,
        ))


def _draw_window_word_overlay(
    ax,
    image_np,
    patches,
    window_word_owners,
    words,
    word_colors,
    mask_gap_px=4,
    use_flip=False,
):
    display_patches = list(reversed(patches)) if use_flip else patches
    word_runs = _word_runs(window_word_owners, words, display_patches)

    gap = max(0.0, float(mask_gap_px))
    for left, right, word in word_runs:
        if word is None:
            continue
        draw_left = min(right, left + gap / 2.0)
        draw_right = max(draw_left, right - gap / 2.0)
        color = word_colors.get(word, (0.15, 0.15, 0.15, 1.0))
        ax.axvspan(draw_left, draw_right, color=color, alpha=0.26)

    for left, right, word in word_runs:
        if word is None:
            continue
        draw_left = min(right, left + gap / 2.0)
        draw_right = max(draw_left, right - gap / 2.0)
        ax.text(
            (draw_left + draw_right) / 2,
            5,
            _display_word_letters(word),
            ha="center",
            va="top",
            fontsize=6,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, pad=0.45),
        )


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
    text_unit_type="char",
    window_size=None,
    stride=None,
    word_mask_gap_px=4,
):
    del window_overlap_mode  # Kept for backward-compatible CLI calls.
    device = torch.device(device)
    if window_size is not None and int(window_size) <= 0:
        raise ValueError("--window_size must be a positive integer.")
    if stride is not None and int(stride) <= 0:
        raise ValueError("--stride must be a positive integer.")
    model = load_image_model(
        checkpoint,
        device,
        window_size_override=window_size,
        stride_override=stride,
    )
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
        elif text_unit_type == "word":
            text_embedder = load_text_embedder(device)
            units, spans, text_emb = _encode_word_units(text_padded, text_embedder, device)
            bank_emb, unit_to_idx, idx_to_unit = None, None, None
        elif text_unit_type == "bigram":
            text_embedder = load_text_embedder(device)
            units, spans, text_emb = _encode_bigram_units(text_padded, text_embedder, device)
            bank_emb, unit_to_idx, idx_to_unit = load_token_bank_if_available(checkpoint, device)
        elif text_unit_type == "char":
            units = list(text_padded)
            spans = [(idx, idx + 1) for idx in range(len(units))]
            text_emb, bank_emb, unit_to_idx, idx_to_unit = _load_text_embeddings(
                text_padded, units, device, checkpoint
            )
        else:
            raise ValueError(f"Unknown text_unit_type: {text_unit_type}")
        result = compute_d3tw_char_pool_for_sample(
            visual_emb=visual_emb,
            text_emb=text_emb,
            transcript_chars=units,
            detach_assignment=True,
        )
        bigram_rows = []
        if text_unit_type in {"ngram", "bigram"}:
            predictions, pred_stats = token_pool_predictions(
                result["pooled_visual"],
                units,
                bank_emb,
                unit_to_idx,
                idx_to_unit,
                valid_mask=result["valid_mask"],
                topk=show_topk,
            )
        elif text_unit_type == "word":
            predictions, pred_stats = None, {
                "token_pool_top1": None,
                "token_pool_top5": None,
                "token_pool_valid_tokens": 0,
            }
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
    if text_unit_type in {"ngram", "bigram", "word"}:
        records = unit_group_records(units, spans, groups, predictions)
    else:
        records = group_records(units, groups, predictions)

    image_np = image.permute(1, 2, 0).cpu().numpy()
    ws = int(getattr(model, "window_size", 16))
    stride = int(getattr(model, "stride", ws))
    patches, raw_windows, subfeatures = _window_patches(image_np, ws, stride, visual_emb.shape[0])
    words_for_guides = _word_spans(text_padded)
    unit_to_word_for_guides = _unit_word_map(spans, words_for_guides)
    window_unit_owners = _window_unit_assignments(assignment, sim, raw_windows, subfeatures)
    window_word_owners = _window_word_assignments(
        window_unit_owners,
        unit_to_word_for_guides,
        assignment,
        sim,
        raw_windows,
        subfeatures,
    )
    word_colors_for_guides = _stable_unit_colors([word["text"] for word in words_for_guides])

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
        f"Arabic line image with merged word masks from assigned {text_unit_type} windows "
        f"- sample {sample_idx} (windows={raw_windows}, window={ws}, stride={stride})",
        fontsize=16,
    )
    _draw_window_word_overlay(
        ax_img,
        image_np,
        patches,
        window_word_owners,
        words_for_guides,
        word_colors_for_guides,
        mask_gap_px=word_mask_gap_px,
        use_flip=bool(getattr(model, "use_flip", False)),
    )
    ax_img.axis("off")

    ax_pipe = fig.add_subplot(grid[1, :])
    ax_pipe.axis("off")
    ax_pipe.text(
        0.5,
        0.5,
        "V[S,D] + E[K,D]  →  S = E @ Vᵀ  →  D3TW path  →  assignment A[K,S]  "
        f"→  pooled M = normalized(A) @ V  →  {text_unit_type} token InfoNCE",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#eef3fb", ec="#4c72b0"),
    )

    ax_sim = fig.add_subplot(grid[2, 0])
    im = ax_sim.imshow(sim, aspect="auto", origin="upper", cmap="viridis")
    _draw_window_word_rectangles(
        ax_sim,
        window_word_owners,
        words_for_guides,
        word_colors_for_guides,
        sim.shape[0],
        sim.shape[1],
        subfeatures,
    )
    _draw_space_row_guides(ax_sim, units, sim.shape[1])
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
    _draw_window_word_rectangles(
        ax_assign,
        window_word_owners,
        words_for_guides,
        word_colors_for_guides,
        assignment.shape[0],
        assignment.shape[1],
        subfeatures,
    )
    _draw_space_row_guides(ax_assign, units, assignment.shape[1])
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
    if text_unit_type in {"ngram", "bigram", "word"}:
        for record in records[: min(42, len(records))]:
            rows.append([
                record["unit_index"],
                _display_char(record["token"]),
                f"{record['span'][0]}-{record['span'][1]}",
                ",".join(map(str, record["assigned_windows"])),
                record["num_windows"],
                _display_char(record["top1_pred"]) if record["top1_pred"] is not None else "n/a",
                " ".join(_display_char(ch) for ch in record["top5_pred"])
                if record["top5_pred"] else "n/a",
                "" if record["correct"] is None else ("✓" if record["correct"] else "✗"),
            ])
        col_labels = ["unit index", text_unit_type, "span", "assigned windows", "#win", "top1", "top5", "ok"]
    else:
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
        col_labels = ["char index", "char", "assigned windows", "#win", "top1", "top5", "ok"]
    if bigram_rows:
        rows.append(["—", "BIGRAMS", "—", "—", "—", "—", "—"])
        for idx, record in enumerate(bigram_rows[:12]):
            rows.append([
                f"p{idx}",
                record["token"],
                f"{record['start_char_index']}-{record['end_char_index']}",
                "2 chars",
                record["top1_pred"] or "n/a",
                " ".join(record["top5_pred"]) if record["top5_pred"] else "n/a",
                "✓" if record["correct"] else "✗",
            ])
    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
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
    parser.add_argument("--window_size", type=int, default=None,
                        help="Override checkpoint window size for figure generation.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Override checkpoint sliding-window stride for figure generation.")
    parser.add_argument("--word_mask_gap_px", type=float, default=4,
                        help="Horizontal pixel gap between adjacent merged word masks.")
    parser.add_argument("--show_topk", type=int, default=5)
    parser.add_argument("--text_unit_type", default="word", choices=["char", "ngram", "bigram", "word"])
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
            window_size=args.window_size,
            stride=args.stride,
            word_mask_gap_px=args.word_mask_gap_px,
        )


if __name__ == "__main__":
    main()

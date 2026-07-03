"""Figure 13: D3TW-guided text-unit/token pooling.

This visualizes the same operation used during training:
V[S,D] + E[K,D] -> sim=E@V.T [K,S] -> D3TW assignment A[K,S]
-> pooled text-unit vectors M = normalized(A) @ V [K,D].

For --text_unit_type ngram, the script loads the exact ngram_vocab.json and
token_bank.json saved next to the checkpoint. It never rebuilds a tokenizer from
the dataset during visualization.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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
from Parameters import vector_size
from ngram_tokenizer import NGramTokenizer, load_ngram_vocab_json
from token_embedding_bank import encode_text_units
from utils.char_pooling import (
    char_pool_predictions,
    compute_d3tw_char_pool_for_sample,
    group_records,
    token_pool_predictions,
    unit_group_records,
)
from utils.model_loading import (
    load_char_bank_if_available,
    load_image_model,
    load_sample,
    load_text_embedder,
    load_token_bank_if_available,
)


def _display_unit(unit, max_len=14):
    unit = str(unit)
    if unit == " ":
        return "sp"
    return unit if len(unit) <= max_len else unit[: max_len - 3] + "..."


def _checkpoint_config(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"  [fig13] Warning: could not read checkpoint config: {exc}")
        return {}
    return ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}


def _load_text_embedder_for_checkpoint(checkpoint_path, device):
    embedder, info = load_text_embedder(device, checkpoint_path, return_info=True)
    print(
        f"  [fig13] text embedder kind={info['text_embedder_type']} "
        f"dim={info['vector_size']} (frozen)"
    )
    return embedder, info


def _load_ngram_tokenizer_from_checkpoint(checkpoint_path):
    vocab_path = Path(checkpoint_path).resolve().parent / "ngram_vocab.json"
    if not vocab_path.is_file():
        print(f"  [fig13] Warning: ngram vocab not found: {vocab_path}")
        return None
    tokens = load_ngram_vocab_json(str(vocab_path))
    print(f"  [fig13] loaded ngram vocab: {vocab_path} ({len(tokens)} tokens)")
    return NGramTokenizer(tokens)


def _load_token_bank_from_checkpoint(checkpoint_path, device, text_embedder):
    embeddings, token_to_idx, idx_to_token, loaded_embeddings = load_token_bank_if_available(
        checkpoint_path, device, text_embedder=text_embedder, return_info=True
    )
    if idx_to_token is not None:
        print(f"  [fig13] loaded token bank ({len(idx_to_token)} tokens)")
    return embeddings, token_to_idx, idx_to_token, loaded_embeddings


def _encode_units(text, text_unit_type, checkpoint_path, device):
    text_embedder, embedder_info = _load_text_embedder_for_checkpoint(checkpoint_path, device)
    text_unit_type = str(text_unit_type).lower()
    diagnostics = {
        "text_embedder_type": embedder_info["text_embedder_type"],
        "loaded_text_embedder_state": embedder_info["loaded_text_embedder_state"],
        "loaded_token_bank_embeddings": False,
        "loaded_char_bank_embeddings": False,
    }

    if text_unit_type == "ngram":
        tokenizer = _load_ngram_tokenizer_from_checkpoint(checkpoint_path)
        if tokenizer is None:
            raise RuntimeError(
                "--text_unit_type ngram requires ngram_vocab.json saved next to the checkpoint."
            )
        units, spans, text_emb = encode_text_units(
            text=text,
            text_unit_type="ngram",
            text_embedder=text_embedder,
            device=device,
            ngram_tokenizer=tokenizer,
        )
        bank_emb, unit_to_idx, idx_to_unit, loaded_token_embeddings = _load_token_bank_from_checkpoint(
            checkpoint_path, device, text_embedder
        )
        diagnostics["loaded_token_bank_embeddings"] = loaded_token_embeddings
        return units, spans, text_emb, bank_emb, unit_to_idx, idx_to_unit, diagnostics

    if text_unit_type == "char":
        units = list(text)
        spans = [(idx, idx + 1) for idx in range(len(units))]
        bank_emb, unit_to_idx, idx_to_unit, loaded_char_embeddings = load_char_bank_if_available(
            checkpoint_path, device, return_info=True
        )
        diagnostics["loaded_char_bank_embeddings"] = loaded_char_embeddings
        if bank_emb is not None and unit_to_idx is not None and all(unit in unit_to_idx for unit in units):
            ids = torch.tensor([unit_to_idx[unit] for unit in units], device=device, dtype=torch.long)
            text_emb = normalize_func(bank_emb[ids])
        else:
            print("  [fig13] Warning: char bank unavailable/incomplete; using frozen text embedder.")
            with torch.no_grad():
                text_emb = normalize_func(text_embedder(text).to(device=device, dtype=torch.float32))
        return units, spans, text_emb, bank_emb, unit_to_idx, idx_to_unit, diagnostics

    raise ValueError("fig13 supports only --text_unit_type char|ngram.")


def _sliding_window_patches(image_width, window_size, stride, rtl=True):
    num_windows = max(1, (image_width - window_size) // stride + 1)
    patches = []
    for raw_idx in range(num_windows):
        x0 = image_width - window_size - raw_idx * stride if rtl else raw_idx * stride
        x0 = max(0, int(x0))
        x1 = min(image_width, x0 + int(window_size))
        patches.append((raw_idx, x0, x1))
    return patches


def _visual_index_ranges(num_visual, image_width, patches, use_flip):
    num_windows = len(patches)
    if num_visual == num_windows:
        return [(x0, x1) for _, x0, x1 in patches], "direct"
    if num_windows > 0 and num_visual % num_windows == 0:
        subfeatures = max(1, num_visual // num_windows)
        ranges = []
        for idx in range(num_visual):
            parent = min(num_windows - 1, idx // subfeatures)
            _, x0, x1 = patches[parent]
            ranges.append((x0, x1))
        return ranges, f"parent_window_x{subfeatures}"

    ranges = []
    for idx in range(num_visual):
        if use_flip:
            x1 = int(image_width - (idx / max(num_visual, 1)) * image_width)
            x0 = int(image_width - ((idx + 1) / max(num_visual, 1)) * image_width)
        else:
            x0 = int((idx / max(num_visual, 1)) * image_width)
            x1 = int(((idx + 1) / max(num_visual, 1)) * image_width)
        ranges.append((max(0, min(x0, x1)), min(image_width, max(x0, x1))))
    return ranges, "proportional"


def _stable_colors(labels):
    palette = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    palette += list(plt.cm.Set3(np.linspace(0, 1, 12)))
    colors = {}
    for label in labels:
        if label not in colors:
            colors[label] = palette[len(colors) % len(palette)]
    return colors


def _select_y_ticks(units, max_y_labels):
    n_units = len(units)
    if n_units == 0:
        return [], []
    step = max(1, int(np.ceil(n_units / max(1, int(max_y_labels)))))
    ticks = list(range(0, n_units, step))
    if (n_units - 1) not in ticks:
        ticks.append(n_units - 1)
    labels = [f"{idx}:{_display_unit(units[idx])}" for idx in ticks]
    return ticks, labels


def _runs_from_indices(indices, visual_ranges, merge_gap_px=2):
    ranges = [visual_ranges[i] for i in sorted(set(indices)) if 0 <= i < len(visual_ranges)]
    if not ranges:
        return []
    ranges.sort(key=lambda pair: pair[0])
    runs = []
    cur_left, cur_right = ranges[0]
    for left, right in ranges[1:]:
        if left <= cur_right + merge_gap_px:
            cur_right = max(cur_right, right)
        else:
            runs.append((cur_left, cur_right))
            cur_left, cur_right = left, right
    runs.append((cur_left, cur_right))
    return runs


def _draw_token_overlay(ax, image_np, records, visual_ranges, colors, max_labels=25, gap_px=3):
    ax.imshow(image_np, aspect="auto")
    labels_drawn = 0
    gap = max(0.0, float(gap_px))
    for record in records:
        token = record.get("token", record.get("char", ""))
        color = colors.get(token, (0.2, 0.2, 0.2, 1.0))
        for left, right in _runs_from_indices(record.get("assigned_windows", []), visual_ranges):
            draw_left = min(right, left + gap / 2.0)
            draw_right = max(draw_left, right - gap / 2.0)
            ax.axvspan(draw_left, draw_right, color=color, alpha=0.24)
            if labels_drawn < max_labels and (draw_right - draw_left) >= 6:
                ax.text(
                    (draw_left + draw_right) / 2,
                    5,
                    _display_unit(token, max_len=8),
                    ha="center",
                    va="top",
                    fontsize=6,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.35),
                )
                labels_drawn += 1


def _draw_assignment_rectangles(ax, records, colors, n_rows, n_cols):
    for record in records:
        row = record.get("unit_index", record.get("char_index"))
        if row is None or row >= n_rows:
            continue
        token = record.get("token", record.get("char", ""))
        color = colors.get(token, (0.2, 0.2, 0.2, 1.0))
        for col in record.get("assigned_windows", []):
            if 0 <= col < n_cols:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1.0, 1.0, color=color, alpha=0.20))


def _records_to_table(records, text_unit_type, max_rows):
    rows = []
    for record in records[: max(0, int(max_rows))]:
        if text_unit_type == "char":
            rows.append([
                record["char_index"],
                _display_unit(record["char"]),
                ",".join(map(str, record["assigned_windows"][:18])),
                record["num_windows"],
                _display_unit(record["top1_pred"]) if record["top1_pred"] is not None else "n/a",
                " ".join(_display_unit(u) for u in (record["top5_pred"] or [])) or "n/a",
                "" if record["correct"] is None else ("✓" if record["correct"] else "✗"),
            ])
        else:
            rows.append([
                record["unit_index"],
                _display_unit(record["token"]),
                f"{record['span'][0]}-{record['span'][1]}",
                ",".join(map(str, record["assigned_windows"][:18])),
                record["num_windows"],
                _display_unit(record["top1_pred"]) if record["top1_pred"] is not None else "n/a",
                " ".join(_display_unit(u) for u in (record["top5_pred"] or [])) or "n/a",
                "" if record["correct"] is None else ("✓" if record["correct"] else "✗"),
            ])
    if text_unit_type == "char":
        labels = ["char index", "char", "assigned visual idx", "#vis", "top1", "top-k", "ok"]
    else:
        labels = ["unit index", "token", "span", "assigned visual idx", "#vis", "top1", "top-k", "ok"]
    return rows, labels


def _write_json_csv(output_dir, stem, payload, records):
    json_path = os.path.join(output_dir, f"{stem}.json")
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "unit_index", "char_index", "token", "char", "span", "assigned_windows",
        "num_windows", "top1_pred", "top5_pred", "correct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fieldnames}
            for key in ("span", "assigned_windows", "top5_pred"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)
    return json_path, csv_path


def generate_figure(
    checkpoint,
    data_dir,
    sample_idx,
    output_dir,
    device,
    show_topk=5,
    text_unit_type=None,
    window_size=None,
    stride=None,
    max_y_labels=25,
    max_table_rows=25,
    token_mask_gap_px=3,
):
    device = torch.device(device)
    cfg = _checkpoint_config(checkpoint)
    text_unit_type = str(text_unit_type or cfg.get("text_unit_type", "char")).lower()
    if text_unit_type not in {"char", "ngram"}:
        raise ValueError("fig13 supports --text_unit_type char|ngram. Old word/bigram modes were removed.")

    model = load_image_model(checkpoint, device, window_size_override=window_size, stride_override=stride)
    image, transcript = load_sample(data_dir, sample_idx, transform=True)
    text_clean = transcript.strip()

    with torch.no_grad():
        visual_emb = normalize_func(model(image.unsqueeze(0).to(device)).squeeze(0).float())
        units, spans, text_emb, bank_emb, unit_to_idx, idx_to_unit, diagnostics = _encode_units(
            text_clean, text_unit_type, checkpoint, device
        )
        result = compute_d3tw_char_pool_for_sample(
            visual_emb=visual_emb,
            text_emb=text_emb,
            transcript_chars=units,
            detach_assignment=True,
        )

        assert text_emb.shape[0] == len(units), f"text_emb rows {text_emb.shape[0]} != len(units) {len(units)}"
        assert result["sim"].shape[0] == len(units), f"sim rows {result['sim'].shape[0]} != len(units) {len(units)}"
        assert result["assignment"].shape[0] == len(units), "assignment rows must equal number of text units"
        assert result["pooled_visual"].shape[0] == len(units), "pooled rows must equal number of text units"

        if text_unit_type == "ngram":
            predictions, pred_stats = token_pool_predictions(
                result["pooled_visual"], units, bank_emb, unit_to_idx, idx_to_unit,
                valid_mask=result["valid_mask"], topk=show_topk,
            )
            records = unit_group_records(units, spans, result["groups"], predictions)
        else:
            predictions, pred_stats = char_pool_predictions(
                pooled_visual=result["pooled_visual"], transcript_chars=units,
                char_bank_embeddings=bank_emb, char_to_idx=unit_to_idx, idx_to_char=idx_to_unit,
                valid_mask=result["valid_mask"], topk=show_topk,
            )
            records = group_records(units, result["groups"], predictions)

    sim = result["sim"].detach().cpu().numpy()
    assignment = result["assignment"].detach().cpu().numpy()
    pooled = result["pooled_visual"].detach().cpu().numpy()
    counts = result["counts"].detach().cpu().numpy()
    path = result["path"]

    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_width = int(image_np.shape[1])
    ws = int(getattr(model, "window_size", window_size or cfg.get("window_size", 16)))
    stride_value = int(getattr(model, "stride", stride or cfg.get("stride", ws)))
    use_flip = bool(getattr(model, "use_flip", False))
    patches = _sliding_window_patches(image_width, ws, stride_value, rtl=use_flip)
    visual_ranges, visual_mapping = _visual_index_ranges(
        int(visual_emb.shape[0]), image_width, patches, use_flip
    )
    if visual_mapping == "proportional":
        print(
            f"  [fig13] Warning: visual sequence length S={visual_emb.shape[0]} "
            f"!= num_windows={len(patches)}; using proportional visual-index mapping."
        )
    elif visual_mapping.startswith("parent_window"):
        print(
            f"  [fig13] visual sequence length S={visual_emb.shape[0]} "
            f"maps to num_windows={len(patches)} using {visual_mapping}."
        )

    print("[fig13 debug]")
    print("  text_unit_type:", text_unit_type)
    print("  text_embedder_type:", diagnostics["text_embedder_type"])
    print("  loaded_text_embedder_state:", diagnostics["loaded_text_embedder_state"])
    print("  loaded_token_bank_embeddings:", diagnostics["loaded_token_bank_embeddings"])
    print("  loaded_char_bank_embeddings:", diagnostics["loaded_char_bank_embeddings"])
    print("  text:", text_clean)
    print("  units:", units)
    print("  spans:", spans)
    print("  num_units:", len(units))
    print("  visual_sequence_len:", int(visual_emb.shape[0]))
    print("  sim shape:", tuple(result["sim"].shape))
    print("  assignment shape:", tuple(result["assignment"].shape))
    print("  pooled shape:", tuple(result["pooled_visual"].shape))
    print("  token/char bank size:", 0 if unit_to_idx is None else len(unit_to_idx))

    if pred_stats.get("token_pool_top1") is not None and pred_stats["token_pool_top1"] <= 0.05:
        print(
            "  [fig13] Warning: token retrieval is very low "
            f"(top1={pred_stats['token_pool_top1']:.3f}, top5={pred_stats['token_pool_top5']:.3f}). "
            "Check that the checkpoint is trained and token_bank.json matches it."
        )

    colors = _stable_colors([record.get("token", record.get("char", "")) for record in records])
    y_ticks, y_labels = _select_y_ticks(units, max_y_labels)

    fig = plt.figure(figsize=(28, 22), constrained_layout=False)
    grid = fig.add_gridspec(5, 2, height_ratios=[2.1, 1.1, 4.0, 2.8, 4.0], width_ratios=[1.25, 1.0], hspace=0.42, wspace=0.18)

    ax_img = fig.add_subplot(grid[0, :])
    _draw_token_overlay(ax_img, image_np, records, visual_ranges, colors, max_labels=max_table_rows, gap_px=token_mask_gap_px)
    ax_img.set_title(
        f"D3TW-assigned {text_unit_type} text-unit regions — sample {sample_idx} "
        f"(num_windows={len(patches)}, visual S={visual_emb.shape[0]}, window={ws}, stride={stride_value}, mapping={visual_mapping})",
        fontsize=15,
    )
    ax_img.axis("off")

    ax_pipe = fig.add_subplot(grid[1, :])
    ax_pipe.axis("off")
    ax_pipe.text(
        0.5, 0.5,
        "V[S,D] + E[K,D]  →  sim = E @ Vᵀ [K,S]  →  D3TW path  →  "
        "assignment A[K,S]  →  pooled M = normalized(A) @ V  →  token InfoNCE",
        ha="center", va="center", fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#eef3fb", ec="#4c72b0"),
    )

    ax_sim = fig.add_subplot(grid[2, 0])
    im = ax_sim.imshow(sim, aspect="auto", origin="upper", cmap="viridis")
    _draw_assignment_rectangles(ax_sim, records, colors, sim.shape[0], sim.shape[1])
    if path:
        rows = [j for j, _ in path]
        cols = [i for _, i in path]
        ax_sim.plot(cols, rows, color="red", linewidth=1.5, label="hard D3TW path")
        ax_sim.legend(loc="upper right", fontsize=9)
    ax_sim.set_title("Pre-pooling similarity sim[j,i] = E[j] · V[i]", fontsize=13)
    ax_sim.set_xlabel("visual position index i")
    ax_sim.set_ylabel("text-unit index j")
    ax_sim.set_yticks(y_ticks)
    ax_sim.set_yticklabels(y_labels, fontsize=8)
    fig.colorbar(im, ax=ax_sim, fraction=0.025, pad=0.01)

    ax_assign = fig.add_subplot(grid[2, 1])
    im_a = ax_assign.imshow(assignment, aspect="auto", origin="upper", cmap="Greys")
    _draw_assignment_rectangles(ax_assign, records, colors, assignment.shape[0], assignment.shape[1])
    ax_assign.set_title(f"D3TW assignment matrix A [K,S] = {assignment.shape}", fontsize=13)
    ax_assign.set_xlabel("visual position index i")
    ax_assign.set_ylabel("text-unit index j")
    ax_assign.set_yticks(y_ticks)
    ax_assign.set_yticklabels(y_labels, fontsize=8)
    fig.colorbar(im_a, ax=ax_assign, fraction=0.035, pad=0.01)

    ax_pool = fig.add_subplot(grid[3, :])
    im_p = ax_pool.imshow(pooled, aspect="auto", origin="upper", cmap="coolwarm")
    ax_pool.set_title(
        f"Pooled text-unit vectors M = normalized(A) @ V (shape={pooled.shape[0]}×{pooled.shape[1]})",
        fontsize=13,
    )
    ax_pool.set_xlabel("embedding dimension")
    ax_pool.set_ylabel("text-unit index")
    ax_pool.set_yticks(y_ticks)
    ax_pool.set_yticklabels(y_labels, fontsize=8)
    fig.colorbar(im_p, ax=ax_pool, fraction=0.015, pad=0.01)

    ax_table = fig.add_subplot(grid[4, :])
    ax_table.axis("off")
    table_rows, col_labels = _records_to_table(records, text_unit_type, max_table_rows)
    table = ax_table.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.20)
    top1 = pred_stats.get("token_pool_top1", pred_stats.get("char_pool_top1"))
    topk_value = pred_stats.get("token_pool_top5", pred_stats.get("char_pool_top5"))
    metric_text = f"top1={top1:.3f}, top{show_topk}={topk_value:.3f}" if top1 is not None and topk_value is not None else "bank unavailable"
    ax_table.set_title(
        f"{text_unit_type} text-unit/window groups from D3TW assignment "
        f"(showing {min(max_table_rows, len(records))}/{len(records)}) | {metric_text}",
        fontsize=13,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig13_d3tw_character_pooling_sample_{sample_idx}"
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    png_path = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

    payload = {
        "text_unit_type": text_unit_type,
        "text": text_clean,
        "sample_idx": int(sample_idx),
        "num_units": int(len(units)),
        "num_visual": int(visual_emb.shape[0]),
        "num_windows": int(len(patches)),
        "window_size": int(ws),
        "stride": int(stride_value),
        "visual_index_mapping": visual_mapping,
        "sim_shape": [int(v) for v in sim.shape],
        "assignment_shape": [int(v) for v in assignment.shape],
        "pooled_shape": [int(v) for v in pooled.shape],
        "token_top1": pred_stats.get("token_pool_top1"),
        "token_top5": pred_stats.get("token_pool_top5"),
        "char_top1": pred_stats.get("char_pool_top1"),
        "char_top5": pred_stats.get("char_pool_top5"),
        "counts": [float(v) for v in counts.tolist()],
        "units": records,
    }
    json_path, csv_path = _write_json_csv(output_dir, stem, payload, records)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_overlap_mode", default=None, help="Ignored; kept for backward-compatible calls.")
    parser.add_argument("--window_size", type=int, default=None, help="Override checkpoint window size for figure generation.")
    parser.add_argument("--stride", type=int, default=None, help="Override checkpoint sliding-window stride for figure generation.")
    parser.add_argument("--show_topk", type=int, default=5)
    parser.add_argument("--text_unit_type", default=None, choices=["char", "ngram"], help="Default: checkpoint model_config text_unit_type.")
    parser.add_argument("--max_y_labels", type=int, default=25)
    parser.add_argument("--max_table_rows", type=int, default=25)
    parser.add_argument("--token_mask_gap_px", type=float, default=3)
    args = parser.parse_args()

    indices = [args.sample_idx] if args.sample_idx is not None else args.sample_indices
    for idx in indices:
        generate_figure(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            sample_idx=idx,
            output_dir=args.output_dir,
            device=args.device,
            show_topk=args.show_topk,
            text_unit_type=args.text_unit_type,
            window_size=args.window_size,
            stride=args.stride,
            max_y_labels=args.max_y_labels,
            max_table_rows=args.max_table_rows,
            token_mask_gap_px=args.token_mask_gap_px,
        )


if __name__ == "__main__":
    main()

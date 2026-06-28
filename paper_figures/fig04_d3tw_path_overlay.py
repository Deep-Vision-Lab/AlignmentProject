"""D3TW hard path overlay plus character-window grouping table."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from utils.sample_data import make_sample, pad_text
from token_bank import build_adjacent_pair_visuals
from token_embedding_bank import encode_text_units


def _display_char(char):
    return "sp" if char == " " else char


def _load_input(data_dir, sample_idx, sentence_idx):
    if data_dir:
        image, text = load_sample(data_dir, sample_idx, transform=True)
        return image, text, sample_idx
    image, text = make_sample(sentence_idx, transform=True)
    return image, text, sentence_idx


def _text_embeddings(chars, checkpoint, device):
    bank_emb, char_to_idx, idx_to_char = load_char_bank_if_available(checkpoint, device)
    if bank_emb is not None and char_to_idx is not None and all(ch in char_to_idx for ch in chars):
        ids = torch.tensor([char_to_idx[ch] for ch in chars], device=device, dtype=torch.long)
        return bank_emb[ids], bank_emb, char_to_idx, idx_to_char

    print("  [fig04] Warning: char bank unavailable/incomplete; falling back to text embedder.")
    embedder = load_text_embedder(device)
    with torch.no_grad():
        return normalize_func(embedder("".join(chars)).to(device)), bank_emb, char_to_idx, idx_to_char


def draw_path_overlay(
    checkpoint,
    output_dir,
    device,
    data_dir=None,
    sample_idx=0,
    sentence_idx=0,
    show_topk=5,
    text_unit_type="char",
):
    device = torch.device(device)
    image, text, shown_idx = _load_input(data_dir, sample_idx, sentence_idx)
    text_padded = pad_text(text)
    text_unit_type = str(text_unit_type).lower()

    model = load_image_model(checkpoint, device)
    with torch.no_grad():
        visual_emb = normalize_func(model(image.unsqueeze(0).to(device)).squeeze(0).float())
        if text_unit_type == "ngram":
            ngram_tokenizer = load_ngram_tokenizer_if_available(checkpoint)
            if ngram_tokenizer is None:
                raise RuntimeError("fig04 --text_unit_type ngram requires ngram_vocab.json next to the checkpoint.")
            text_embedder = load_text_embedder(device)
            units, spans, text_emb = encode_text_units(
                text_padded, "ngram", text_embedder, device, ngram_tokenizer=ngram_tokenizer
            )
            bank_emb, unit_to_idx, idx_to_unit = load_token_bank_if_available(checkpoint, device)
        else:
            units = list(text_padded)
            spans = [(idx, idx + 1) for idx in range(len(units))]
            text_emb, bank_emb, unit_to_idx, idx_to_unit = _text_embeddings(units, checkpoint, device)
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
                result["pooled_visual"],
                units,
                bank_emb,
                unit_to_idx,
                idx_to_unit,
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
    path = result["path"]
    if text_unit_type == "ngram":
        records = unit_group_records(units, spans, result["groups"], predictions)
    else:
        records = group_records(units, result["groups"], predictions)
    assert tuple(result["pooled_visual"].shape) == (sim.shape[0], visual_emb.shape[1])

    fig_h = max(14, sim.shape[0] * 0.45 + 8)
    fig = plt.figure(figsize=(28, fig_h))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.0, 2.4], hspace=0.25)

    ax = fig.add_subplot(grid[0])
    im = ax.imshow(sim, aspect="auto", origin="upper", cmap="viridis")
    if path:
        ax.plot([i for _, i in path], [j for j, _ in path], color="red", linewidth=1.8)
    ax.set_title(
        f"D3TW path over pre-pooling similarity matrix — sample {shown_idx}\n"
        f"S[j,i] = similarity between transcript {text_unit_type} unit j and visual window i",
        fontsize=13,
    )
    ax.set_xlabel("visual window/subfeature index i")
    ax.set_ylabel(f"transcript {text_unit_type} unit index j")
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([_display_char(ch) for ch in units], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="cosine similarity")

    ax_tbl = fig.add_subplot(grid[1])
    ax_tbl.axis("off")
    if text_unit_type == "ngram":
        rows = [
            [
                r["unit_index"],
                _display_char(r["token"]),
                f"{r['span'][0]}-{r['span'][1]}",
                ",".join(map(str, r["assigned_windows"])),
                r["num_windows"],
                _display_char(r["top1_pred"]) if r["top1_pred"] is not None else "n/a",
                " ".join(_display_char(ch) for ch in r["top5_pred"]) if r["top5_pred"] else "n/a",
                "" if r["correct"] is None else ("✓" if r["correct"] else "✗"),
            ]
            for r in records[: min(44, len(records))]
        ]
        col_labels = ["unit index", "token", "span", "assigned windows", "# windows", "top1", "top5", "correct"]
    else:
        rows = [
            [
                r["char_index"],
                _display_char(r["char"]),
                ",".join(map(str, r["assigned_windows"])),
                r["num_windows"],
                _display_char(r["top1_pred"]) if r["top1_pred"] is not None else "n/a",
                " ".join(_display_char(ch) for ch in r["top5_pred"]) if r["top5_pred"] else "n/a",
                "" if r["correct"] is None else ("✓" if r["correct"] else "✗"),
            ]
            for r in records[: min(44, len(records))]
        ]
        col_labels = ["char index", "character", "assigned window indices", "# windows", "top1", "top5", "correct"]
    if bigram_rows:
        rows.append(["—", "BIGRAMS", "—", "—", "—", "—", "—"])
        for idx, r in enumerate(bigram_rows[:12]):
            rows.append([
                f"p{idx}",
                r["token"],
                f"{r['start_char_index']}-{r['end_char_index']}",
                "2 chars",
                r["top1_pred"] or "n/a",
                " ".join(r["top5_pred"]) if r["top5_pred"] else "n/a",
                "✓" if r["correct"] else "✗",
            ])
    table = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax_tbl.set_title(
        f"{text_unit_type} text-unit/window groups from D3TW assignment"
        + (
            f" | top1={pred_stats.get('char_pool_top1', pred_stats.get('token_pool_top1')):.3f}, "
            f"top5={pred_stats.get('char_pool_top5', pred_stats.get('token_pool_top5')):.3f}"
            if pred_stats.get("char_pool_top1", pred_stats.get("token_pool_top1")) is not None
            else " | bank unavailable"
        ),
        fontsize=12,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = f"fig04_d3tw_path_overlay_sample_{shown_idx}"
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.pdf in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sentence_idx", type=int, default=None)
    parser.add_argument("--sentence_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output_dir", default="paper_figures/outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show_topk", type=int, default=5)
    parser.add_argument("--text_unit_type", default="char", choices=["char", "ngram"])
    args = parser.parse_args()
    os.chdir(_ROOT)
    if args.data_dir:
        indices = [args.sample_idx] if args.sample_idx is not None else args.sample_indices
        for idx in indices:
            draw_path_overlay(
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                data_dir=args.data_dir,
                sample_idx=idx,
                sentence_idx=0,
                show_topk=args.show_topk,
                text_unit_type=args.text_unit_type,
            )
    else:
        indices = [args.sentence_idx] if args.sentence_idx is not None else args.sentence_indices
        for idx in indices:
            draw_path_overlay(
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                device=args.device,
                data_dir=None,
                sample_idx=0,
                sentence_idx=idx,
                show_topk=args.show_topk,
                text_unit_type=args.text_unit_type,
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose whether ViT image windows live near the correct Arabic text units.

The script compares image-window embeddings against a bank of single-character
text embeddings from the checkpoint's text encoder. Because the current
Synthetic63 layout contains transcripts but no per-character bounding boxes,
window labels used for Top-k metrics are *sequence-reference labels* obtained
from the checkpoint's hard Span-DTW path. They are not claimed as geometric
character ground truth.

Only hard-path steps whose core text is exactly one non-space character are used
for quantitative retrieval metrics. Multi-character spans, spaces, and blanks
are skipped. PCA is fit jointly to sampled image windows and the character
prototype embeddings, then visualized in one common 2-D basis.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Parameters as P
P.export_environment()

from Evaluation.vit_evaluation import install_vit_evaluation_loader
install_vit_evaluation_loader()

from Evaluation._eval_utils import get_image_features, load_evaluation_models
from span_alignment_loss import hard_span_dtw_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA + nearest-token diagnostic for image-window embeddings"
    )
    parser.add_argument("--dataset", required=True, help="Synthetic63-style dataset root")
    parser.add_argument("--weights", required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--feature",
        choices=("contextual", "local", "grouped"),
        default=str(getattr(P, "evaluation_feature", "contextual")),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(getattr(P, "evaluation_n_samples", 50)),
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--max-pca-windows", type=int, default=5000)
    parser.add_argument("--top-k-confusion", type=int, default=30)
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def synthetic_records(root: Path, start_index: int, n_samples: int):
    images = root / "images"
    texts = root / "texts"
    if not images.is_dir() or not texts.is_dir():
        raise ValueError(
            "This diagnostic currently expects Synthetic63-style images/ and texts/ folders."
        )
    records = []
    stop = start_index + max(0, n_samples)
    for index in range(max(1, start_index), stop):
        for side in (1, 2):
            image = images / f"img{side}_{index}.png"
            text = texts / f"text{side}_{index}.txt"
            if image.is_file() and text.is_file():
                records.append((index, side, image, text))
    if not records:
        raise ValueError(f"No Synthetic63 samples found under {root}")
    return records


def encode_character(text_model, char: str) -> torch.Tensor:
    """Return one normalized embedding for one visible character."""
    with torch.no_grad():
        encoded = text_model(char)
    if torch.is_tensor(encoded):
        if encoded.ndim != 2 or encoded.shape[0] == 0:
            raise ValueError(f"Text encoder produced no embedding for {char!r}")
        vector = encoded[0]
    else:
        embeddings = encoded.embeddings
        starts = list(getattr(encoded, "starts", []))
        lengths = list(getattr(encoded, "lengths", []))
        raw_texts = getattr(encoded, "raw_texts", None)
        candidates = []
        for i, (start, length) in enumerate(zip(starts, lengths)):
            if int(start) == 0 and int(length) == 1:
                if raw_texts is None or str(raw_texts[i]).strip() == char:
                    candidates.append(i)
        if not candidates:
            candidates = [
                i for i, length in enumerate(lengths)
                if int(length) == 1 and str(encoded.texts[i]).strip() == char
            ]
        if not candidates:
            raise ValueError(f"No single-character span embedding found for {char!r}")
        vector = embeddings[candidates[0]]
    return F.normalize(vector.float(), p=2, dim=-1)


def reference_labels(span_encoding, image_embeddings: torch.Tensor):
    """Map windows to single characters using the hard Span-DTW sequence path."""
    with torch.no_grad():
        path = hard_span_dtw_path(span_encoding, image_embeddings)
    labels = {}
    for step in path:
        if bool(step.get("is_blank", False)) or bool(step.get("is_space", False)):
            continue
        raw = str(step.get("raw_text", step.get("text", ""))).strip()
        if len(raw) != 1 or raw.isspace():
            continue
        for window in range(int(step["window_start"]), int(step["window_end"])):
            labels[window] = raw
    return labels, path


def fit_pca_2d(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    projected = centered @ basis
    denom = max(1, matrix.shape[0] - 1)
    eigenvalues = (singular ** 2) / denom
    total = float(eigenvalues.sum())
    ratio = eigenvalues[:2] / total if total > 0 else np.zeros(2)
    return projected.astype(np.float32), ratio.astype(np.float64)


def save_pca_plot(output: Path, window_xy, window_labels, proto_xy, chars, ratio):
    fig, ax = plt.subplots(figsize=(14, 10))
    labels = np.asarray(window_labels, dtype=object)
    for char in chars:
        mask = labels == char
        if np.any(mask):
            ax.scatter(window_xy[mask, 0], window_xy[mask, 1], s=12, alpha=0.45, label=char)
    ax.scatter(proto_xy[:, 0], proto_xy[:, 1], s=110, marker="X", edgecolors="black", linewidths=0.8)
    for i, char in enumerate(chars):
        ax.annotate(char, (proto_xy[i, 0], proto_xy[i, 1]), fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({100.0 * ratio[0]:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * ratio[1]:.2f}% variance)")
    ax.set_title("Image windows and Arabic character prototypes in one PCA space")
    if len(chars) <= 20:
        ax.legend(title="Span-DTW reference character", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_confusion(output: Path, confusion, chars, counts, limit):
    ranked = sorted(chars, key=lambda c: counts[c], reverse=True)[: max(1, limit)]
    index = {char: i for i, char in enumerate(ranked)}
    matrix = np.zeros((len(ranked), len(ranked)), dtype=np.int64)
    for (truth, pred), count in confusion.items():
        if truth in index and pred in index:
            matrix[index[truth], index[pred]] += int(count)
    fig, ax = plt.subplots(figsize=(max(9, len(ranked) * 0.45), max(8, len(ranked) * 0.45)))
    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax, label="window count")
    ax.set_xticks(range(len(ranked)), ranked, rotation=90)
    ax.set_yticks(range(len(ranked)), ranked)
    ax.set_xlabel("nearest character prototype")
    ax.set_ylabel("Span-DTW reference character")
    ax.set_title("Window-to-character retrieval confusion")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    job_name = weights.parent.name
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "Results" / "Evaluation" / "TokenPCA" / job_name
    )
    output.mkdir(parents=True, exist_ok=True)

    models = load_evaluation_models(weights, device=args.device, load_text_model=True)
    if models.text_model is None:
        raise RuntimeError("Checkpoint evaluation loader did not construct a text encoder")

    records = synthetic_records(dataset, args.start_index, args.n_samples)
    transcripts = [(record, read_text(record[3])) for record in records]
    chars = sorted({char for _record, text in transcripts for char in text if not char.isspace()})
    if not chars:
        raise ValueError("No visible transcript characters found")

    prototypes = torch.stack([encode_character(models.text_model, char) for char in chars], dim=0)
    prototypes = F.normalize(prototypes.float(), p=2, dim=-1)

    rows = []
    all_windows = []
    all_labels = []
    confusion = Counter()
    reference_counts = Counter()
    rank_values = []
    correct_cosines = []
    hardest_wrong_cosines = []

    for (record, transcript) in transcripts:
        index, side, image_path, _text_path = record
        features = get_image_features(models, image_path, "synthetic")
        image_embeddings = F.normalize(features.select(args.feature).float(), p=2, dim=-1)
        with torch.no_grad():
            span_encoding = models.text_model(transcript)
        if torch.is_tensor(span_encoding):
            raise RuntimeError(
                "Sequence-reference labeling requires the arabic_span text encoder; "
                "this checkpoint uses a token/char encoder."
            )
        labels, _path = reference_labels(span_encoding, image_embeddings)
        if not labels:
            print(f"[{index}:{side}] no single-character reference windows; skipped", flush=True)
            continue

        similarity = image_embeddings @ prototypes.to(image_embeddings.device).T
        order = torch.argsort(similarity, dim=1, descending=True)

        for window, truth in sorted(labels.items()):
            if truth not in chars or window >= image_embeddings.shape[0]:
                continue
            truth_idx = chars.index(truth)
            ranked = order[window].detach().cpu().tolist()
            rank = ranked.index(truth_idx) + 1
            pred_idx = ranked[0]
            pred = chars[pred_idx]
            correct_cosine = float(similarity[window, truth_idx].item())
            if len(chars) > 1:
                wrong = similarity[window].clone()
                wrong[truth_idx] = -float("inf")
                hardest_wrong = float(wrong.max().item())
            else:
                hardest_wrong = float("nan")

            rank_values.append(rank)
            correct_cosines.append(correct_cosine)
            if np.isfinite(hardest_wrong):
                hardest_wrong_cosines.append(hardest_wrong)
            confusion[(truth, pred)] += 1
            reference_counts[truth] += 1
            all_windows.append(image_embeddings[window].detach().cpu().numpy())
            all_labels.append(truth)
            rows.append({
                "sample": int(index),
                "side": int(side),
                "window": int(window),
                "reference_char": truth,
                "nearest_char": pred,
                "rank": int(rank),
                "top1_correct": int(rank == 1),
                "top3_correct": int(rank <= 3),
                "top5_correct": int(rank <= 5),
                "correct_cosine": correct_cosine,
                "hardest_wrong_cosine": hardest_wrong,
                "margin": correct_cosine - hardest_wrong if np.isfinite(hardest_wrong) else float("nan"),
                "image": str(image_path),
            })
        print(f"[{index}:{side}] windows={len(labels)} accumulated={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No evaluable single-character windows were produced")

    with (output / "nearest_tokens_per_window.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_token = []
    for char in chars:
        token_rows = [row for row in rows if row["reference_char"] == char]
        if not token_rows:
            continue
        per_token.append({
            "character": char,
            "windows": len(token_rows),
            "top1": float(np.mean([row["top1_correct"] for row in token_rows])),
            "top3": float(np.mean([row["top3_correct"] for row in token_rows])),
            "top5": float(np.mean([row["top5_correct"] for row in token_rows])),
            "mean_rank": float(np.mean([row["rank"] for row in token_rows])),
            "mean_correct_cosine": float(np.mean([row["correct_cosine"] for row in token_rows])),
            "mean_margin": float(np.nanmean([row["margin"] for row in token_rows])),
        })
    with (output / "per_token_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_token[0].keys()))
        writer.writeheader()
        writer.writerows(per_token)

    windows_np = np.asarray(all_windows, dtype=np.float32)
    labels_np = np.asarray(all_labels, dtype=object)
    if windows_np.shape[0] > args.max_pca_windows:
        rng = np.random.default_rng(0)
        chosen = np.sort(rng.choice(windows_np.shape[0], args.max_pca_windows, replace=False))
        windows_np = windows_np[chosen]
        labels_np = labels_np[chosen]
    proto_np = prototypes.detach().cpu().numpy()
    combined = np.concatenate([windows_np, proto_np], axis=0)
    projected, explained = fit_pca_2d(combined)
    window_xy = projected[: len(windows_np)]
    proto_xy = projected[len(windows_np):]
    save_pca_plot(output / "pca_tokens_and_windows.png", window_xy, labels_np, proto_xy, chars, explained)
    save_confusion(output / "token_confusion_matrix.png", confusion, chars, reference_counts, args.top_k_confusion)

    ranks = np.asarray(rank_values, dtype=np.int64)
    summary = {
        "weights": str(weights),
        "dataset": str(dataset),
        "feature": args.feature,
        "reference": "hard Span-DTW single-character steps (sequence reference, not geometric GT)",
        "evaluated_windows": int(len(rows)),
        "characters": int(len(chars)),
        "top1_accuracy": float(np.mean(ranks <= 1)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "top5_accuracy": float(np.mean(ranks <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mean_correct_cosine": float(np.mean(correct_cosines)),
        "mean_hardest_wrong_cosine": float(np.mean(hardest_wrong_cosines)) if hardest_wrong_cosines else None,
        "mean_margin": (
            float(np.mean(np.asarray(correct_cosines) - np.asarray(hardest_wrong_cosines)))
            if hardest_wrong_cosines and len(hardest_wrong_cosines) == len(correct_cosines)
            else None
        ),
        "pca_pc1_explained_variance": float(explained[0]),
        "pca_pc2_explained_variance": float(explained[1]),
        "pca_first2_explained_variance": float(explained.sum()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved diagnostics to: {output}")


if __name__ == "__main__":
    main()

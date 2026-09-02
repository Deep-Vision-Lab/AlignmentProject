#!/usr/bin/env python3
"""Diagnose whether image windows live near the correct Arabic character embeddings.

PCA is used only for visualization.  The actual token decision is cosine nearest
neighbour in the original shared embedding space.

Reference window labels are alignment-conditioned: the checkpoint's hard
Span-DTW path assigns image windows to text spans, and quantitative character
metrics use only non-blank one-character spans.  This is deliberately reported
as an alignment-conditioned diagnostic rather than independent character-box
accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

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

from unified_line_geometry import install_evaluation_geometry
install_evaluation_geometry()

from Evaluation.vit_evaluation import install_vit_evaluation_loader
install_vit_evaluation_loader()

from Evaluation._eval_utils import (
    align_text_to_windows,
    get_image_features,
    load_evaluation_models,
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Character/token retrieval + PCA diagnostic for ViT image windows."
    )
    parser.add_argument("--dataset", required=True, help="Synthetic dataset root containing images/ and texts/.")
    parser.add_argument("--weights", required=True, help="Checkpoint path, e.g. Weights/vit_baseline/model_best.pth")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--side", choices=("1", "2", "both"), default="both")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--min-ink", type=float, default=0.02)
    parser.add_argument("--pca-max-windows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=int(getattr(P, "dataset_split_seed", 42)))
    parser.add_argument("--include-punctuation", action="store_true")
    return parser.parse_args()


def _find_image(images: Path, side: int, index: int) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = images / f"img{side}_{index}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _discover_samples(root: Path, start: int, count: int, side_mode: str):
    images, texts = root / "images", root / "texts"
    if not images.is_dir() or not texts.is_dir():
        raise ValueError(f"Expected synthetic dataset with images/ and texts/: {root}")
    sides = (1, 2) if side_mode == "both" else (int(side_mode),)
    found = []
    index = max(1, int(start))
    wanted = max(1, int(count))
    misses = 0
    while len(found) < wanted and misses < 1000:
        any_found = False
        for side in sides:
            image = _find_image(images, side, index)
            text_path = texts / f"text{side}_{index}.txt"
            if image is not None and text_path.is_file():
                found.append((index, side, image, text_path))
                any_found = True
                if len(found) >= wanted:
                    break
        misses = 0 if any_found else misses + 1
        index += 1
    if not found:
        raise RuntimeError(f"No image/text samples found under {root}")
    return found


def _valid_unit(ch: str, include_punctuation: bool) -> bool:
    if not ch or ch.isspace():
        return False
    if include_punctuation:
        return True
    return ch.isalpha() or ("\u0600" <= ch <= "\u06ff" and ch not in "،؛؟ـ")


def _prototype_for_character(text_model, ch: str) -> torch.Tensor:
    with torch.no_grad():
        encoding = text_model(ch)
    candidates = []
    raw = getattr(encoding, "raw_texts", None)
    blanks = getattr(encoding, "is_blank", None)
    spaces = getattr(encoding, "is_space", None)
    for i, (start, length) in enumerate(zip(encoding.starts, encoding.lengths)):
        if int(start) != 0 or int(length) != 1:
            continue
        if blanks is not None and bool(blanks[i]):
            continue
        if spaces is not None and bool(spaces[i]):
            continue
        raw_text = raw[i] if raw is not None else ch
        if str(raw_text) == ch:
            candidates.append(i)
    if not candidates:
        raise RuntimeError(f"Text encoder did not expose a one-character span for {ch!r}")
    vector = encoding.embeddings[candidates[0]].float()
    return F.normalize(vector, p=2, dim=-1)


def _character_prototypes(models, vocabulary):
    if models.text_model is None:
        raise RuntimeError("Checkpoint text encoder is required for this diagnostic")
    text_type = str(models.config.get("text_encoder_type", "arabic_span"))
    if text_type != "arabic_span":
        raise RuntimeError(
            f"This diagnostic currently requires text_encoder_type='arabic_span', got {text_type!r}"
        )
    vectors, kept = [], []
    for ch in vocabulary:
        try:
            vectors.append(_prototype_for_character(models.text_model, ch))
            kept.append(ch)
        except Exception as exc:
            print(f"Skipping prototype {ch!r}: {exc}", flush=True)
    if not vectors:
        raise RuntimeError("No character prototypes could be encoded")
    return kept, torch.stack(vectors, dim=0)


def _alignment_labels(models, text: str, features, vocabulary: set[str]):
    _prepared, _encoding, path = align_text_to_windows(models, text, features, True)
    labels = {}
    for step in path:
        if bool(step.get("is_blank", False)) or bool(step.get("is_space", False)):
            continue
        raw = str(step.get("raw_text", step.get("text", "")))
        if len(raw) != 1 or raw not in vocabulary:
            continue
        w0, w1 = int(step["window_start"]), int(step["window_end"])
        for window in range(w0, w1):
            labels[window] = raw
    return labels, path


def _rank_metrics(similarities: np.ndarray, candidate_labels: list[str], reference: str):
    order = np.argsort(-similarities)
    ranked = [candidate_labels[int(i)] for i in order]
    rank = ranked.index(reference) + 1
    correct_idx = candidate_labels.index(reference)
    correct_score = float(similarities[correct_idx])
    wrong = [float(similarities[i]) for i, label in enumerate(candidate_labels) if label != reference]
    hardest_wrong = max(wrong) if wrong else float("nan")
    return {
        "predicted": ranked[0],
        "rank": rank,
        "top1": rank <= 1,
        "top3": rank <= 3,
        "top5": rank <= 5,
        "rr": 1.0 / rank,
        "correct_cosine": correct_score,
        "top1_cosine": float(similarities[order[0]]),
        "hardest_wrong_cosine": hardest_wrong,
        "correct_margin": correct_score - hardest_wrong if wrong else float("nan"),
        "top5_labels": ranked[:5],
        "top5_scores": [float(similarities[i]) for i in order[:5]],
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _finite_mean(values):
    values = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _pca_2d(matrix: np.ndarray):
    centered = matrix.astype(np.float64) - matrix.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    variance = singular ** 2
    ratio = variance / max(float(variance.sum()), 1e-12)
    return coords.astype(np.float32), ratio[:2].astype(np.float64)


def _save_pca(output: Path, prototype_labels, prototype_vectors, window_rows, window_vectors, max_windows, seed):
    if not window_rows:
        return None
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(window_rows))
    if len(indices) > max_windows:
        indices = np.sort(rng.choice(indices, size=max_windows, replace=False))
    sampled = window_vectors[indices]
    rows = [window_rows[int(i)] for i in indices]
    combined = np.concatenate([prototype_vectors, sampled], axis=0)
    coords, ratio = _pca_2d(combined)
    p = len(prototype_labels)
    proto_xy, win_xy = coords[:p], coords[p:]

    fig, ax = plt.subplots(figsize=(14, 11))
    label_to_index = {label: i for i, label in enumerate(prototype_labels)}
    cmap = plt.get_cmap("tab20")
    for label in prototype_labels:
        mask = np.asarray([row["reference"] == label for row in rows], dtype=bool)
        if not mask.any():
            continue
        idx = label_to_index[label]
        ax.scatter(win_xy[mask, 0], win_xy[mask, 1], s=16, alpha=0.38, color=cmap(idx % 20))
    ax.scatter(proto_xy[:, 0], proto_xy[:, 1], s=130, marker="X", edgecolors="black", linewidths=0.8)
    for label, (x, y) in zip(prototype_labels, proto_xy):
        ax.annotate(label, (float(x), float(y)), xytext=(4, 4), textcoords="offset points", fontsize=11, fontweight="bold")
    ax.set_title("PCA: alignment-conditioned image windows around character prototypes")
    ax.set_xlabel(f"PC1 ({100.0 * ratio[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * ratio[1]:.1f}% variance)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output / "pca_reference_characters.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    coord_rows = []
    for label, xy in zip(prototype_labels, proto_xy):
        coord_rows.append({"kind": "prototype", "label": label, "pc1": float(xy[0]), "pc2": float(xy[1])})
    for row, xy in zip(rows, win_xy):
        coord_rows.append({"kind": "window", "label": row["reference"], "pc1": float(xy[0]), "pc2": float(xy[1])})
    _write_csv(output / "pca_coordinates.csv", coord_rows, ["kind", "label", "pc1", "pc2"])
    return {"pc1_variance_ratio": float(ratio[0]), "pc2_variance_ratio": float(ratio[1]), "windows_plotted": len(rows)}


def _save_confusion(output: Path, labels: list[str], rows: list[dict]):
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for row in rows:
        if row["reference"] in index and row["predicted"] in index:
            matrix[index[row["reference"]], index[row["predicted"]]] += 1
    np.save(output / "confusion_matrix.npy", matrix)
    if not matrix.size:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.42), max(9, len(labels) * 0.42)))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Nearest character prototype")
    ax.set_ylabel("Alignment-conditioned reference character")
    ax.set_title("Window → character retrieval confusion")
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(output / "character_confusion_matrix.png", dpi=180)
    plt.close(fig)


def _save_mean_similarity(output: Path, labels: list[str], rows: list[dict], all_similarity: np.ndarray):
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=np.float32)
    for i, label in enumerate(labels):
        selected = [k for k, row in enumerate(rows) if row["reference"] == label]
        if selected:
            matrix[i] = np.mean(all_similarity[selected], axis=0)
    np.save(output / "mean_similarity_by_reference.npy", matrix)
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.42), max(9, len(labels) * 0.42)))
    image = ax.imshow(matrix, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Character prototype")
    ax.set_ylabel("Reference character")
    ax.set_title("Mean cosine similarity: labeled windows vs character prototypes")
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(output / "character_similarity_heatmap.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    samples = _discover_samples(dataset, args.start_index, args.n_samples, args.side)

    texts = [path.read_text(encoding="utf-8").strip() for _idx, _side, _img, path in samples]
    vocabulary = sorted({ch for text in texts for ch in text if _valid_unit(ch, args.include_punctuation)})
    if len(vocabulary) < 2:
        raise RuntimeError(f"Need at least two character classes, found {vocabulary}")

    models = load_evaluation_models(weights, args.device, load_text_model=True)
    labels, prototype_tensor = _character_prototypes(models, vocabulary)
    label_set = set(labels)
    prototype_tensor = F.normalize(prototype_tensor.float(), p=2, dim=-1)

    job_name = weights.parent.name
    dataset_name = dataset.name
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "Results" / "Evaluation" / "TokenEmbedding" / dataset_name / job_name / args.feature
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    vectors = []
    similarities_all = []
    skipped_alignment = 0
    for position, (dataset_index, side, image, text_path) in enumerate(samples, 1):
        text = text_path.read_text(encoding="utf-8").strip()
        features = get_image_features(models, image, "synthetic")
        visual = F.normalize(features.select(args.feature).float(), p=2, dim=-1)
        reference, _path = _alignment_labels(models, text, features, label_set)
        similarity = (visual @ prototype_tensor.T).detach().cpu().numpy()
        ink = features.ink.detach().cpu().numpy()
        visual_np = visual.detach().cpu().numpy()

        used = 0
        for window_index, ref in sorted(reference.items()):
            if window_index < 0 or window_index >= similarity.shape[0]:
                continue
            if float(ink[window_index]) < float(args.min_ink):
                continue
            metrics = _rank_metrics(similarity[window_index], labels, ref)
            rows.append({
                "sample_position": position,
                "dataset_index": dataset_index,
                "side": side,
                "window_index": window_index,
                "ink": float(ink[window_index]),
                "reference": ref,
                "predicted": metrics["predicted"],
                "rank": metrics["rank"],
                "top1": int(metrics["top1"]),
                "top3": int(metrics["top3"]),
                "top5": int(metrics["top5"]),
                "reciprocal_rank": metrics["rr"],
                "correct_cosine": metrics["correct_cosine"],
                "top1_cosine": metrics["top1_cosine"],
                "hardest_wrong_cosine": metrics["hardest_wrong_cosine"],
                "correct_margin": metrics["correct_margin"],
                "top5_labels": " ".join(metrics["top5_labels"]),
                "top5_scores": " ".join(f"{v:.6f}" for v in metrics["top5_scores"]),
                "image": str(image),
                "text": text,
            })
            vectors.append(visual_np[window_index])
            similarities_all.append(similarity[window_index])
            used += 1
        if used == 0:
            skipped_alignment += 1
        print(f"[{position}/{len(samples)}] index={dataset_index} side={side} windows={visual.shape[0]} labeled={used}", flush=True)

    if not rows:
        raise RuntimeError("No one-character alignment-conditioned windows survived filtering")

    vectors_np = np.asarray(vectors, dtype=np.float32)
    similarities_np = np.asarray(similarities_all, dtype=np.float32)
    prototypes_np = prototype_tensor.detach().cpu().numpy().astype(np.float32)
    np.save(output / "character_prototypes.npy", prototypes_np)
    np.save(output / "labeled_window_embeddings.npy", vectors_np)
    np.save(output / "window_character_cosine.npy", similarities_np)

    window_fields = [
        "sample_position", "dataset_index", "side", "window_index", "ink",
        "reference", "predicted", "rank", "top1", "top3", "top5",
        "reciprocal_rank", "correct_cosine", "top1_cosine",
        "hardest_wrong_cosine", "correct_margin", "top5_labels", "top5_scores",
        "image", "text",
    ]
    _write_csv(output / "per_window_retrieval.csv", rows, window_fields)

    token_rows = []
    for label in labels:
        subset = [row for row in rows if row["reference"] == label]
        if not subset:
            continue
        token_rows.append({
            "character": label,
            "windows": len(subset),
            "top1_accuracy": float(np.mean([row["top1"] for row in subset])),
            "top3_accuracy": float(np.mean([row["top3"] for row in subset])),
            "top5_accuracy": float(np.mean([row["top5"] for row in subset])),
            "mrr": float(np.mean([row["reciprocal_rank"] for row in subset])),
            "mean_correct_cosine": _finite_mean([row["correct_cosine"] for row in subset]),
            "mean_correct_margin": _finite_mean([row["correct_margin"] for row in subset]),
        })
    _write_csv(
        output / "per_character_metrics.csv",
        token_rows,
        ["character", "windows", "top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr", "mean_correct_cosine", "mean_correct_margin"],
    )

    _save_confusion(output, labels, rows)
    _save_mean_similarity(output, labels, rows, similarities_np)
    pca_info = _save_pca(output, labels, prototypes_np, rows, vectors_np, args.pca_max_windows, args.seed)

    summary = {
        "dataset": str(dataset),
        "weights": str(weights),
        "feature": args.feature,
        "samples_requested": int(args.n_samples),
        "samples_processed": len(samples),
        "samples_without_usable_character_windows": skipped_alignment,
        "reference_label_source": "hard_span_dtw_single_character_steps",
        "reference_is_independent_ground_truth": False,
        "character_count": len(labels),
        "characters": labels,
        "labeled_windows": len(rows),
        "min_ink": float(args.min_ink),
        "top1_accuracy": float(np.mean([row["top1"] for row in rows])),
        "top3_accuracy": float(np.mean([row["top3"] for row in rows])),
        "top5_accuracy": float(np.mean([row["top5"] for row in rows])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in rows])),
        "mean_correct_cosine": _finite_mean([row["correct_cosine"] for row in rows]),
        "mean_top1_cosine": _finite_mean([row["top1_cosine"] for row in rows]),
        "mean_correct_margin": _finite_mean([row["correct_margin"] for row in rows]),
        "predicted_character_counts": dict(Counter(row["predicted"] for row in rows)),
        "reference_character_counts": dict(Counter(row["reference"] for row in rows)),
        "pca": pca_info,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nToken/character embedding diagnostic")
    print(f"output              = {output}")
    print(f"feature             = {args.feature}")
    print(f"characters          = {len(labels)}")
    print(f"labeled windows     = {len(rows)}")
    print(f"top-1               = {summary['top1_accuracy']:.4f}")
    print(f"top-3               = {summary['top3_accuracy']:.4f}")
    print(f"top-5               = {summary['top5_accuracy']:.4f}")
    print(f"MRR                 = {summary['mrr']:.4f}")
    print(f"mean correct cosine = {summary['mean_correct_cosine']:.4f}")
    print(f"mean correct margin = {summary['mean_correct_margin']:.4f}")
    print("NOTE: reference labels are hard-Span-DTW-conditioned, not independent character boxes.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Needleman-Wunsch directly over line-image window embeddings.

The prediction is fully image-to-image: the NW matrix is the cosine similarity
between every window in line 1 and every window in line 2. Transcripts are used
only to annotate/evaluate matched windows with the token assigned by the trained
blank-aware Span-DTW path.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    PairPaths,
    align_text_to_windows,
    compute_similarity,
    get_image_features,
    iter_synthetic_pairs,
    json_ready,
    load_evaluation_models,
    needleman_wunsch,
    read_text,
    synthetic_pair_paths,
    validate_pair_paths,
)
from Evaluation.window_alignment import window_alignment_metrics, window_token_labels


def _window_pixels(index, n_windows, image_width, config, flipped):
    window = int(config.get("window_size", 32))
    stride = int(config.get("stride", max(1, window // 2)))
    model_width = int(config.get("input_width", 1024))
    logical = int(index)
    physical = n_windows - 1 - logical if flipped else logical
    start = max(0, physical * stride)
    end = min(model_width, start + window)
    scale = float(image_width) / max(1, model_width)
    return start * scale, end * scale


def _labels_for_line(models, text_path, features):
    text = read_text(text_path, boundary_spaces=False)
    _prepared, _encoding, path = align_text_to_windows(models, text, features, True)
    return text, path, window_token_labels(path, len(features.contextual))


def evaluate_pair(models, pair, feature, gap, similarity_offset, dataset_type):
    validate_pair_paths(pair)
    features1 = get_image_features(models, pair.image1, dataset_type)
    features2 = get_image_features(models, pair.image2, dataset_type)
    selected1 = features1.select(feature)
    selected2 = features2.select(feature)
    similarity = compute_similarity(selected1, selected2)
    alignment = needleman_wunsch(similarity, gap, similarity_offset)
    text1, span_path1, labels1 = _labels_for_line(models, pair.text1, features1)
    text2, span_path2, labels2 = _labels_for_line(models, pair.text2, features2)
    metrics = window_alignment_metrics(alignment, labels1, labels2)
    metrics.update(
        {
            "index": pair.index,
            "line1_windows": len(selected1),
            "line2_windows": len(selected2),
            "feature": feature,
            "span_steps_line1": len(span_path1),
            "span_steps_line2": len(span_path2),
        }
    )
    return {
        "pair": pair,
        "features1": features1,
        "features2": features2,
        "similarity": similarity,
        "alignment": alignment,
        "labels1": labels1,
        "labels2": labels2,
        "text1": text1,
        "text2": text2,
        "metrics": metrics,
    }


def _matched_steps(result, min_similarity, max_drawn_pairs):
    steps = [
        step
        for step in result.steps
        if step.index1 is not None
        and step.index2 is not None
        and step.similarity is not None
        and float(step.similarity) >= float(min_similarity)
    ]
    limit = int(max_drawn_pairs)
    if limit <= 0 or len(steps) <= limit:
        return steps
    positions = np.linspace(0, len(steps) - 1, limit).round().astype(int)
    return [steps[int(position)] for position in positions]


def save_visualization(result, output, min_similarity=-1.0, max_drawn_pairs=64, show_heatmap=True):
    pair = result["pair"]
    with Image.open(pair.image1) as opened:
        image1 = np.asarray(opened.convert("RGB"))
    with Image.open(pair.image2) as opened:
        image2 = np.asarray(opened.convert("RGB"))
    n1 = len(result["features1"].contextual)
    n2 = len(result["features2"].contextual)
    flipped = bool(result.get("flipped", True))
    steps = _matched_steps(result["alignment"], min_similarity, max_drawn_pairs)
    cmap = plt.get_cmap("turbo", max(2, len(steps)))

    rows = 4 if show_heatmap else 3
    ratios = [2.2, 0.7, 2.2, 3.5] if show_heatmap else [2.2, 0.7, 2.2]
    fig = plt.figure(figsize=(16, 10 if show_heatmap else 7))
    grid = fig.add_gridspec(rows, 1, height_ratios=ratios, hspace=0.12)
    ax1 = fig.add_subplot(grid[0])
    axc = fig.add_subplot(grid[1])
    ax2 = fig.add_subplot(grid[2])
    for ax, image, label in ((ax1, image1, "line 1"), (ax2, image2, "line 2")):
        ax.imshow(image, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=32, va="center")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")

    for order, step in enumerate(steps):
        color = cmap(order)
        i, j = int(step.index1), int(step.index2)
        x01, x11 = _window_pixels(i, n1, image1.shape[1], result["models_config"], flipped)
        x02, x12 = _window_pixels(j, n2, image2.shape[1], result["models_config"], flipped)
        for ax, image, x0, x1 in (
            (ax1, image1, x01, x11),
            (ax2, image2, x02, x12),
        ):
            ax.add_patch(
                Rectangle(
                    (x0, 1),
                    max(2.0, x1 - x0),
                    max(2.0, image.shape[0] - 2),
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.20,
                    linewidth=1.2,
                )
            )
        axc.add_line(
            Line2D(
                [((x01 + x11) / 2) / image1.shape[1], ((x02 + x12) / 2) / image2.shape[1]],
                [1.0, 0.0],
                transform=axc.transAxes,
                color=color,
                alpha=0.55,
                linewidth=1.0,
            )
        )

    metrics = result["metrics"]
    fig.suptitle(
        "Window-level Needleman–Wunsch alignment\n"
        f"pair={pair.index}  matched={metrics['matched_window_pairs']}  "
        f"mean cosine={metrics['mean_matched_cosine']:.3f}  "
        f"token agreement={metrics['token_agreement']:.3f}",
        fontsize=11,
        fontweight="bold",
    )

    if show_heatmap:
        axh = fig.add_subplot(grid[3])
        matrix = result["similarity"].detach().cpu().numpy()
        image = axh.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        matched = [
            step
            for step in result["alignment"].steps
            if step.index1 is not None and step.index2 is not None
        ]
        axh.plot(
            [int(step.index2) for step in matched],
            [int(step.index1) for step in matched],
            color="black",
            linewidth=1.1,
            marker=".",
            markersize=2,
        )
        axh.set_xlabel("line 2 windows")
        axh.set_ylabel("line 1 windows")
        fig.colorbar(image, ax=axh, fraction=0.025, pad=0.02, label="cosine similarity")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _aggregate(rows):
    if not rows:
        return {"samples": 0}
    numeric = [
        key
        for key in rows[0]
        if key != "index" and all(isinstance(row.get(key), (int, float)) for row in rows)
    ]
    summary = {"samples": len(rows)}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std())
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--gap", type=float, default=-0.25)
    parser.add_argument("--similarity-offset", type=float, default=0.0)
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument("--max-drawn-pairs", type=int, default=64)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output", default="Results/Evaluation/NW/window_nw.png")
    parser.add_argument("--output-dir", default="Results/Evaluation/NW/windows")
    parser.add_argument("--no-heatmap", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    if args.dataset_type != "synthetic":
        raise SystemExit("Batch pair discovery currently supports synthetic layout; use a manifest adapter for real data.")
    pairs = (
        list(iter_synthetic_pairs(args.data_dir, args.start_index, args.n_samples))
        if args.batch
        else [synthetic_pair_paths(args.data_dir, args.index)]
    )
    rows = []
    output_dir = Path(args.output_dir)
    for pair in pairs:
        result = evaluate_pair(
            models,
            pair,
            args.feature,
            args.gap,
            args.similarity_offset,
            args.dataset_type,
        )
        result["models_config"] = models.config
        result["flipped"] = bool(models.image_model.use_flip)
        rows.append(result["metrics"])
        output = output_dir / f"window_nw_{pair.index}.png" if args.batch else Path(args.output)
        save_visualization(
            result,
            output,
            min_similarity=args.min_similarity,
            max_drawn_pairs=args.max_drawn_pairs,
            show_heatmap=not args.no_heatmap,
        )
        print(json.dumps(json_ready(result["metrics"]), ensure_ascii=False), flush=True)

    if args.batch:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]) if rows else ["index"])
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "summary.json").write_text(
            json.dumps(json_ready(_aggregate(rows)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

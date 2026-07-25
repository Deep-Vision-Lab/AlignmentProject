#!/usr/bin/env python3
"""Cluster image windows and visualize each cluster with one real window crop.

Clustering uses only visual embeddings. The token shown for a cluster is the
majority token assigned to its member windows by the trained blank-aware
Span-DTW transcript alignment; text does not affect cluster membership.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    align_text_to_windows,
    get_image_features,
    json_ready,
    load_evaluation_models,
    read_text,
    synthetic_pair_paths,
    validate_pair_paths,
)
from Evaluation.window_alignment import (
    cluster_representatives,
    cosine_kmeans,
    pca_project_2d,
    summarize_cluster_tokens,
    window_token_labels,
)


def _exact_window_crop(image, logical_index, n_windows, config, flipped):
    window = int(config.get("window_size", 32))
    stride = int(config.get("stride", max(1, window // 2)))
    model_width = int(config.get("input_width", 1024))
    physical = n_windows - 1 - int(logical_index) if flipped else int(logical_index)
    x0_model = max(0, physical * stride)
    x1_model = min(model_width, x0_model + window)
    scale = float(image.shape[1]) / max(1, model_width)
    x0 = max(0, int(math.floor(x0_model * scale)))
    x1 = min(image.shape[1], int(math.ceil(x1_model * scale)))
    if x1 <= x0:
        x1 = min(image.shape[1], x0 + 1)
    return image[:, x0:x1]


def _line_labels(models, text_path, features):
    text = read_text(text_path, boundary_spaces=False)
    _prepared, _encoding, path = align_text_to_windows(models, text, features, True)
    return window_token_labels(path, len(features.contextual))


def collect_pair_windows(models, pair, feature, dataset_type, min_ink):
    validate_pair_paths(pair)
    features1 = get_image_features(models, pair.image1, dataset_type)
    features2 = get_image_features(models, pair.image2, dataset_type)
    labels1 = _line_labels(models, pair.text1, features1)
    labels2 = _line_labels(models, pair.text2, features2)
    with Image.open(pair.image1) as opened:
        image1 = np.asarray(opened.convert("RGB"))
    with Image.open(pair.image2) as opened:
        image2 = np.asarray(opened.convert("RGB"))

    records = []
    tensors = []
    for line, (features, labels, image, image_path) in enumerate(
        (
            (features1, labels1, image1, pair.image1),
            (features2, labels2, image2, pair.image2),
        ),
        start=1,
    ):
        selected = features.select(feature)
        for window_index in range(int(selected.shape[0])):
            ink = float(features.ink[window_index].item())
            if ink < float(min_ink):
                continue
            records.append(
                {
                    "line": line,
                    "window_index": window_index,
                    "token": labels[window_index],
                    "ink": ink,
                    "image_path": str(image_path),
                    "image": image,
                    "n_windows": int(selected.shape[0]),
                }
            )
            tensors.append(selected[window_index])
    if not tensors:
        raise RuntimeError("No windows remained after the --min-ink filter")
    return torch.stack(tensors), records


def save_cluster_figure(
    embeddings,
    records,
    labels,
    centers,
    config,
    flipped,
    output,
    representative_zoom,
):
    projected = pca_project_2d(embeddings)
    representatives = cluster_representatives(embeddings, labels, centers)
    summaries = {
        item.cluster: item
        for item in summarize_cluster_tokens(labels, [record["token"] for record in records])
    }
    clusters = sorted(representatives)
    columns = min(4, max(1, int(math.ceil(math.sqrt(len(clusters))))))
    rows = int(math.ceil(len(clusters) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 4.2 * rows),
        squeeze=False,
    )
    colors = plt.get_cmap("tab20", max(2, len(clusters)))

    for panel, cluster in enumerate(clusters):
        ax = axes.flat[panel]
        members = np.flatnonzero(np.asarray(labels) == cluster)
        representative = representatives[cluster]
        other_members = members[members != representative]
        ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.08, color="gray")
        if len(other_members):
            ax.scatter(
                projected[other_members, 0],
                projected[other_members, 1],
                s=28,
                alpha=0.8,
                color=colors(cluster),
                label=f"other windows ({len(other_members)})",
            )
        record = records[representative]
        crop = _exact_window_crop(
            record["image"],
            record["window_index"],
            record["n_windows"],
            config,
            flipped,
        )
        image_box = OffsetImage(crop, zoom=float(representative_zoom))
        annotation = AnnotationBbox(
            image_box,
            (projected[representative, 0], projected[representative, 1]),
            frameon=True,
            bboxprops=dict(
                edgecolor=colors(cluster),
                linewidth=2,
                facecolor="white",
            ),
        )
        ax.add_artist(annotation)
        summary = summaries[cluster]
        ax.set_title(
            f"Cluster {cluster} — token: {summary.token}\n"
            f"n={summary.count}, purity={summary.purity:.2f}, "
            f"representative=L{record['line']}:W{record['window_index']}",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.15)

    for panel in range(len(clusters), rows * columns):
        axes.flat[panel].axis("off")
    fig.suptitle(
        "Visual window clusters: one representative crop, remaining members as dots\n"
        "Cluster tokens are majority labels from Span-DTW; clustering itself is image-only",
        fontsize=12,
        fontweight="bold",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return projected, representatives, summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="local")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ink", type=float, default=0.01)
    parser.add_argument("--representative-zoom", type=float, default=2.4)
    parser.add_argument("--output", default="Results/Evaluation/Clusters/window_clusters.png")
    parser.add_argument("--output-dir", default="Results/Evaluation/Clusters")
    return parser.parse_args()


def main():
    args = parse_args()
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    if args.dataset_type != "synthetic":
        raise SystemExit("This entrypoint currently discovers pairs from the synthetic layout.")
    pair = synthetic_pair_paths(args.data_dir, args.index)
    embeddings, records = collect_pair_windows(
        models,
        pair,
        args.feature,
        args.dataset_type,
        args.min_ink,
    )
    labels, centers = cosine_kmeans(
        embeddings,
        args.clusters,
        args.max_iter,
        args.seed,
    )
    projected, representatives, summaries = save_cluster_figure(
        embeddings,
        records,
        labels,
        centers,
        models.config,
        bool(models.image_model.use_flip),
        args.output,
        args.representative_zoom,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        cluster = int(labels[index])
        rows.append(
            {
                "cluster": cluster,
                "cluster_token": summaries[cluster].token,
                "cluster_purity": summaries[cluster].purity,
                "is_representative": int(representatives.get(cluster) == index),
                "line": record["line"],
                "window_index": record["window_index"],
                "window_token": record["token"],
                "ink": record["ink"],
                "pca_x": float(projected[index, 0]),
                "pca_y": float(projected[index, 1]),
            }
        )
    with (output_dir / f"window_clusters_{pair.index}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_payload = {
        "index": pair.index,
        "feature": args.feature,
        "windows": len(records),
        "clusters": [json_ready(vars(summaries[key])) for key in sorted(summaries)],
    }
    (output_dir / f"window_clusters_{pair.index}.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

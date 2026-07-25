#!/usr/bin/env python3
"""Visualize the trained blank-aware Span-DTW text-to-image alignment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    align_text_to_windows,
    get_image_features,
    load_evaluation_models,
    read_text,
    synthetic_pair_paths,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--line", choices=("1", "2"), default="1")
    parser.add_argument("--image")
    parser.add_argument("--text-file")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    parser.add_argument(
        "--output",
        default="Results/Evaluation/span_dtw_heatmap.png",
    )
    args = parser.parse_args()

    pair = synthetic_pair_paths(args.data_dir, args.index)
    image_path = Path(args.image) if args.image else (
        pair.image1 if args.line == "1" else pair.image2
    )
    text_path = Path(args.text_file) if args.text_file else (
        pair.text1 if args.line == "1" else pair.text2
    )
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    text = read_text(text_path, boundary_spaces=False)
    features = get_image_features(models, image_path, args.dataset_type)
    prepared, encoding, path = align_text_to_windows(models, text, features, True)
    alignment_embeddings = getattr(encoding, "context_embeddings", None)
    if alignment_embeddings is None:
        alignment_embeddings = encoding.embeddings
    text_embeddings = F.normalize(alignment_embeddings.float(), p=2, dim=-1)
    similarity = (text_embeddings @ features.contextual.T).detach().cpu().numpy()

    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB"))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        gridspec_kw={"height_ratios": [1, 4]},
        constrained_layout=True,
    )
    axes[0].imshow(image, aspect="auto")
    axes[0].axis("off")
    heatmap = axes[1].imshow(
        similarity,
        aspect="auto",
        cmap="magma",
        origin="lower",
        vmin=-1,
        vmax=1,
    )
    for step in path:
        if step.get("is_blank", False):
            continue
        span_idx = int(step["span_idx"])
        center = 0.5 * (
            int(step["window_start"]) + int(step["window_end"]) - 1
        )
        axes[1].scatter([center], [span_idx], c="cyan", s=12)
    axes[1].set_xlabel("image windows in model order")
    axes[1].set_ylabel("candidate spans")
    fig.colorbar(heatmap, ax=axes[1], label="cosine similarity")
    fig.suptitle(f"Blank-aware Span-DTW alignment | {prepared!r}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved={output} path_steps={len(path)}")


if __name__ == "__main__":
    main()

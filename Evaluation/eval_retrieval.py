#!/usr/bin/env python3
"""Image-to-image retrieval metrics for optimized checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    get_image_features,
    load_evaluation_models,
    synthetic_pair_paths,
)


def line_embedding(features, feature):
    sequence = features.select(feature)
    ink = features.ink.clamp_min(0.0)
    if float(ink.sum()) > 1e-8:
        pooled = (sequence * (ink / ink.sum()).unsqueeze(-1)).sum(0)
    else:
        pooled = sequence.mean(0)
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--feature",
        choices=("contextual", "local", "grouped"),
        default="contextual",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    args = parser.parse_args()

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    left, right, indices = [], [], []
    for index in range(args.start_index, args.start_index + args.n_samples):
        pair = synthetic_pair_paths(args.data_dir, index)
        if not pair.image1.is_file() or not pair.image2.is_file():
            continue
        left.append(
            line_embedding(
                get_image_features(models, pair.image1, args.dataset_type),
                args.feature,
            )
        )
        right.append(
            line_embedding(
                get_image_features(models, pair.image2, args.dataset_type),
                args.feature,
            )
        )
        indices.append(index)
    if not left:
        raise SystemExit("No valid image pairs found")

    left_tensor = torch.stack(left)
    right_tensor = torch.stack(right)
    similarity = left_tensor @ right_tensor.T
    order = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(len(indices), device=order.device)
    ranks = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    off_diagonal_count = max(1, similarity.numel() - len(indices))
    metrics = {
        "samples": len(indices),
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= 5).float().mean()),
        "recall_at_10": float((ranks <= 10).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
        "mean_positive_cosine": float(similarity.diag().mean()),
        "mean_negative_cosine": float(
            (similarity.sum() - similarity.diag().sum()) / off_diagonal_count
        ),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

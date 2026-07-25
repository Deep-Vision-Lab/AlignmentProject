#!/usr/bin/env python3
"""Local alignment MAE using the trained blank-aware Span-DTW path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    align_text_to_windows,
    get_image_features,
    iter_synthetic_pairs,
    load_evaluation_models,
    read_text,
)


def sample_mae(text: str, path: list[dict], image_steps: int) -> float:
    prepared = f" {' '.join(text.split())} "
    mapped = {index: [] for index in range(len(prepared))}
    for step in path:
        if step.get("is_blank", False):
            continue
        center = 0.5 * (int(step["window_start"]) + int(step["window_end"]) - 1)
        for index in range(int(step["text_start"]), int(step["text_end"])):
            mapped.setdefault(index, []).append(center)
    errors = []
    for index, character in enumerate(prepared):
        if character.isspace():
            continue
        expected = index * (image_steps - 1) / max(1, len(prepared) - 1)
        values = mapped.get(index, [])
        predicted = float(np.median(values)) if values else expected
        errors.append(abs(predicted - expected))
    return float(np.mean(errors)) if errors else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    args = parser.parse_args()

    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    values = []
    normalized = []
    for pair in iter_synthetic_pairs(args.data_dir, args.start_index, args.n_samples):
        text = read_text(pair.text1, boundary_spaces=False)
        features = get_image_features(models, pair.image1, args.dataset_type)
        _prepared, _encoding, path = align_text_to_windows(models, text, features, True)
        mae = sample_mae(text, path, len(features.contextual))
        values.append(mae)
        normalized.append(mae / max(1, len(features.contextual)) * 100.0)
    if not values:
        raise SystemExit("No valid image/text pairs found")
    metrics = {
        "samples": len(values),
        "mean_mae_windows": float(np.mean(values)),
        "median_mae_windows": float(np.median(values)),
        "std_mae_windows": float(np.std(values)),
        "mean_mae_percent_width": float(np.mean(normalized)),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

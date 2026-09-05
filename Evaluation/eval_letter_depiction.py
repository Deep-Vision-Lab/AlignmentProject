#!/usr/bin/env python3
"""Inspect what Arabic letters each local visual window depicts.

Example:
    python Evaluation/eval_letter_depiction.py \
      --weights Weights/vit_vlm_letters/model_latest.pth \
      --image DataSet/Synthetic63/images/img1_1.png \
      --top-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Evaluation.vit_evaluation import install_vit_evaluation_loader
from Evaluation import _eval_utils
from vlm_letter_grounding import DEFAULT_ARABIC_LETTERS, _project_letter_inventory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-ink", type=float, default=0.01)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    install_vit_evaluation_loader()
    models = _eval_utils.load_evaluation_models(
        args.weights,
        device="auto",
        load_text_model=True,
    )
    if models.text_model is None:
        raise RuntimeError("letter depiction diagnostic requires the text encoder")
    if not bool(models.config.get("letter_depiction_enabled", False)):
        raise RuntimeError(
            "checkpoint does not report letter_depiction_enabled=True; "
            "use a checkpoint from agent/vlm-letter-depiction-hierarchy"
        )

    features = _eval_utils.get_image_features(
        models,
        args.image,
        dataset_type=args.dataset_type,
    )
    inventory = str(models.config.get("letter_depiction_inventory", DEFAULT_ARABIC_LETTERS))
    letters = list(inventory)

    with torch.no_grad():
        prototypes = _project_letter_inventory(models.text_model, inventory)
        local = torch.nn.functional.normalize(features.local.float(), p=2, dim=-1)
        similarities = local @ prototypes.T

    k = min(max(1, int(args.top_k)), len(letters))
    rows = []
    for window_index in range(int(similarities.shape[0])):
        ink = float(features.ink[window_index].item())
        values, indices = torch.topk(similarities[window_index], k=k)
        predictions = [
            {
                "letter": letters[int(index)],
                "cosine": float(value),
            }
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        ]
        row = {
            "window": window_index,
            "ink": ink,
            "active": bool(ink >= float(args.min_ink)),
            "top_letters": predictions,
        }
        rows.append(row)
        if row["active"]:
            pretty = " ".join(
                f"{item['letter']}:{item['cosine']:.3f}" for item in predictions
            )
            print(f"window={window_index:03d} ink={ink:.3f}  {pretty}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "weights": str(args.weights),
                    "image": str(args.image),
                    "inventory": inventory,
                    "windows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {output}")


if __name__ == "__main__":
    main()

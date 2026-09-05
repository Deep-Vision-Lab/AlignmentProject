#!/usr/bin/env python3
"""Needleman-Wunsch image alignment using a local/contextual hybrid embedding.

For every image window i:

    h_i = normalize(alpha * local_i + (1 - alpha) * contextual_i)

The canonical NW diagnostic is then reused unchanged: cosine similarity between
hybrid windows -> score normalization -> ink-aware match scores -> GLOBAL NW.

This wrapper intentionally keeps eval_img_align_nw_diagnostic.py unchanged so
pure contextual/local/grouped runs remain directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch.nn.functional as F

from Evaluation import eval_img_align_nw_diagnostic as base
from Evaluation._eval_utils import ImageFeatures

_ALPHA = 0.5
_OUTPUT_DIR: Path | None = None
_ORIGINAL_SELECT = ImageFeatures.select
_ORIGINAL_PARSE_ARGS = base.parse_args
_ORIGINAL_EVALUATE = base.evaluate


def _hybrid_select(self: ImageFeatures, name: str):
    if str(name).lower() != "hybrid":
        return _ORIGINAL_SELECT(self, name)
    if self.local.shape != self.contextual.shape:
        raise ValueError(
            "Hybrid NW requires local and contextual features with identical "
            f"shape, got local={tuple(self.local.shape)} and "
            f"contextual={tuple(self.contextual.shape)}"
        )
    mixed = _ALPHA * self.local.float() + (1.0 - _ALPHA) * self.contextual.float()
    return F.normalize(mixed, p=2, dim=-1)


def _remove_feature_override(argv: list[str]) -> list[str]:
    """The hybrid wrapper owns feature selection; ignore canonical --feature."""
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--feature":
            index += 2
            continue
        if value.startswith("--feature="):
            index += 1
            continue
        cleaned.append(value)
        index += 1
    return cleaned


def _hybrid_parse_args():
    global _ALPHA, _OUTPUT_DIR

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--alpha", type=float, default=0.5)
    known, remaining = pre.parse_known_args()
    if not 0.0 <= float(known.alpha) <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1 inclusive")
    _ALPHA = float(known.alpha)

    remaining = _remove_feature_override(remaining)
    original_argv = sys.argv
    try:
        # Feed only canonical arguments to the canonical parser.
        sys.argv = [original_argv[0], *remaining]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.feature = "hybrid"
    args.alpha = _ALPHA

    if args.output_dir:
        _OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    else:
        dataset = Path(args.dataset).expanduser().resolve()
        weights = Path(args.weights).expanduser().resolve()
        alpha_name = f"alpha_{_ALPHA:.2f}".replace(".", "p")
        _OUTPUT_DIR = (
            base.ROOT
            / "Results"
            / "Evaluation"
            / "NW_Hybrid"
            / (dataset.stem if dataset.is_file() else dataset.name)
            / weights.parent.name
            / alpha_name
        )
        args.output_dir = str(_OUTPUT_DIR)
    return args


def _hybrid_evaluate(models, pair, args, output_dir):
    row = _ORIGINAL_EVALUATE(models, pair, args, output_dir)
    row["feature"] = "hybrid"
    row["hybrid_alpha"] = float(_ALPHA)
    row["hybrid_formula"] = "normalize(alpha*local + (1-alpha)*contextual)"
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return row


def main():
    ImageFeatures.select = _hybrid_select
    base.parse_args = _hybrid_parse_args
    base.evaluate = _hybrid_evaluate
    try:
        print(
            "Hybrid NW feature: h = normalize(alpha*local + "
            "(1-alpha)*contextual)",
            flush=True,
        )
        base.main()
        if _OUTPUT_DIR is not None:
            aggregate = _OUTPUT_DIR / "summary.json"
            if aggregate.is_file():
                payload = json.loads(aggregate.read_text(encoding="utf-8"))
                payload["feature"] = "hybrid"
                payload["hybrid_alpha"] = float(_ALPHA)
                payload["hybrid_formula"] = (
                    "normalize(alpha*local + (1-alpha)*contextual)"
                )
                aggregate.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(f"Hybrid alpha={_ALPHA:.3f} output={_OUTPUT_DIR}", flush=True)
    finally:
        ImageFeatures.select = _ORIGINAL_SELECT
        base.parse_args = _ORIGINAL_PARSE_ARGS
        base.evaluate = _ORIGINAL_EVALUATE


if __name__ == "__main__":
    main()

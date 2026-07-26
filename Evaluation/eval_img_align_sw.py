#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment."""
from __future__ import annotations

from pathlib import Path
import sys

# When this file is executed directly, Python adds Evaluation/ rather than the
# repository root to sys.path. Add the project root before importing the
# Evaluation package so both of these forms work:
#   python Evaluation/eval_img_align_sw.py ...
#   python -m Evaluation.eval_img_align_sw ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Install the shared aspect-preserving real-image preprocessing and balanced
# ArabicDataset split/sampling before sw_runner imports functions from
# Evaluation.sw_dataset.
from Evaluation.zero_shot_sw import install_dataset_patches

install_dataset_patches()

from Evaluation import sw_dataset as _sw_dataset
from Evaluation.balanced_sampling import balanced_group_split_pairs as _balanced_split


def _diverse_group_split(pairs, seed):
    return _balanced_split(pairs, seed, _sw_dataset.random_split_pairs)


_sw_dataset.group_split_pairs = _diverse_group_split
_sw_dataset._group_split_pairs = _diverse_group_split

from Evaluation import sw_runner as _implementation
from Evaluation.zero_shot_sw import install_runner_patches

install_runner_patches(_implementation)

# Re-export public and private helpers for backward-compatible imports/tests.
globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)


if __name__ == "__main__":
    _implementation.main()

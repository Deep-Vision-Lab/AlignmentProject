#!/usr/bin/env python3
"""Checkpoint-compatible Needleman-Wunsch global image alignment."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_line_geometry import install_evaluation_geometry

_EVALUATION_GEOMETRY = install_evaluation_geometry()

from Evaluation.vit_evaluation import install_vit_evaluation_loader

install_vit_evaluation_loader()

from Evaluation.zero_shot_sw import install_dataset_patches

install_dataset_patches()

from Evaluation import sw_dataset as _sw_dataset


def _diverse_group_split(pairs, seed):
    """Keep pair_id groups intact while preserving validation/test diversity."""
    groups = OrderedDict()
    for position, pair in enumerate(pairs):
        groups.setdefault(pair.pair_id or f"sample_{position}", []).append(pair)
    if len(groups) < 3:
        return _sw_dataset.random_split_pairs(pairs, seed)

    rng = random.Random(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]))
    assigned = {"train": [], "valid": [], "test": []}
    minimum_eval_groups = 2 if len(items) >= 6 else 1
    for _ in range(minimum_eval_groups):
        _group_id, members = items.pop(0)
        assigned["test"].extend(members)
    for _ in range(minimum_eval_groups):
        _group_id, members = items.pop(0)
        assigned["valid"].extend(members)
    if items:
        _group_id, members = items.pop()
        assigned["train"].extend(members)

    targets = {
        "train": 0.60 * len(pairs),
        "valid": 0.20 * len(pairs),
        "test": 0.20 * len(pairs),
    }
    for _group_id, members in sorted(
        items, key=lambda item: len(item[1]), reverse=True
    ):
        destination = max(
            ("train", "valid", "test"),
            key=lambda split: targets[split] - len(assigned[split]),
        )
        assigned[destination].extend(members)
    return assigned["train"], assigned["valid"], assigned["test"]


_sw_dataset.group_split_pairs = _diverse_group_split
_sw_dataset._group_split_pairs = _diverse_group_split

from Evaluation import nw_runner as _implementation
from Evaluation.nw_component_regions import install as install_component_regions

# Use the same component-aware Needleman-Wunsch interpretation for every
# dataset. Ground-truth masks/bboxes are not consulted by the alignment path.
install_component_regions(_implementation)

globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)


if __name__ == "__main__":
    _implementation.main()

#!/usr/bin/env python3
"""Checkpoint-compatible Needleman-Wunsch global image alignment."""
from __future__ import annotations

from collections import OrderedDict
import os
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
from Evaluation.nw_discontinuous_regions import install as install_discontinuous_regions
from Evaluation.nw_physical_mapping import install as install_physical_mapping


def _requested_dataset_type() -> str:
    """Read --dataset-type early so region extraction matches the dataset."""
    for index, argument in enumerate(sys.argv):
        if argument == "--dataset-type" and index + 1 < len(sys.argv):
            return str(sys.argv[index + 1]).strip().lower()
        if argument.startswith("--dataset-type="):
            return argument.split("=", 1)[1].strip().lower()
    return "synthetic"


# The canonical synthetic NW launcher displays the accumulated DP-score heatmap.
# The physical-mapping patch historically replaced that with match-score only for
# real inputs, so real and synthetic figures were visually comparing different
# matrices even when both launchers requested dp-score. Keep the requested DP
# view by default so the real figure has the same heatmap semantics as synthetic.
# Users can still request match-score or cosine explicitly with --heatmap-source.
os.environ.setdefault("NW_REAL_KEEP_DP_HEATMAP", "1")

dataset_type = _requested_dataset_type()

# Synthetic evaluation deliberately reconnects small holes between neighboring
# supported pieces (three traceback steps in the canonical component launcher).
# Use the same tolerance for real manuscripts: a one/two/three-step wobble in an
# otherwise continuous NW path is not evidence for a genuinely different phrase.
# Longer unsupported valleys still split the predicted aligned regions.
if dataset_type == "real":
    os.environ.setdefault("NW_REGION_MAX_BRIDGE_STEPS", "3")
    # A red region is a visualization of selected heatmap/window cells, not the
    # union of their heavily-overlapping 32-pixel receptive fields. Map each
    # selected cell through neighboring window-center boundaries instead.
    os.environ.setdefault("NW_VIS_REGION_MAPPING", "cell")

# Synthetic evaluation uses the strict component-aware selector because its
# generator explicitly contains 1/2/3 shared regions. Real manuscript pairs do
# not obey those synthetic component-size/percentile/quality priors. Applying
# them to real data can erase the entire predicted region even when the global
# NW traceback contains many valid diagonal correspondences. For real inputs,
# derive masks directly from sustained positive runs on that fixed NW traceback;
# the physical-mapping layer then converts those logical Arabic window ranges to
# line-image pixels.
if dataset_type == "real":
    install_discontinuous_regions(_implementation)
else:
    install_component_regions(_implementation)
install_physical_mapping(_implementation)

globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)


if __name__ == "__main__":
    _implementation.main()

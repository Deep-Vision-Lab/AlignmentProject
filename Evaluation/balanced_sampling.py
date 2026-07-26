"""Pair-id-safe split helpers with explicit validation/test diversity."""
from __future__ import annotations

from collections import OrderedDict
import random


def balanced_group_split_pairs(pairs, seed: int, fallback):
    groups = OrderedDict()
    for position, pair in enumerate(pairs):
        groups.setdefault(pair.pair_id or f"sample_{position}", []).append(pair)
    if len(groups) < 3:
        return fallback(pairs, seed)

    rng = random.Random(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]))

    assigned = {"train": [], "valid": [], "test": []}
    minimum_eval_groups = 2 if len(items) >= 6 else 1

    # Small complete groups are used to seed validation/test diversity.  The
    # remaining, usually larger groups are allocated by sample-count deficit.
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
    for _group_id, members in sorted(items, key=lambda item: len(item[1]), reverse=True):
        destination = max(
            ("train", "valid", "test"),
            key=lambda split: targets[split] - len(assigned[split]),
        )
        assigned[destination].extend(members)

    return assigned["train"], assigned["valid"], assigned["test"]

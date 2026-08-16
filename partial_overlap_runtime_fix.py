"""Runtime guard for feasible partial-overlap positive anchors.

Some canonical high/medium real lines are already longer than the configured
partial-overlap composite text cap. Those rows are valid canonical training
samples, but they can never satisfy the synthetic-composite cap even after the
partial-overlap generator falls back to one shared island with no distractor.

This wrapper keeps those long rows in ordinary canonical training while removing
them only from the online partial-overlap anchor/partner pool. That makes the
one-island fallback guaranteed to be feasible without truncating transcripts or
inventing character-to-pixel boundaries.
"""
from __future__ import annotations

from PartialOverlapRealAugmentation import (
    PartialOverlapRealPairDataset as _BasePartialOverlapRealPairDataset,
    _read_clean_text,
)


class FeasiblePartialOverlapRealPairDataset(_BasePartialOverlapRealPairDataset):
    """Partial-overlap dataset with a guaranteed feasible shared-anchor pool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        feasible: list[int] = []
        rejected = 0
        max_observed = 0
        for index in self.positive_indices:
            sample = self.positive_dataset.samples[int(index)]
            text_a = _read_clean_text(self.positive_dataset, sample["A"])
            text_b = _read_clean_text(self.positive_dataset, sample["B"])
            length = max(len(text_a), len(text_b))
            max_observed = max(max_observed, length)
            if text_a and text_b and length <= self.max_text_chars:
                feasible.append(int(index))
            else:
                rejected += 1

        if not feasible:
            raise RuntimeError(
                "No positive training rows fit REAL_PARTIAL_OVERLAP_MAX_TEXT_CHARS="
                f"{self.max_text_chars}; cannot synthesize partial-overlap pairs."
            )

        # _choose_shared() and __getitem__() both consult self.positive_indices,
        # so replacing this pool makes every chosen shared island individually
        # feasible. Multi-island attempts may still exceed the cap, but the
        # existing 3 -> 2 -> 1 retry path then has a guaranteed valid endpoint.
        self.positive_indices = feasible
        self.partial_overlap_rejected_long_positive_anchors = rejected
        self.partial_overlap_max_observed_positive_chars = max_observed

        print(
            "Partial-overlap feasible anchor pool: "
            f"kept={len(feasible)} rejected_long_or_empty={rejected} "
            f"cap={self.max_text_chars} max_observed={max_observed}",
            flush=True,
        )


def install(base) -> None:
    """Install the normal partial-overlap objective with the feasible-pool fix."""
    import joint_real_training_partial_overlap as partial

    partial.PartialOverlapRealPairDataset = FeasiblePartialOverlapRealPairDataset
    partial.install(base)

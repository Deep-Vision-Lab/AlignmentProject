"""Phrase-level grouping for window-level Needleman-Wunsch alignments.

The alignment itself remains window-level. This module only merges nearby,
monotonic NW matches into larger regions for visualization and reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PhraseAlignmentGroup:
    """One visually continuous aligned region in both line images."""

    anchors: tuple
    line1_start: int
    line1_end: int
    line2_start: int
    line2_end: int
    path_start: int
    path_end: int
    mean_similarity: float

    @property
    def anchor_pairs(self) -> int:
        return len(self.anchors)

    @property
    def line1_span_windows(self) -> int:
        return self.line1_end - self.line1_start

    @property
    def line2_span_windows(self) -> int:
        return self.line2_end - self.line2_start


def phrase_alignment_groups(
    result,
    merge_gap_tolerance_windows: int = 2,
    merge_max_jump_windows: int = 4,
    merge_min_similarity: float = 0.30,
) -> list[PhraseAlignmentGroup]:
    """Merge nearby monotonic NW matches into phrase/sentence-like regions.

    Matches whose raw cosine is at least ``merge_min_similarity`` act as strong
    anchors. Weak matches and gap transitions may bridge two anchors when:

    * the number of intervening traceback steps does not exceed
      ``merge_gap_tolerance_windows``;
    * the next anchor advances by no more than ``merge_max_jump_windows`` in
      either line; and
    * the anchor order remains strictly monotonic.

    Each returned span covers every window between its first and last anchor, so
    tolerated weak windows and gaps are included in the continuous mask. The NW
    path and scores are never changed by this grouping.
    """
    tolerance = max(0, int(merge_gap_tolerance_windows))
    max_jump = max(1, int(merge_max_jump_windows))
    threshold = float(merge_min_similarity)

    groups: list[PhraseAlignmentGroup] = []
    anchors: list = []
    path_start = -1
    path_end = -1
    bridge_steps = 0

    def flush() -> None:
        nonlocal anchors, path_start, path_end, bridge_steps
        if anchors:
            similarities = [float(step.similarity) for step in anchors]
            groups.append(
                PhraseAlignmentGroup(
                    anchors=tuple(anchors),
                    line1_start=int(anchors[0].index1),
                    line1_end=int(anchors[-1].index1) + 1,
                    line2_start=int(anchors[0].index2),
                    line2_end=int(anchors[-1].index2) + 1,
                    path_start=path_start,
                    path_end=path_end,
                    mean_similarity=float(np.mean(similarities)),
                )
            )
        anchors = []
        path_start = -1
        path_end = -1
        bridge_steps = 0

    for path_index, step in enumerate(result.steps):
        strong_match = (
            step.index1 is not None
            and step.index2 is not None
            and step.similarity is not None
            and float(step.similarity) >= threshold
        )

        if not strong_match:
            if anchors:
                bridge_steps += 1
            continue

        if not anchors:
            anchors = [step]
            path_start = path_index
            path_end = path_index + 1
            bridge_steps = 0
            continue

        previous = anchors[-1]
        delta1 = int(step.index1) - int(previous.index1)
        delta2 = int(step.index2) - int(previous.index2)
        monotonic = delta1 > 0 and delta2 > 0
        local_jump = delta1 <= max_jump and delta2 <= max_jump
        short_bridge = bridge_steps <= tolerance

        if monotonic and local_jump and short_bridge:
            anchors.append(step)
            path_end = path_index + 1
            bridge_steps = 0
        else:
            flush()
            anchors = [step]
            path_start = path_index
            path_end = path_index + 1

    flush()
    return groups


def phrase_group_metrics(groups: Sequence[PhraseAlignmentGroup]) -> dict:
    """Return compact statistics for phrase-level visualization groups."""
    anchor_counts = [group.anchor_pairs for group in groups]
    line1_spans = [group.line1_span_windows for group in groups]
    line2_spans = [group.line2_span_windows for group in groups]
    similarities = [group.mean_similarity for group in groups]
    return {
        "phrase_groups": len(groups),
        "longest_phrase_group_pairs": max(anchor_counts, default=0),
        "mean_phrase_group_pairs": float(np.mean(anchor_counts)) if anchor_counts else 0.0,
        "longest_phrase_span_line1": max(line1_spans, default=0),
        "longest_phrase_span_line2": max(line2_spans, default=0),
        "mean_phrase_group_cosine": float(np.mean(similarities)) if similarities else 0.0,
    }

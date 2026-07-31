"""Deterministic feasibility checks for positive real Span-DTW samples."""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Subset


@dataclass(frozen=True)
class FeasibilityStats:
    split_name: str
    kept: int
    removed: int
    max_required_spans: int
    examples: tuple[str, ...]


def minimum_required_spans(text: str, max_span_chars: int = 2) -> int:
    """Return the exact minimum number of non-blank text transitions.

    The optimized Arabic span encoder permits spans of up to ``max_span_chars``
    consecutive non-space characters. Every whitespace character is a standalone
    transcript position. Leading/trailing whitespace is stripped by the encoder.
    """
    max_span_chars = int(max_span_chars)
    if max_span_chars <= 0:
        raise ValueError("max_span_chars must be positive")

    prepared = str(text).strip()
    required = 0
    run_length = 0

    for character in prepared:
        if character.isspace():
            if run_length:
                required += (run_length + max_span_chars - 1) // max_span_chars
                run_length = 0
            required += 1
        else:
            run_length += 1

    if run_length:
        required += (run_length + max_span_chars - 1) // max_span_chars
    return required


def filter_subset_by_span_feasibility(
    base_dataset,
    subset,
    *,
    split_name: str,
    max_image_windows: int,
    max_span_chars: int,
):
    """Remove positive pairs whose A or B text cannot fit the image lattice.

    Filtering an already-created subset preserves the deterministic pair-ID split
    assignment. Only the infeasible samples are removed; groups are never moved
    between train, validation, and test.
    """
    max_image_windows = int(max_image_windows)
    max_span_chars = int(max_span_chars)
    if max_image_windows <= 0:
        raise ValueError("max_image_windows must be positive")
    if max_span_chars <= 0:
        raise ValueError("max_span_chars must be positive")

    kept_indices = []
    removed_examples = []
    max_required = 0

    for raw_index in subset.indices:
        sample_index = int(raw_index)
        sample = base_dataset.samples[sample_index]
        required = {}

        for side_name in ("A", "B"):
            side = sample[side_name]
            text = base_dataset._read_text(side[base_dataset.text_key])
            required[side_name] = minimum_required_spans(
                text,
                max_span_chars=max_span_chars,
            )

        sample_max = max(required.values())
        max_required = max(max_required, sample_max)
        if sample_max <= max_image_windows:
            kept_indices.append(sample_index)
            continue

        if len(removed_examples) < 5:
            removed_examples.append(
                f"pair_id={sample.get('pair_id', sample_index)} "
                f"required_A={required['A']} required_B={required['B']}"
            )

    if not kept_indices:
        raise ValueError(
            f"Span-DTW feasibility filtering removed every {split_name} sample: "
            f"max_image_windows={max_image_windows}, max_span_chars={max_span_chars}."
        )

    stats = FeasibilityStats(
        split_name=str(split_name),
        kept=len(kept_indices),
        removed=len(subset.indices) - len(kept_indices),
        max_required_spans=max_required,
        examples=tuple(removed_examples),
    )
    return Subset(base_dataset, kept_indices), stats

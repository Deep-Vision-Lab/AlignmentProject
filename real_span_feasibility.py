"""Deterministic feasibility checks for positive real Span-DTW samples."""
from __future__ import annotations

from dataclasses import dataclass
import os

from torch.utils.data import Subset


_CONNECTED_MODES = {
    "connected_subword",
    "connected-subword",
    "joining_run",
    "joining-run",
}


@dataclass(frozen=True)
class FeasibilityStats:
    split_name: str
    kept: int
    removed: int
    max_required_spans: int
    examples: tuple[str, ...]


def _tokenization_mode(tokenization_mode: str | None = None) -> str:
    if tokenization_mode is not None:
        return str(tokenization_mode).strip().lower()
    return os.environ.get(
        "SPAN_TOKENIZATION_MODE", "character_span"
    ).strip().lower()


def minimum_required_spans(
    text: str,
    max_span_chars: int = 2,
    *,
    tokenization_mode: str | None = None,
) -> int:
    """Return the exact minimum number of non-blank text transitions.

    Character-span mode can cover up to ``max_span_chars`` consecutive
    non-space characters with one transition. Connected-subword mode instead
    emits one transition for every connected Arabic run, every explicit
    ``<SUBWORD_BOUNDARY>``, and every ``<SPACE>`` state. Those structural states
    also consume at least one image window in the current Span-DTW lattice, so
    the exact feasibility count is ``len(connected_units(text))``.
    """
    mode = _tokenization_mode(tokenization_mode)
    if mode in _CONNECTED_MODES:
        from connected_subword_mode import connected_units

        return len(connected_units(str(text)))

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
    assignment. Only infeasible samples are removed; groups are never moved
    between train, validation, and test. The count follows the active tokenizer,
    so connected-subword samples are checked using their actual connected-unit
    sequence rather than the older character-span approximation.
    """
    max_image_windows = int(max_image_windows)
    max_span_chars = int(max_span_chars)
    if max_image_windows <= 0:
        raise ValueError("max_image_windows must be positive")
    if max_span_chars <= 0:
        raise ValueError("max_span_chars must be positive")

    mode = _tokenization_mode()
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
                tokenization_mode=mode,
            )

        sample_max = max(required.values())
        max_required = max(max_required, sample_max)
        if sample_max <= max_image_windows:
            kept_indices.append(sample_index)
            continue

        if len(removed_examples) < 10:
            side_a = sample.get("A") or {}
            side_b = sample.get("B") or {}
            removed_examples.append(
                f"pair_id={sample.get('pair_id', sample_index)} "
                f"required_A={required['A']} required_B={required['B']} "
                f"line_A={side_a.get('line_image_path', '<missing>')} "
                f"line_B={side_b.get('line_image_path', '<missing>')}"
            )

    if not kept_indices:
        raise ValueError(
            f"Span-DTW feasibility filtering removed every {split_name} sample: "
            f"mode={mode}, max_image_windows={max_image_windows}, "
            f"max_span_chars={max_span_chars}."
        )

    stats = FeasibilityStats(
        split_name=str(split_name),
        kept=len(kept_indices),
        removed=len(subset.indices) - len(kept_indices),
        max_required_spans=max_required,
        examples=tuple(removed_examples),
    )
    return Subset(base_dataset, kept_indices), stats

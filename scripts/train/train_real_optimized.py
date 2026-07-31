#!/usr/bin/env python3
"""Optimized real-dataset fine-tuning entrypoint.

Real manuscript transcripts can contain many more Unicode characters than the
fixed 63-window image sequence, especially when vocalization marks are present
or when training-only RTL stitching creates longer lines.  The generic synthetic
profile uses at most two characters per text span, which can make a positive
Span-DTW path mathematically impossible before the first optimizer step.

This wrapper configures a wider, real-data-only span lattice *before* importing
``train_optimized`` (and therefore before ``Parameters`` and the Arabic span
encoder are initialized).  All shorter span candidates are retained; the wider
cap merely allows Span-DTW to consume longer visible text cores when needed.
Synthetic training continues to use its existing defaults.
"""

from __future__ import annotations

import os
import sys


def _argument_value(name: str) -> str | None:
    """Read ``--name value`` or ``--name=value`` from the current CLI."""
    prefix = name + "="
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return None


def _configure_real_span_profile() -> int:
    dataset_type = (_argument_value("--dataset_type") or "").strip().lower()
    if dataset_type != "real":
        raise SystemExit(
            "train_real_optimized.py is only for --dataset_type real. "
            "Use train_optimized.py for synthetic training."
        )

    raw_limit = os.environ.get("REAL_MAX_TEXT_SPAN_CHARS", "32")
    try:
        span_limit = int(raw_limit)
    except ValueError as exc:
        raise SystemExit(
            f"REAL_MAX_TEXT_SPAN_CHARS must be an integer, got {raw_limit!r}."
        ) from exc
    if span_limit < 2:
        raise SystemExit(
            "REAL_MAX_TEXT_SPAN_CHARS must be at least 2 for real fine-tuning."
        )

    # These must be set before train_optimized imports train.py/Parameters.py.
    # setdefault keeps explicit experiment overrides authoritative.
    os.environ.setdefault("MAX_TEXT_SPAN_CHARS", str(span_limit))
    os.environ.setdefault("SPAN_MAX_CORE_CHARS_CAP", str(span_limit))
    return span_limit


REAL_SPAN_LIMIT = _configure_real_span_profile()

from train_optimized import main  # noqa: E402  (profile must be set first)


if __name__ == "__main__":
    print(
        "real_span_profile "
        f"max_text_span_chars={os.environ['MAX_TEXT_SPAN_CHARS']} "
        f"max_core_chars_cap={os.environ['SPAN_MAX_CORE_CHARS_CAP']}",
        flush=True,
    )
    main()

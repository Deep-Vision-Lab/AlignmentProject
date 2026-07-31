#!/usr/bin/env python3
"""Optimized real-dataset fine-tuning entrypoint.

This wrapper applies the safe real-training span profile before importing the
optimized trainer. Visible text cores remain at one or two characters, matching
the assumptions enforced by ``training_optimizations.py``. Training-only RTL
line stitching is disabled by default because concatenated transcripts can
require more Span-DTW transitions than the fixed image-window sequence provides.
Appearance, scan, blur, noise, morphology, and speckle augmentations remain
available through the normal real-data augmentation pipeline.
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

    raw_limit = os.environ.get("REAL_MAX_TEXT_SPAN_CHARS", "2")
    try:
        span_limit = int(raw_limit)
    except ValueError as exc:
        raise SystemExit(
            f"REAL_MAX_TEXT_SPAN_CHARS must be an integer, got {raw_limit!r}."
        ) from exc
    if span_limit < 1 or span_limit > 2:
        raise SystemExit(
            "REAL_MAX_TEXT_SPAN_CHARS must be 1 or 2 for truthful alignment. "
            "Do not enlarge spans to repair stitched transcripts."
        )

    # These must be set before train_optimized imports train.py/Parameters.py.
    os.environ.setdefault("MAX_TEXT_SPAN_CHARS", str(span_limit))
    os.environ.setdefault("SPAN_MAX_CORE_CHARS_CAP", str(span_limit))
    os.environ.setdefault("MAX_TEXT_TOKEN_CHARS", "2")
    os.environ.setdefault("MAX_WINDOWS_PER_SPAN", "3")
    os.environ.setdefault("SPAN_INCLUDE_SPACE_CONTEXT", "0")
    os.environ.setdefault("SPAN_ALLOW_CHARACTER_SPACE_SURFACES", "0")
    os.environ.setdefault("ALLOW_UNSAFE_SPAN_CONFIG", "0")
    os.environ.setdefault("REAL_AUG_STITCH_PROB", "0")
    return span_limit


REAL_SPAN_LIMIT = _configure_real_span_profile()

from train_optimized import main  # noqa: E402  (profile must be set first)


if __name__ == "__main__":
    print(
        "real_span_profile "
        f"max_text_span_chars={os.environ['MAX_TEXT_SPAN_CHARS']} "
        f"max_core_chars_cap={os.environ['SPAN_MAX_CORE_CHARS_CAP']} "
        f"stitch_probability={os.environ['REAL_AUG_STITCH_PROB']}",
        flush=True,
    )
    main()

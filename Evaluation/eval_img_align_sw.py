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

from Evaluation import sw_runner as _implementation

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

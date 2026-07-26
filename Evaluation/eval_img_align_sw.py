#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment."""
from __future__ import annotations

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

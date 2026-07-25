#!/usr/bin/env python3
"""Backward-compatible entrypoint for window-level image alignment.

The canonical implementation is ``Evaluation/eval_needleman_wunsch_windows.py``.
Needleman-Wunsch consumes the full window-to-window visual similarity matrix.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.eval_needleman_wunsch_windows import main


if __name__ == "__main__":
    main()

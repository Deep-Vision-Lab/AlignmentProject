#!/usr/bin/env python3
"""Deprecated compatibility entrypoint for window-level Needleman-Wunsch.

Needleman-Wunsch now runs directly over image windows. Use
``Evaluation/eval_needleman_wunsch_windows.py`` in new commands.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.eval_needleman_wunsch_windows import main


if __name__ == "__main__":
    print(
        "NOTE: eval_needleman_wunsch_words.py now delegates to the "
        "window-level evaluator.",
        file=sys.stderr,
    )
    main()

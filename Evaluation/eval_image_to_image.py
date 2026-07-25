#!/usr/bin/env python3
"""Backward-compatible entrypoint for image-to-image word alignment.

The old evaluator used proportional transcript-to-patch ranges. It now delegates
to the checkpoint-compatible Needleman-Wunsch evaluator, which obtains word
regions from the trained Span-DTW model and masks paired words on both lines.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.eval_needleman_wunsch_words import main

if __name__ == "__main__":
    main()

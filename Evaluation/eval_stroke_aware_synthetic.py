#!/usr/bin/env python3
"""Run synthetic Smith-Waterman evaluation with stroke-aware preprocessing."""
from __future__ import annotations

from Evaluation import eval_img_align_sw as shared
from stroke_aware_preprocessing import install_evaluation_preprocessing

# The shared evaluator reconstructs the CNN+BiLSTM checkpoint. This branch also
# needs the same soft-ink, distance-proximity, and Sobel channels used in
# training before images are sent through the visual encoder.
install_evaluation_preprocessing(shared._implementation)


if __name__ == "__main__":
    shared._implementation.main()

#!/usr/bin/env python3
"""Run synthetic Needleman-Wunsch evaluation with stroke-aware preprocessing."""
from __future__ import annotations

from Evaluation import eval_img_align_nw as shared
from stroke_aware_preprocessing import install_evaluation_preprocessing

# Reconstruct the checkpoint first, then install the same deterministic
# soft-ink, distance-proximity, and Sobel input channels used for validation.
install_evaluation_preprocessing(shared._implementation)


if __name__ == "__main__":
    shared._implementation.main()

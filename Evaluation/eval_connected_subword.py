#!/usr/bin/env python3
"""Connected-subword bootstrap for the canonical real Smith-Waterman evaluator."""
from connected_subword_mode import install_connected_subword_mode

install_connected_subword_mode()

from Evaluation import eval_img_align_sw as implementation


if __name__ == "__main__":
    implementation.main()

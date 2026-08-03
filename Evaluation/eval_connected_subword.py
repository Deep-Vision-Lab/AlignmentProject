#!/usr/bin/env python3
"""Connected-subword bootstrap for the canonical real Smith-Waterman evaluator."""
from connected_subword_mode import install_connected_subword_mode

install_connected_subword_mode()

from Evaluation import eval_img_align_sw as implementation
from stroke_aware_preprocessing import install_evaluation_preprocessing

# Patch after eval_img_align_sw installs its geometry/runtime hooks, then replace
# the runner's model loader and transform with checkpoint-aware stroke channels.
install_evaluation_preprocessing(implementation)


if __name__ == "__main__":
    implementation.main()

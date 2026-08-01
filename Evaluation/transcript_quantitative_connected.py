#!/usr/bin/env python3
"""Connected-subword bootstrap for transcript-supervised evaluation."""
from connected_subword_mode import (
    install_connected_subword_evaluation,
    install_connected_subword_mode,
)

install_connected_subword_mode()

from Evaluation import _eval_utils

install_connected_subword_evaluation(_eval_utils)

from Evaluation import transcript_quantitative as implementation


if __name__ == "__main__":
    implementation.main()

#!/usr/bin/env python3
"""Run the canonical real Smith-Waterman evaluator without saving PNG files."""
from __future__ import annotations

from Evaluation import eval_img_align_sw as evaluation


def _skip_visualization(*_args, **_kwargs):
    return None


# ``sw_runner.evaluate_sample`` resolves this function from its own module
# globals. Replacing it preserves all alignment and bbox metric calculations
# while skipping the expensive per-pair visualization render and PNG write.
evaluation._implementation.save_visualization = _skip_visualization


if __name__ == "__main__":
    evaluation._implementation.main()

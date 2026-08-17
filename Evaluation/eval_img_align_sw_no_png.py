#!/usr/bin/env python3
"""Run the canonical real Smith-Waterman evaluator without saving PNG files."""
from __future__ import annotations

import sys

# Configure the historical/new CNN backbone before importing evaluation modules,
# because those modules import EmbeddingModel at module-import time.
def _configure_checkpoint_backbone() -> None:
    weights = None
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == "--weights" and index + 1 < len(sys.argv):
            weights = sys.argv[index + 1]
            break
        if argument.startswith("--weights="):
            weights = argument.split("=", 1)[1]
            break
    if not weights:
        return
    from checkpoint_backbone_runtime import configure_for_checkpoint

    backbone = configure_for_checkpoint(weights)
    if backbone:
        print(f"evaluation_cnn_backbone={backbone}", flush=True)


_configure_checkpoint_backbone()

from Evaluation import eval_img_align_sw as evaluation


def _skip_visualization(*_args, **_kwargs):
    return None


# ``sw_runner.evaluate_sample`` resolves this function from its own module
# globals. Replacing it preserves all alignment and bbox metric calculations
# while skipping the expensive per-pair visualization render and PNG write.
evaluation._implementation.save_visualization = _skip_visualization


if __name__ == "__main__":
    evaluation._implementation.main()

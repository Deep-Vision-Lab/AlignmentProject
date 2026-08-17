#!/usr/bin/env python3
"""Run the canonical real Smith-Waterman evaluator without saving PNG files."""
from __future__ import annotations

import sys


def _weights_argument() -> str | None:
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == "--weights" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if argument.startswith("--weights="):
            return argument.split("=", 1)[1]
    return None


weights = _weights_argument()
if weights:
    from checkpoint_backbone_runtime import configure_for_checkpoint

    backbone = configure_for_checkpoint(weights)
    if backbone:
        print(f"evaluation_cnn_backbone={backbone}", flush=True)

from Evaluation import eval_img_align_sw as evaluation
import Evaluation._eval_utils as eval_utils
import model_backend


def _branch_embedding_model(
    window_size=32,
    stride=16,
    vector_size=128,
    device="cuda",
    use_flip=False,
    use_bilstm=True,
    bilstm_layers=2,
    bilstm_hidden_dim=None,
    use_local_grouping=True,
    local_group_size=3,
    **kwargs,
):
    return model_backend.build_visual_model(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        use_local_grouping=use_local_grouping,
        local_group_size=local_group_size,
        **kwargs,
    )


eval_utils.EmbeddingModel = _branch_embedding_model
print(f"evaluation_model_backend={model_backend.MODEL_NAME}", flush=True)


def _skip_visualization(*_args, **_kwargs):
    return None


evaluation._implementation.save_visualization = _skip_visualization


if __name__ == "__main__":
    evaluation._implementation.main()

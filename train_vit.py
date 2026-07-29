#!/usr/bin/env python3
"""Train the ViT visual encoder through the optimized generic DDP trainer.

The generic ``train.py`` owns dataset construction, losses, AMP, distributed
sampling, NCCL setup, checkpointing, validation, and W&B. This entry point only
replaces the visual-model constructor and records the ViT architecture in the
checkpoint configuration.

It is safe to launch with either plain Python (one GPU) or torchrun (one process
per GPU). Import ``train`` before importing torch-dependent ViT modules so the
trainer can isolate each torchrun rank's CUDA device before Torch/JAX imports.
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("VISUAL_ENCODER_TYPE", "vit")
os.environ.setdefault("USE_BILSTM", "0")
os.environ.setdefault("USE_LOCAL_WINDOW_GROUPING", "0")

# This import performs per-rank CUDA isolation before importing torch/JAX.
import train as trainer

from vit_embedding_model import build_vit_from_environment, prepare_vit_model


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _vit_constructor(
    window_size=32,
    stride=16,
    vector_size=128,
    device="cuda",
    use_flip=False,
    **_ignored,
):
    model = build_vit_from_environment(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
    )
    return prepare_vit_model(model)


# train.py resolves EmbeddingModel at module scope. Replace only that constructor;
# every other optimized training component remains unchanged.
trainer.EmbeddingModel = _vit_constructor


_original_model_config = trainer.model_config


def _vit_model_config(stride, args):
    config = _original_model_config(stride, args)
    config.update(
        {
            "visual_encoder_type": "vit",
            "use_bilstm": False,
            "use_local_window_grouping": False,
            "vit_input_height": _integer("VIT_INPUT_HEIGHT", 128),
            "vit_layers": _integer("VIT_LAYERS", 4),
            "vit_heads": _integer("VIT_HEADS", 4),
            "vit_mlp_dim": _integer("VIT_MLP_DIM", 512),
            "vit_dropout": _number("VIT_DROPOUT", 0.10),
            "vit_max_tokens": _integer("VIT_MAX_TOKENS", 256),
            "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
            "ddp_static_graph": _flag("DDP_STATIC_GRAPH", True),
        }
    )
    return config


trainer.model_config = _vit_model_config


# Preserve the optimized branch's static-graph DDP behavior when the installed
# PyTorch version supports the argument. Because trainer.DDP remains a class,
# train.py's isinstance-based unwrap/checkpoint logic continues to work.
_BaseDDP = trainer.DDP
_DDP_SUPPORTS_STATIC_GRAPH = (
    "static_graph" in inspect.signature(_BaseDDP.__init__).parameters
)


class ViTDistributedDataParallel(_BaseDDP):
    def __init__(self, module, *args, **kwargs):
        if _DDP_SUPPORTS_STATIC_GRAPH and _flag("DDP_STATIC_GRAPH", True):
            kwargs.setdefault("static_graph", True)
        super().__init__(module, *args, **kwargs)


trainer.DDP = ViTDistributedDataParallel


if __name__ == "__main__":
    trainer.main()

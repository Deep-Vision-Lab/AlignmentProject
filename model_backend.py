"""Branch-selected visual backend for hierarchical letter grounding.

The image side keeps the proven 4-layer raw-RGB ViT baseline, but inserts a
trainable local *depiction* head between primitive patch features and the
contextual Transformer.  The depiction head receives direct letter-level VLM
supervision in ``vlm_letter_grounding.py``.
"""
from __future__ import annotations

import os

import Parameters as P
from vlm_letter_grounding import apply_branch_config, install_training_objective

# Apply this branch's quality-first experiment settings before the trainer builds
# loaders/criterion/model.  Re-export standard settings so helper modules see the
# same resolved values even though the shared Parameters.py remains untouched.
apply_branch_config(P)
P.export_environment()

MODEL_NAME = "vit_vlm_letter_depiction"
VISUAL_ENCODER_TYPE = "vit"


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


def build_visual_model(
    *,
    window_size,
    stride,
    vector_size,
    device,
    use_flip,
    **_ignored,
):
    from embeddingModel import build_vit_from_environment
    from vlm_letter_grounding import attach_depiction_head

    model = build_vit_from_environment(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
    )
    return attach_depiction_head(model)


def install_training_backend(base_module) -> None:
    os.environ["VISUAL_ENCODER_TYPE"] = VISUAL_ENCODER_TYPE
    os.environ["USE_BILSTM"] = "0"
    os.environ["USE_LOCAL_WINDOW_GROUPING"] = "0"

    def constructor(
        window_size=32,
        stride=16,
        vector_size=128,
        device="cuda",
        use_flip=False,
        **kwargs,
    ):
        return build_visual_model(
            window_size=window_size,
            stride=stride,
            vector_size=vector_size,
            device=device,
            use_flip=use_flip,
            **kwargs,
        )

    base_module.EmbeddingModel = constructor
    install_training_objective(base_module)


def prepare_visual_model(model) -> None:
    from embeddingModel import prepare_vit_model

    prepare_vit_model(model)


def visual_model_config() -> dict:
    from vlm_letter_grounding import model_config as grounding_model_config

    config = {
        "model_backend": MODEL_NAME,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "use_bilstm": False,
        "use_local_window_grouping": False,
        "local_representation": "trainable_letter_depiction",
        "context_input": "letter_depiction_tokens",
        "vit_input_height": _integer("VIT_INPUT_HEIGHT", 128),
        "vit_layers": _integer("VIT_LAYERS", 4),
        "vit_heads": _integer("VIT_HEADS", 4),
        "vit_mlp_dim": _integer("VIT_MLP_DIM", 512),
        "vit_dropout": _number("VIT_DROPOUT", 0.10),
        "vit_max_tokens": _integer("VIT_MAX_TOKENS", 256),
        "vit_position_base_tokens": _integer("VIT_POSITION_BASE_TOKENS", 63),
        "vit_binarize_input": _flag("VIT_BINARIZE_INPUT", False),
        "vit_binarize_method": "otsu" if _flag("VIT_BINARIZE_INPUT", False) else "none",
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
    config.update(grounding_model_config(P))
    return config

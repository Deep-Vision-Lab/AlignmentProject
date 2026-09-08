"""Direct CNN window vectors followed by the visual Transformer.

Exact RGB windows enter one shared trainable spatial CNN. Its 128D outputs
receive local letter supervision and feed the 4-layer contextual Transformer.
No extra letter-depiction MLP is added in this variant.
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
# The shared optimization validator and text encoder were written for a later
# <=2-character ablation. This branch deliberately restores the proven
# three-character contextual span baseline.
os.environ["ALLOW_UNSAFE_SPAN_CONFIG"] = "1"
os.environ["SPAN_MAX_CORE_CHARS_CAP"] = "3"

MODEL_NAME = "vit_vlm_letter_depiction_window_cnn"
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
        "window_cnn_enabled": bool(P.window_cnn_enabled),
        "letter_depiction_head": False,
        "window_cnn_architecture": "spatial_16_32_64_pool4x2_v1",
        "window_extraction": "exact_rgb_unfold",
        "local_representation": "direct_cnn_window_features",
        "context_input": "direct_cnn_window_tokens",
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


"""Meta DINOv3 ConvNeXt branch backend.

The shared data/loss/training runtime is unchanged.  Only the visual window encoder
is replaced by the official DINOv3 ConvNeXt-Tiny foundation model plus a projection
into the project's embedding dimension. ``USE_BILSTM=0/1`` remains available as a
controlled sequence-context ablation on top of the same ConvNeXt windows.
"""
from __future__ import annotations

import os

MODEL_NAME = "dinov3_convnext"
VISUAL_ENCODER_TYPE = "dinov3_convnext"


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


def build_visual_model(
    *,
    window_size,
    stride,
    vector_size,
    device,
    use_flip,
    use_bilstm=True,
    bilstm_layers=2,
    bilstm_hidden_dim=None,
    use_local_grouping=True,
    local_group_size=3,
    **_ignored,
):
    from dinov3_convnext_embedding_model import build_dinov3_from_environment

    return build_dinov3_from_environment(
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
    )


def install_training_backend(base_module) -> None:
    os.environ["VISUAL_ENCODER_TYPE"] = VISUAL_ENCODER_TYPE

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


def prepare_visual_model(model) -> None:
    from dinov3_convnext_embedding_model import prepare_dinov3_model

    prepare_dinov3_model(model)


def visual_model_config() -> dict:
    return {
        "model_backend": MODEL_NAME,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "dinov3_arch": "dinov3_convnext_tiny",
        "dinov3_freeze_backbone": _flag("DINOV3_FREEZE_BACKBONE", True),
        "dinov3_window_chunk_size": _integer("DINOV3_WINDOW_CHUNK_SIZE", 256),
        "use_bilstm": _flag("USE_BILSTM", True),
        "use_local_window_grouping": _flag("USE_LOCAL_WINDOW_GROUPING", True),
        "torch_compile_visual": False,
    }

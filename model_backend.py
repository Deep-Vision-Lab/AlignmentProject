"""Meta DINOv3 ConvNeXt branch backend.

The DINO backbone supplies per-window visual features. ``DINOV3_SEQUENCE_MODE``
selects how the 63 projected window vectors are connected:

- ``bilstm`` preserves historical DINO checkpoint compatibility;
- ``transformer`` applies global self-attention and disables three-window fusion;
- ``none`` leaves the projected windows independent.
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


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _sequence_mode(use_bilstm: bool | None = None) -> str:
    raw = os.environ.get("DINOV3_SEQUENCE_MODE", "").strip().lower().replace("-", "_")
    if raw in {"", "auto", "legacy"}:
        if use_bilstm is None:
            use_bilstm = _flag("USE_BILSTM", True)
        return "bilstm" if bool(use_bilstm) else "none"
    aliases = {
        "lstm": "bilstm",
        "bi_lstm": "bilstm",
        "attention": "transformer",
        "self_attention": "transformer",
        "off": "none",
        "identity": "none",
    }
    raw = aliases.get(raw, raw)
    if raw not in {"bilstm", "transformer", "none"}:
        raise ValueError(
            "DINOV3_SEQUENCE_MODE must be bilstm, transformer, or none; "
            f"got {raw!r}."
        )
    return raw


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

    mode = _sequence_mode(use_bilstm)
    if mode == "transformer" and use_local_grouping:
        raise ValueError(
            "DINOV3_SEQUENCE_MODE=transformer requires USE_LOCAL_WINDOW_GROUPING=0."
        )
    return build_dinov3_from_environment(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
        use_bilstm=(mode == "bilstm"),
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        use_local_grouping=use_local_grouping,
        local_group_size=local_group_size,
        sequence_mode=mode,
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
    mode = _sequence_mode()
    grouping = _flag("USE_LOCAL_WINDOW_GROUPING", True)
    if mode == "transformer" and grouping:
        raise ValueError(
            "DINOV3_SEQUENCE_MODE=transformer requires USE_LOCAL_WINDOW_GROUPING=0."
        )
    return {
        "model_backend": MODEL_NAME,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "dinov3_arch": "dinov3_convnext_tiny",
        "dinov3_freeze_backbone": _flag("DINOV3_FREEZE_BACKBONE", True),
        "dinov3_window_chunk_size": _integer("DINOV3_WINDOW_CHUNK_SIZE", 256),
        "dinov3_sequence_mode": mode,
        "dinov3_transformer_layers": _integer("DINOV3_TRANSFORMER_LAYERS", 4),
        "dinov3_transformer_heads": _integer("DINOV3_TRANSFORMER_HEADS", 4),
        "dinov3_transformer_mlp_dim": _integer("DINOV3_TRANSFORMER_MLP_DIM", 512),
        "dinov3_transformer_dropout": _number("DINOV3_TRANSFORMER_DROPOUT", 0.10),
        "dinov3_transformer_max_tokens": _integer("DINOV3_TRANSFORMER_MAX_TOKENS", 256),
        "dinov3_transformer_position_base_tokens": _integer(
            "DINOV3_TRANSFORMER_POSITION_BASE_TOKENS", 63
        ),
        "use_bilstm": mode == "bilstm",
        "bilstm_layers": _integer("BILSTM_LAYERS", 2),
        "bilstm_hidden_dim": _integer("BILSTM_HIDDEN_DIM", 128),
        "use_local_window_grouping": grouping,
        "local_group_size": _integer("LOCAL_GROUP_SIZE", 3),
        "torch_compile_visual": False,
    }

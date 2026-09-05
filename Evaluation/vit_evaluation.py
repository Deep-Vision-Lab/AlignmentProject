"""Checkpoint-aware ViT reconstruction for shared evaluation utilities."""
from __future__ import annotations

import os

import torch

from Evaluation import _eval_utils
from embeddingModel import EmbeddingModel


def _bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _checkpoint_config(weights_path) -> dict:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    return {}


def _uses_letter_depiction(config: dict) -> bool:
    return bool(config.get("letter_depiction_enabled", False)) or str(
        config.get("local_representation", "")
    ).strip().lower() == "trainable_letter_depiction"


def _uses_cross_attention(config: dict) -> bool:
    return _bool(config.get("cross_attention_enabled", False), False)


def _attach_cross_attention_from_checkpoint(models) -> None:
    if models.text_model is None or not _uses_cross_attention(models.config):
        return
    from vlm_pair_cross_attention import SymmetricPairCrossAttention

    if not hasattr(models.text_model, "pair_cross_attention"):
        module = SymmetricPairCrossAttention(
            int(models.config.get("vector_size", 128)),
            num_heads=int(models.config.get("cross_attention_heads", 4)),
            dropout=float(models.config.get("cross_attention_dropout", 0.10)),
            ff_multiplier=int(models.config.get("cross_attention_ff_multiplier", 2)),
            initial_gate=float(models.config.get("cross_attention_initial_gate", 0.20)),
        ).to(models.device)
        models.text_model.add_module("pair_cross_attention", module)

    # The shared loader already loaded the ordinary text state before this pair
    # module existed. Reload once so pair_cross_attention.* parameters are also
    # restored from the checkpoint.
    checkpoint = models.checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("text_encoder_state_dict")
        if state is None:
            state = checkpoint.get("text_embedder_state_dict")
        if state:
            models.text_model.load_state_dict(
                _eval_utils._strip_module_prefix(state), strict=False
            )
    models.text_model.eval()


def install_vit_evaluation_loader() -> None:
    """Reconstruct the exact hierarchy recorded in a checkpoint."""
    if getattr(_eval_utils, "_vit_evaluation_loader_installed", False):
        return

    original_loader = _eval_utils.load_evaluation_models

    def load_evaluation_models(weights_path, device="auto", load_text_model=True):
        config = _checkpoint_config(weights_path)
        encoder_type = str(
            config.get(
                "visual_encoder_type",
                os.environ.get("VISUAL_ENCODER_TYPE", "vit"),
            )
        ).strip().lower()
        if encoder_type != "vit":
            return original_loader(weights_path, device, load_text_model)

        os.environ["SPAN_MAX_CORE_CHARS_CAP"] = str(
            int(config.get("max_text_span_chars", 3))
        )
        previous_constructor = _eval_utils.EmbeddingModel

        def vit_constructor(
            window_size=32,
            stride=16,
            vector_size=128,
            device="cpu",
            use_flip=False,
            **_ignored,
        ):
            binarize_input = _bool(
                config.get("vit_binarize_input", False),
                False,
            )
            model = EmbeddingModel(
                window_size=int(window_size),
                stride=int(stride),
                vector_size=int(vector_size),
                device=device,
                use_flip=_bool(use_flip),
                input_height=int(config.get("vit_input_height", 128)),
                vit_layers=int(config.get("vit_layers", 4)),
                vit_heads=int(config.get("vit_heads", 4)),
                vit_mlp_dim=int(config.get("vit_mlp_dim", 512)),
                vit_dropout=float(config.get("vit_dropout", 0.10)),
                vit_max_tokens=int(config.get("vit_max_tokens", 256)),
                vit_position_base_tokens=int(
                    config.get("vit_position_base_tokens", 63)
                ),
                vit_binarize_input=binarize_input,
            )
            if _uses_letter_depiction(config):
                from vlm_letter_grounding import attach_depiction_head

                model = attach_depiction_head(model)
            return model

        try:
            _eval_utils.EmbeddingModel = vit_constructor
            models = original_loader(weights_path, device, load_text_model)
            _attach_cross_attention_from_checkpoint(models)
            return models
        finally:
            _eval_utils.EmbeddingModel = previous_constructor

    _eval_utils.load_evaluation_models = load_evaluation_models
    _eval_utils._vit_evaluation_loader_installed = True

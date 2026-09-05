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


def install_vit_evaluation_loader() -> None:
    """Reconstruct the canonical ViT with the architecture stored in a checkpoint."""
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

        # ArabicSpanTextEncoder has an environment-level visible-core cap.  Make
        # evaluation reconstruct the same span semantics recorded at training.
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
            # Legacy ViT checkpoints were trained before model-side binarization
            # existed and therefore do not contain vit_binarize_input. Missing
            # means False. New checkpoints explicitly store True and use Otsu.
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
            return original_loader(weights_path, device, load_text_model)
        finally:
            _eval_utils.EmbeddingModel = previous_constructor

    _eval_utils.load_evaluation_models = load_evaluation_models
    _eval_utils._vit_evaluation_loader_installed = True

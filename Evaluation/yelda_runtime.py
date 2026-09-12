"""Strict image-only checkpoint loading and explicit pair fusion for Yelda."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from embeddingModel import EmbeddingModel
from Evaluation._eval_utils import EvaluationModels, _model_state, get_image_features


def flag(value):
    return value.strip().lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def read_checkpoint(path):
    # These are the user's trusted training checkpoints, including metadata.
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint was being overwritten while loading; retry with a stable checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("Yelda evaluation requires a training checkpoint with model_config")
    if flag(checkpoint["model_config"].get("window_cnn_enabled", False)):
        raise ValueError("This branch evaluates original depiction checkpoints; use the window-CNN branch for window_cnn_enabled=True")
    checkpoint["_evaluation_sha256"] = digest.hexdigest()
    return checkpoint


def load_visual_models(checkpoint, device="auto", expected_branch="auto"):
    config = dict(checkpoint["model_config"])
    if flag(config.get("window_cnn_enabled", False)):
        raise ValueError("Window-CNN checkpoints require their corresponding branch")
    family = str(config.get("architecture_family", ""))
    restoration = family == "restoration-positive-dtw-window-encoder"
    spatial = family == "cfm-inspired-spatial-language-alignment"
    cross = flag(config.get("cross_attention_enabled", False))
    branch = (
        "restoration"
        if restoration
        else ("spatial" if spatial else ("cross" if cross else "hierarchy"))
    )
    if expected_branch not in {"auto", branch}:
        raise ValueError(f"Requested {expected_branch}, but checkpoint is {branch}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = EmbeddingModel(
        window_size=int(config.get("window_size", 32)),
        stride=int(config.get("stride", 16)),
        vector_size=int(config.get("vector_size", 128)),
        device=dev,
        use_flip=str(config.get("lang", "Arabic")).lower() == "arabic",
        input_height=int(config.get("vit_input_height", 128)),
        vit_layers=int(config.get("vit_layers", 4)),
        vit_heads=int(config.get("vit_heads", 4)),
        vit_mlp_dim=int(config.get("vit_mlp_dim", 512)),
        vit_dropout=float(config.get("vit_dropout", 0.10)),
        vit_max_tokens=int(config.get("vit_max_tokens", 256)),
        vit_position_base_tokens=int(config.get("vit_position_base_tokens", 63)),
        vit_binarize_input=flag(config.get("vit_binarize_input", False)),
        vit_binarize_contrast_threshold=float(config.get("vit_binarize_contrast_threshold", 0.15)),
    ).to(dev)
    # Reconstruct the exact branch-specific stages BEFORE loading.
    if restoration:
        from types import SimpleNamespace
        from vlm_restoration_positive_dtw import attach_restoration_dtw_stages

        restoration_config = SimpleNamespace(
            restoration_decoder_channels=int(
                config.get("restoration_decoder_channels", 64)
            ),
            restoration_contrast_scale=float(
                config.get("restoration_contrast_scale", 0.15)
            ),
            # Historical checkpoints from this branch predate the config field
            # and therefore used the residual MLP. New diagnostic-first runs
            # record identity explicitly.
            restoration_semantic_adapter=str(
                config.get("restoration_semantic_adapter", "residual_mlp")
            ),
            restoration_local_encoder=str(
                config.get("restoration_local_encoder", "fullheight_conv")
            ),
        )
        model = attach_restoration_dtw_stages(model, restoration_config)
    elif spatial:
        from types import SimpleNamespace
        from vlm_spatial_language_alignment import attach_spatial_language_stages

        spatial_config = SimpleNamespace(
            spatial_affinity_radius=int(config.get("spatial_affinity_radius", 3)),
            spatial_affinity_temperature=float(config.get("spatial_affinity_temperature", 0.15)),
            spatial_affinity_distance_penalty=float(config.get("spatial_affinity_distance_penalty", 0.12)),
            spatial_affinity_initial_gate=float(config.get("spatial_affinity_initial_gate", 0.15)),
            context_residual_initial_gate=float(config.get("context_residual_initial_gate", 0.25)),
        )
        model = attach_spatial_language_stages(model, spatial_config)
    else:
        from vlm_letter_grounding import attach_depiction_head
        model = attach_depiction_head(model)
    model.load_state_dict(_model_state(checkpoint), strict=True)
    model.eval()
    # No Arabic text encoder/tokenizer is constructed or used for alignment.
    models = EvaluationModels(model, None, config, checkpoint, dev)
    models.pair_cross_attention = None
    if cross:
        try:
            from vlm_pair_cross_attention import SymmetricPairCrossAttention
        except ImportError as exc:
            raise RuntimeError("Evaluate a cross checkpoint from the cross-attention branch") from exc
        module = SymmetricPairCrossAttention(
            int(config.get("vector_size", 128)),
            num_heads=int(config.get("cross_attention_heads", 4)),
            dropout=float(config.get("cross_attention_dropout", 0.10)),
            ff_multiplier=int(config.get("cross_attention_ff_multiplier", 2)),
            initial_gate=float(config.get("cross_attention_initial_gate", 0.20)),
        ).to(dev)
        text_state = checkpoint.get("text_encoder_state_dict")
        if text_state is None:
            text_state = checkpoint.get("text_embedder_state_dict", {})
        state = {}
        for key, value in text_state.items():
            key = key.removeprefix("module.")
            if key.startswith("pair_cross_attention."):
                state[key.removeprefix("pair_cross_attention.")] = value
        if not state:
            raise ValueError("Cross checkpoint has no saved pair_cross_attention weights")
        module.load_state_dict(state, strict=True)
        module.eval()
        models.pair_cross_attention = module
    return models


def configure_image_preprocessing(models, image_preprocessing):
    """Override pixel binarization for the original-image evaluation experiment."""
    if image_preprocessing not in {"original", "training"}:
        raise ValueError(f"Unknown image preprocessing: {image_preprocessing}")
    checkpoint_binarize = flag(models.config.get("vit_binarize_input", False))
    effective = checkpoint_binarize if image_preprocessing == "training" else False
    models.image_model.vit_binarize_input = effective
    models.image_model.vit_encoder.binarize_input = effective
    # Keep the saved configuration intact; report evaluation overrides separately.
    return {"image_preprocessing": image_preprocessing,
            "checkpoint_vit_binarize_input": checkpoint_binarize,
            "effective_vit_binarize_input": effective,
            "tensor_normalization": "ImageNet mean/std"}


def pair_features(models, image1, image2, representation="primary"):
    """Pair-local state only: an error cannot mix features from different pairs."""
    first = get_image_features(models, image1, "synthetic")
    second = get_image_features(models, image2, "synthetic")
    module = models.pair_cross_attention
    if representation in {"primary", "joint"} and module is not None:
        with torch.inference_mode():
            fused1, fused2, _, _ = module(
                first.contextual.unsqueeze(0), second.contextual.unsqueeze(0),
                ink1=first.ink.unsqueeze(0), ink2=second.ink.unsqueeze(0),
                min_ink=float(models.config.get("cross_attention_min_ink", 0.01)),
                return_weights=False,
            )
        first = replace(first, contextual=F.normalize(fused1[0].float(), dim=-1))
        second = replace(second, contextual=F.normalize(fused2[0].float(), dim=-1))
    for features in (first, second):
        if representation == "joint" and not torch.isfinite(features.local).all():
            raise ValueError("Non-finite local image embeddings")
        if not torch.isfinite(features.select("local" if representation == "local" else "contextual")).all():
            raise ValueError("Non-finite image embeddings")
    return first, second

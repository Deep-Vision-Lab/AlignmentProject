"""Safe pretrained-checkpoint migration for the one-layer ViT transition."""
from __future__ import annotations

import re

import torch

_LAYER_KEY = re.compile(r"^vit_encoder\.encoder\.layers\.(\d+)\.")


def _is_removed_transformer_layer(key: str, kept_layers: int) -> bool:
    match = _LAYER_KEY.match(str(key))
    return bool(match and int(match.group(1)) >= int(kept_layers))


def install(base_module) -> None:
    """Allow pretrained deeper ViTs to initialize the new one-layer model safely.

    Resume remains strict because optimizer/scheduler state belongs to the exact
    architecture. Only --weights/pretrained initialization may discard encoder
    layers above the configured depth.
    """
    if getattr(base_module, "_vit_checkpoint_migration_installed", False):
        return

    original_loader = base_module._load_initial_states

    def load_initial_states(args, model, text_encoder):
        if args.resume or not args.pretrained_weights:
            return original_loader(args, model, text_encoder)

        loaded = torch.load(args.pretrained_weights, map_location=base_module.P.device)
        state = base_module.extract_model_state(loaded)
        incompatible = model.load_state_dict(state, strict=False)

        raw_model = base_module._unwrap_model(model)
        kept_layers = int(getattr(raw_model, "vit_layers", 1))
        serious_missing = list(incompatible.missing_keys)
        serious_unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not _is_removed_transformer_layer(key, kept_layers)
        ]
        removed_layers = [
            key
            for key in incompatible.unexpected_keys
            if _is_removed_transformer_layer(key, kept_layers)
        ]

        if serious_missing or serious_unexpected:
            raise RuntimeError(
                "Pretrained ViT checkpoint is incompatible beyond the intentional "
                f"depth reduction: missing={serious_missing[:10]} "
                f"unexpected={serious_unexpected[:10]}"
            )

        if isinstance(loaded, dict) and "text_encoder_state_dict" in loaded:
            text_encoder.load_state_dict(
                loaded["text_encoder_state_dict"], strict=False
            )
        elif isinstance(loaded, dict) and "text_embedder_state_dict" in loaded:
            text_encoder.load_state_dict(
                loaded["text_embedder_state_dict"], strict=False
            )

        if base_module.CTX.is_main and removed_layers:
            removed_indices = sorted(
                {
                    int(_LAYER_KEY.match(key).group(1))
                    for key in removed_layers
                    if _LAYER_KEY.match(key)
                }
            )
            print(
                "Loaded pretrained ViT with intentional depth migration: "
                f"kept_layers=0..{kept_layers - 1} "
                f"discarded_layers={removed_indices}",
                flush=True,
            )
        return None

    base_module._load_initial_states = load_initial_states
    base_module._vit_checkpoint_migration_installed = True

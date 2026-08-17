"""Keep AraBERT frozen while training the bridge text projection head.

The Arabic span encoder intentionally uses a frozen AraBERT backbone followed by a
small trainable adapter into the shared image/text embedding dimension.  For bridge
training we preserve that design: the language backbone stays fixed, while the
projection, LayerNorm, and learned special-token embeddings remain trainable.

This is preferable to freezing the whole text encoder because the projection is the
learned coordinate transform from AraBERT's hidden size into the AlignmentProject
128/256-D space.  The bridge fine-tuning learning rate is already very small, so v1
keeps the adapter on the normal optimizer LR rather than introducing another
confounding hyperparameter.
"""
from __future__ import annotations


def install(base) -> None:
    original_build_text_encoder = base.build_text_encoder

    def build_bridge_text_encoder():
        encoder = original_build_text_encoder()

        # Enforce a frozen language-model backbone even if a future global setting
        # changes, but preserve the trainable projection-head policy.
        backbone = getattr(encoder, "backbone", None)
        if backbone is not None:
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
            backbone.eval()

        trainable_names = []
        trainable_parameters = 0
        total_parameters = 0
        for name, parameter in encoder.named_parameters():
            total_parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += parameter.numel()
                trainable_names.append(name)

        if getattr(base.CTX, "is_main", True):
            print(
                "bridge_text_policy "
                "arabert_frozen=1 projection_trainable=1 "
                f"trainable_parameters={trainable_parameters} "
                f"total_parameters={total_parameters} "
                f"trainable_names={','.join(trainable_names)}",
                flush=True,
            )
        return encoder

    base.build_text_encoder = build_bridge_text_encoder

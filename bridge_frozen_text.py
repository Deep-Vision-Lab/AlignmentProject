"""Freeze the complete text encoder for real-conditioned bridge training.

The normal Arabic span encoder freezes AraBERT but intentionally leaves its
projection and learned special embeddings trainable.  The bridge hypothesis is
stronger and cleaner if text is a fixed teacher: positive/negative synthetic text
vectors must not move to accommodate the image encoder.  This installer therefore
freezes every text-side parameter before the optimizer is built.
"""
from __future__ import annotations


def install(base) -> None:
    original_build_text_encoder = base.build_text_encoder

    def build_frozen_text_encoder():
        encoder = original_build_text_encoder()
        total = 0
        previously_trainable = 0
        for parameter in encoder.parameters():
            total += parameter.numel()
            if parameter.requires_grad:
                previously_trainable += parameter.numel()
            parameter.requires_grad_(False)
        encoder.eval()
        if getattr(base.CTX, "is_main", True):
            print(
                "bridge_text_teacher frozen=1 "
                f"parameters={total} previously_trainable={previously_trainable}",
                flush=True,
            )
        return encoder

    base.build_text_encoder = build_frozen_text_encoder

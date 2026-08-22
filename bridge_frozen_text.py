"""Keep AraBERT frozen while training the bridge text projection head.

The Arabic span encoder intentionally uses a frozen AraBERT backbone followed by a
small trainable adapter into the shared image/text embedding dimension. For bridge
training we preserve that design: the language backbone stays fixed, while the
projection, LayerNorm, and learned special-token embeddings remain trainable.

Bridge jobs run with Hugging Face networking disabled. Before constructing the
encoder, resolve one complete local AraBERT snapshot and temporarily use that
exact path for Transformers. The canonical model id is restored immediately so
checkpoint/config metadata remains portable.
"""
from __future__ import annotations

import os

from hf_offline_runtime import install_resolved_hf_environment


def install(base) -> None:
    original_build_text_encoder = base.build_text_encoder

    def build_bridge_text_encoder():
        logical_model_id = os.environ.get("ARABIC_TEXT_MODEL_ID", "").strip() or str(
            getattr(base.P, "arabic_text_model_name", "aubmindlab/bert-base-arabertv02")
        )
        resolution = install_resolved_hf_environment(
            logical_model_id,
            project_dir=os.environ.get("PROJECT_DIR") or os.getcwd(),
        )

        previous_model_name = getattr(base.P, "arabic_text_model_name", logical_model_id)
        base.P.arabic_text_model_name = str(resolution.snapshot_path)
        try:
            encoder = original_build_text_encoder()
        finally:
            base.P.arabic_text_model_name = previous_model_name

        if hasattr(encoder, "model_name"):
            encoder.model_name = logical_model_id

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
                f"model_id={logical_model_id} "
                f"resolved_snapshot={resolution.snapshot_path} "
                f"trainable_parameters={trainable_parameters} "
                f"total_parameters={total_parameters} "
                f"trainable_names={','.join(trainable_names)}",
                flush=True,
            )
        return encoder

    base.build_text_encoder = build_bridge_text_encoder

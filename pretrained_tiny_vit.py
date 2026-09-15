"""Initialize the branch's 1-D context transformer from pretrained DeiT-Tiny.

The branch keeps its manuscript-specific tokenization:
ResNet-18 window feature -> 192-D token -> transformer sequence context.

Only the DeiT transformer/position weights are transferred. DeiT's original
16x16 image patch projection and classification head are intentionally not used.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DEFAULT_TINY_VIT_MODEL = "facebook/deit-tiny-patch16-224"


def _copy_parameter(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.copy_(
        source.detach().to(device=destination.device, dtype=destination.dtype)
    )


def _horizontal_position_prior(
    position_embeddings: torch.Tensor,
    destination_tokens: int,
) -> torch.Tensor:
    """Collapse DeiT's 2-D patch positions into a 1-D horizontal line prior."""
    if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
        raise ValueError(
            "Expected pretrained position embeddings with shape [1,N,D], got "
            f"{tuple(position_embeddings.shape)}"
        )

    total = int(position_embeddings.shape[1])
    patch_positions = None
    for prefix_tokens in (1, 2, 0):
        patch_count = total - prefix_tokens
        side = int(round(math.sqrt(max(0, patch_count))))
        if side > 0 and side * side == patch_count:
            patch_positions = position_embeddings[:, prefix_tokens:]
            patch_positions = patch_positions.reshape(
                1, side, side, position_embeddings.shape[-1]
            )
            # DeiT patch order is row-major. Average over image height and keep
            # the horizontal axis, which matches our manuscript-line sequence.
            patch_positions = patch_positions.mean(dim=1)
            break

    if patch_positions is None:
        raise ValueError(
            "Could not infer the square DeiT patch grid from pretrained "
            f"position shape {tuple(position_embeddings.shape)}"
        )

    return F.interpolate(
        patch_positions.transpose(1, 2).float(),
        size=int(destination_tokens),
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def initialize_tiny_vit_from_pretrained(
    vit,
    *,
    model_name: str = DEFAULT_TINY_VIT_MODEL,
    local_files_only: bool = True,
) -> None:
    """Load ImageNet-pretrained DeiT-Tiny weights into the context transformer."""
    try:
        from transformers import ViTModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for pretrained ViT-Tiny initialization."
        ) from exc

    try:
        source = ViTModel.from_pretrained(
            str(model_name),
            local_files_only=bool(local_files_only),
        )
    except OSError as exc:
        mode = "local cache" if local_files_only else "Hugging Face"
        raise FileNotFoundError(
            f"Could not load pretrained ViT-Tiny '{model_name}' from {mode}. "
            "Run: python scripts/cache_pretrained_models.py before submitting "
            "the offline SLURM job."
        ) from exc

    config = source.config
    expected = {
        "hidden_size": 192,
        "num_hidden_layers": 12,
        "num_attention_heads": 3,
        "intermediate_size": 768,
    }
    actual = {name: int(getattr(config, name)) for name in expected}
    if actual != expected:
        raise ValueError(
            f"Pretrained checkpoint {model_name} is not ViT-Tiny: "
            f"expected={expected}, actual={actual}"
        )

    destination_layers = list(vit.encoder.layers)
    source_layers = list(source.encoder.layer)
    if len(destination_layers) != 12 or len(source_layers) != 12:
        raise ValueError(
            "Both destination and pretrained ViT-Tiny must have 12 layers."
        )

    with torch.no_grad():
        for destination, pretrained in zip(destination_layers, source_layers):
            _copy_parameter(destination.norm1.weight, pretrained.layernorm_before.weight)
            _copy_parameter(destination.norm1.bias, pretrained.layernorm_before.bias)
            destination.norm1.eps = float(pretrained.layernorm_before.eps)
            _copy_parameter(destination.norm2.weight, pretrained.layernorm_after.weight)
            _copy_parameter(destination.norm2.bias, pretrained.layernorm_after.bias)
            destination.norm2.eps = float(pretrained.layernorm_after.eps)

            query = pretrained.attention.attention.query
            key = pretrained.attention.attention.key
            value = pretrained.attention.attention.value
            _copy_parameter(
                destination.self_attn.in_proj_weight,
                torch.cat([query.weight, key.weight, value.weight], dim=0),
            )
            _copy_parameter(
                destination.self_attn.in_proj_bias,
                torch.cat([query.bias, key.bias, value.bias], dim=0),
            )
            _copy_parameter(
                destination.self_attn.out_proj.weight,
                pretrained.attention.output.dense.weight,
            )
            _copy_parameter(
                destination.self_attn.out_proj.bias,
                pretrained.attention.output.dense.bias,
            )
            _copy_parameter(destination.linear1.weight, pretrained.intermediate.dense.weight)
            _copy_parameter(destination.linear1.bias, pretrained.intermediate.dense.bias)
            _copy_parameter(destination.linear2.weight, pretrained.output.dense.weight)
            _copy_parameter(destination.linear2.bias, pretrained.output.dense.bias)

        if vit.encoder.norm is None:
            raise ValueError("Destination ViT-Tiny must have a final LayerNorm.")
        _copy_parameter(vit.encoder.norm.weight, source.layernorm.weight)
        _copy_parameter(vit.encoder.norm.bias, source.layernorm.bias)
        vit.encoder.norm.eps = float(source.layernorm.eps)

        one_dimensional_positions = _horizontal_position_prior(
            source.embeddings.position_embeddings,
            vit.position_embedding.shape[1],
        )
        _copy_parameter(vit.position_embedding, one_dimensional_positions)

    vit.context_pretrained = True
    vit.context_pretrained_model = str(model_name)
    vit.context_pretrained_position_mapping = "2d-horizontal-mean-to-1d"
    del source

"""Explicit opt-in architecture; legacy checkpoints retain their original graph."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANT = "resnet18_128d_5l_1h_no_pos"
LEGACY = "resnet18_tinyvit_192d_12l_3h"


class ReferenceSelfAttention(nn.MultiheadAttention):
    """PyTorch 2.0 odd-head-safe attention; respects eval dropout=off.

    Identical MHA parameters/math, bypassing only the native even-head fastpath.
    """
    _reference_attention = True

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if query is not key or key is not value:
            raise ValueError("This variant uses bidirectional self-attention only")
        q = query.transpose(0, 1) if self.batch_first else query
        result, weights = F.multi_head_attention_forward(q, q, q, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
            average_attn_weights=average_attn_weights, is_causal=is_causal)
        return (result.transpose(0, 1) if self.batch_first else result), weights


def is_compact(config):
    name = getattr(config, "architecture_variant", LEGACY)
    if name not in {VARIANT, LEGACY}:
        raise ValueError(f"Unknown architecture_variant: {name!r}")
    return name == VARIANT


class LinearContextFusion(nn.Module):
    """Returns h, not z: exactly concat -> Linear -> final LayerNorm."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, local, contextual):
        return self.norm(self.projection(torch.cat((local, contextual), dim=-1)))


def install_compact_context(model):
    vit = model.vit_encoder
    actual = (vit.embed_dim, len(vit.encoder.layers),
              vit.encoder.layers[0].self_attn.num_heads,
              vit.encoder.layers[0].linear1.out_features)
    if actual != (128, 5, 1, 512):
        raise ValueError(f"{VARIANT} requires (128,5,1,512), got {actual}")
    reference = next(vit.parameters())
    dropout = vit.encoder.layers[0].dropout.p
    # TransformerEncoder clones layers. Construct each layer separately instead.
    vit.encoder.layers = nn.ModuleList([
        nn.TransformerEncoderLayer(128, 1, 512, dropout=dropout,
                                   activation="gelu", batch_first=True, norm_first=True)
        for _ in range(5)
    ]).to(device=reference.device, dtype=reference.dtype)
    for layer in vit.encoder.layers:
        layer.self_attn.__class__ = ReferenceSelfAttention
    vit.position_embedding = None
    vit.position_mode = "none"
    vit.local_norm = nn.Identity()
    vit.input_dropout = nn.Identity()
    model.vision_norm = nn.Identity()


def metadata(config):
    if not is_compact(config):
        return {"architecture_variant": LEGACY}
    return {
        "architecture_variant": VARIANT, "vector_size": 128, "vit_embed_dim": 128,
        "vit_variant": "fresh-bidirectional-transformer", "vit_layers": 5,
        "vit_heads": 1, "vit_mlp_dim": 512, "position_mode": "none",
        "architecture_revision": VARIANT, "context_transformer_layers": 5,
        "fusion_mode": "single-linear-final-layernorm",
        "local_dropout": float(config.local_dropout), "input_dropout": 0.0,
        "fusion_dropout": 0.0, "local_projection": "Linear(512,128)->Dropout",
        "normalization_layout": "raw-local; transformer-preLN+finalLN; fused-LN-once->L2",
        "fusion": "Linear(256,128)->LayerNorm(128)",
        "initialization": "ImageNet-ResNet18; fresh-local/context/fusion",
        "tiny_vit_pretrained": False, "tiny_vit_pretrained_scope": "none",
        "context_patch_projection": "resnet18(512)->linear(128)->dropout",
        "transformer_position_prior": "none; DTW position prior is separate and unchanged",
    }

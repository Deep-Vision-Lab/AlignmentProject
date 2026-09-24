"""Bidirectional Transformer over window tokens, not an image patchifier."""
import math

import torch
from torch import nn
from torch.nn import functional as F


PRESETS = {'tiny': (128, 5, 1, 512), 'base': (256, 6, 4, 1024),
           'large': (512, 12, 8, 2048)}


class _SelfAttention(nn.MultiheadAttention):
    """Reference MHA path, including PyTorch 2.0 odd-head eval mask support."""
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        q = query.transpose(0, 1)
        output, weights = F.multi_head_attention_forward(
            q, q, q, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn, self.dropout,
            self.out_proj.weight, self.out_proj.bias, training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
        return output.transpose(0, 1), weights


class TransformerEncoder(nn.Module):
    def __init__(self, preset='tiny', embedding_dim=None, use_positional_encoding=False, dropout=0.):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f'Unknown Transformer preset: {preset}')
        dim, depth, heads, mlp = PRESETS[preset]
        self.dim = dim if embedding_dim is None else embedding_dim
        if self.dim % heads:
            raise ValueError('embedding_dim must be divisible by attention heads')
        self.heads, self.feedforward_dim = heads, mlp
        self.use_positional_encoding = use_positional_encoding
        # Construct each layer independently; no cloned initial parameters.
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.TransformerEncoderLayer(self.dim, heads, mlp, dropout,
                                                activation='gelu', batch_first=True, norm_first=True)
            layer.self_attn = _SelfAttention(self.dim, heads, dropout=dropout, batch_first=True)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, tokens, token_valid=None):
        if token_valid is not None:
            if token_valid.shape != tokens.shape[:2] or not token_valid.any(dim=1).all():
                raise ValueError('Each sequence needs at least one valid token')
        if self.use_positional_encoding:
            position = torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)[:, None]
            frequency = torch.exp(torch.arange(0, self.dim, 2, device=tokens.device, dtype=tokens.dtype)
                                  * (-math.log(10000.) / self.dim))
            pe = torch.zeros_like(tokens[0])
            pe[:, 0::2], pe[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)[:, :self.dim // 2]
            tokens = tokens + pe
        # Explicit layer arithmetic avoids the version-dependent fused inference
        # fastpath; parameters and pre-norm Transformer math are unchanged.
        mask = ~token_valid if token_valid is not None else None
        for layer in self.layers:
            normalized = layer.norm1(tokens)
            attention = layer.self_attn(normalized, normalized, normalized, key_padding_mask=mask)[0]
            tokens = tokens + layer.dropout1(attention)
            hidden = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(tokens)))))
            tokens = tokens + layer.dropout2(hidden)
        return self.norm(tokens)

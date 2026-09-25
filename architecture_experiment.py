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


def resolve_fusion(config):
    """Checkpoint-owned settings; historical concat metadata remains loadable."""
    get = config.get if isinstance(config, dict) else lambda k, d: getattr(config, k, d)
    mode = get("fusion_mode", "concat")
    if mode in {"single-linear-final-layernorm", "concat_projection_norm"}:
        mode = "concat"
    gate = get("use_gated_fusion", 0)
    if mode not in {"concat", "sum"}:
        raise ValueError(f"fusion_mode must be concat or sum, got {mode!r}")
    if gate not in (0, 1, False, True):
        raise ValueError("use_gated_fusion must be 0 or 1")
    if gate and mode != "sum":
        raise ValueError("Gated fusion requires fusion_mode=sum; concat + gate ON is invalid")
    return mode, bool(gate)


def add_fusion_arguments(parser):
    parser.add_argument("--fusion-mode", choices=("concat", "sum"), default="concat")
    parser.add_argument("--use-gated-fusion", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gate-diagnostics", type=int, choices=(0, 1), default=0,
                        help="Log detached gate statistics over valid windows on rank zero")


class SumContextFusion(nn.Module):
    """Same-window local residual: LN(L + C) or LN(L + sigmoid(MLP([L,C]))*C).

    Both encoders already return D features, so no alignment projections are
    needed. This module never mixes windows. Compact output is pre-L2 h; the
    caller applies its existing normalization chain. The older 192-D variant
    additionally normalized inside its fusion head, which is retained there.
    """
    def __init__(self, dim, gated=False, *, capture_gate=False, normalize_output=False):
        super().__init__()
        self.dim = int(dim)
        self.gate = (nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                   nn.Linear(dim, dim), nn.Sigmoid()) if gated else None)
        self.norm = nn.LayerNorm(dim)
        self.capture_gate = bool(capture_gate)
        self.normalize_output = bool(normalize_output)
        self.last_gate = None

    def forward(self, local, contextual):
        if local.shape != contextual.shape or local.ndim != 3 or local.shape[-1] != self.dim:
            raise ValueError("Fusion requires corresponding local/context tensors [B,T,D] of equal shape")
        if self.gate is None:
            self.last_gate = None
            fused = local + contextual
        else:
            alpha = self.gate(torch.cat((local, contextual), dim=-1))
            self.last_gate = alpha.detach() if self.capture_gate else None
            fused = local + alpha * contextual
        fused = self.norm(fused)
        if self.normalize_output:
            fused = F.normalize(fused.float(), p=2, dim=-1).to(dtype=local.dtype)
        return fused


def build_fusion(config, dim, *, compact):
    mode, gated = resolve_fusion(config)
    if mode == "concat":
        # Preserve original graph, state keys, and RNG draws exactly.
        if compact:
            return LinearContextFusion()
        from restoration_recommended_components import LocalContextFusion
        return LocalContextFusion(dim)
    return SumContextFusion(dim, gated,
        capture_gate=getattr(config, "gate_diagnostics", False), normalize_output=not compact)


def fusion_metadata(config, dim, *, compact):
    mode, gated = resolve_fusion(config)
    concat = (f"Linear({2*dim},{dim})->LayerNorm({dim})" if compact
              else f"Linear({2*dim},{2*dim})->GELU->Linear({2*dim},{dim})->LayerNorm({dim})->L2")
    return {
        "fusion_mode": mode, "use_gated_fusion": gated,
        "gate_diagnostics": bool(getattr(config, "gate_diagnostics", False)),
        "fusion": concat if mode == "concat" else (
            "L+sigmoid(MLP([L;C]))*C" if gated else "L+C") + f"->LayerNorm({dim})" + ("" if compact else "->L2"),
        "gate_mlp": f"Linear({2*dim},{dim})->GELU->Linear({dim},{dim})->Sigmoid" if gated else None,
        "gate_parameter_count": 3*dim*dim + 2*dim if gated else 0,
        "fusion_dimension_projections": "none; local and contextual dimensions already equal",
    }


def gate_statistics(alpha, valid):
    """Detached feature-wise population stats; exclude padding, use percent bins."""
    if alpha is None:
        return {}
    values = alpha.detach().float()[valid.bool()].reshape(-1)
    if values.numel() == 0:
        return {}
    bins = ((values < .25), ((values >= .25) & (values < .5)),
            ((values >= .5) & (values < .75)), (values >= .75))
    stats = torch.stack([values.mean(), values.std(unbiased=False), values.min(), values.max(),
                         *(b.float().mean() * 100 for b in bins)]).cpu().tolist()
    return dict(zip(("mean", "std", "min", "max", "pct_lt_025", "pct_025_050",
                     "pct_050_075", "pct_ge_075"), stats))


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
        return {"architecture_variant": LEGACY, **fusion_metadata(config, 192, compact=False)}
    return {
        "architecture_variant": VARIANT, "vector_size": 128, "vit_embed_dim": 128,
        "vit_variant": "fresh-bidirectional-transformer", "vit_layers": 5,
        "vit_heads": 1, "vit_mlp_dim": 512, "position_mode": "none",
        "architecture_revision": VARIANT, "context_transformer_layers": 5,
        "local_dropout": float(config.local_dropout), "input_dropout": 0.0,
        "fusion_dropout": 0.0, "local_projection": "Linear(512,128)->Dropout",
        "normalization_layout": "raw-local; transformer-preLN+finalLN; fused-LN-once->L2",
        **fusion_metadata(config, 128, compact=True),
        "initialization": "ImageNet-ResNet18; fresh-local/context/fusion",
        "tiny_vit_pretrained": False, "tiny_vit_pretrained_scope": "none",
        "context_patch_projection": "resnet18(512)->linear(128)->dropout",
        "transformer_position_prior": "none; DTW position prior is separate and unchanged",
    }

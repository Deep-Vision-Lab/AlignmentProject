"""Line -> explicit windows -> CNN -> Transformer -> selectable fusion -> unit vectors."""
import torch
from torch import nn
from torch.nn import functional as F

from cnn_encoder import CNNEncoder
from transformer_encoder import TransformerEncoder


def extract_windows(images, window_width=32, stride=16):
    """[B,C,H,W] -> [B,T,C,H,window_width], physical left-to-right order."""
    if images.ndim != 4 or window_width < 1 or stride < 1 or images.shape[-1] < window_width:
        raise ValueError('Require [B,C,H,W], positive stride/window and W >= window_width')
    return images.unfold(-1, window_width, stride).permute(0, 3, 1, 2, 4).contiguous()


def validate_fusion_config(fusion_mode='concat', use_gated_fusion=0):
    if fusion_mode not in ('concat', 'sum'):
        raise ValueError('fusion_mode must be concat or sum')
    if use_gated_fusion not in (0, 1, False, True):
        raise ValueError('use_gated_fusion must be 0 or 1')
    gated = bool(use_gated_fusion)
    if fusion_mode == 'concat' and gated:
        raise ValueError('Invalid fusion configuration: concat + gate ON is unsupported')
    return fusion_mode, gated


class SelectableFusion(nn.Module):
    def __init__(self, local_dim, context_dim, output_dim, mode='concat', use_gated_fusion=0):
        super().__init__()
        self.mode, self.use_gated_fusion = validate_fusion_config(mode, use_gated_fusion)
        self.local_projection = nn.Identity()
        self.context_projection = nn.Identity()
        self.concat_projection = None
        self.gate_mlp = None
        if self.mode == 'concat':
            self.concat_projection = nn.Linear(local_dim + context_dim, output_dim)
        else:
            if local_dim != output_dim:
                self.local_projection = nn.Linear(local_dim, output_dim)
            if context_dim != output_dim:
                self.context_projection = nn.Linear(context_dim, output_dim)
            if self.use_gated_fusion:
                self.gate_mlp = nn.Sequential(
                    nn.Linear(2 * output_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim),
                )

    def forward(self, local, context):
        if self.mode == 'concat':
            return self.concat_projection(torch.cat((local, context), dim=-1)), None
        local_projected = self.local_projection(local)
        context_projected = self.context_projection(context)
        if not self.use_gated_fusion:
            return local_projected + context_projected, None
        gate = torch.sigmoid(self.gate_mlp(torch.cat((local_projected, context_projected), dim=-1)))
        return local_projected + gate * context_projected, gate


def _gate_stats(gate):
    if gate is None:
        return None
    detached = gate.detach().float()
    return dict(mean=float(detached.mean()), std=float(detached.std(unbiased=False)),
                min=float(detached.min()), max=float(detached.max()))


class AlignmentModel(nn.Module):
    def __init__(self, cnn_type='resnet18', transformer_type='tiny', embedding_dim=128,
                 window_width=32, stride=16, pretrained_cnn=True, use_positional_encoding=False,
                 input_channels=1, local_dropout=.10, transformer_dropout=0., rtl=True,
                 fusion_mode='concat', use_gated_fusion=0, transformer_layers=0,
                 transformer_heads=0):
        super().__init__()
        self.window_width, self.stride, self.rtl = window_width, stride, rtl
        self.fusion_mode, self.use_gated_fusion = validate_fusion_config(fusion_mode, use_gated_fusion)
        self.cnn = CNNEncoder(cnn_type, embedding_dim, pretrained_cnn, input_channels, local_dropout)
        self.transformer = TransformerEncoder(transformer_type, embedding_dim,
                                               use_positional_encoding, transformer_dropout,
                                               transformer_layers, transformer_heads)
        self.fusion = SelectableFusion(embedding_dim, self.transformer.dim, embedding_dim,
                                       self.fusion_mode, self.use_gated_fusion)
        self.fusion_norm = nn.LayerNorm(embedding_dim)

    def forward(self, images, token_valid=None):
        windows = extract_windows(images, self.window_width, self.stride)
        b, t, c, h, w = windows.shape
        # One projection/dropout draw; the SAME local tensor feeds both branches.
        local = self.cnn(windows.reshape(b * t, c, h, w)).reshape(b, t, -1)
        valid = torch.ones((b, t), dtype=torch.bool, device=images.device) if token_valid is None else token_valid.bool().to(images.device)
        if valid.shape != (b, t):
            raise ValueError('token_valid must match physical [B,T] window grid')
        physical = torch.arange(t, device=images.device).expand(b, -1)
        if self.rtl:
            local, valid, physical = local.flip(1), valid.flip(1), physical.flip(1)
        context = self.transformer(local, valid)
        fused, gate = self.fusion(local, context)
        pre_l2 = self.fusion_norm(fused)
        fused = F.normalize(pre_l2.float(), dim=-1)
        return dict(local=local, context=context, fused=fused, fused_pre_l2=pre_l2,
                    token_valid=valid, physical_window_indices=physical,
                    fusion_gate=gate, fusion_gate_stats=_gate_stats(gate))

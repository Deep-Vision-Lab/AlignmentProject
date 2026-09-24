"""Line -> explicit windows -> CNN -> Transformer -> fusion -> unit vectors."""
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


class AlignmentModel(nn.Module):
    def __init__(self, cnn_type='resnet18', transformer_type='tiny', embedding_dim=128,
                 window_width=32, stride=16, pretrained_cnn=True, use_positional_encoding=False,
                 input_channels=1, local_dropout=.10, transformer_dropout=0., rtl=True):
        super().__init__()
        self.window_width, self.stride, self.rtl = window_width, stride, rtl
        self.cnn = CNNEncoder(cnn_type, embedding_dim, pretrained_cnn, input_channels, local_dropout)
        self.transformer = TransformerEncoder(transformer_type, embedding_dim,
                                               use_positional_encoding, transformer_dropout)
        self.fusion = nn.Linear(2 * embedding_dim, embedding_dim)
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
        pre_l2 = self.fusion_norm(self.fusion(torch.cat((local, context), dim=-1)))
        fused = F.normalize(pre_l2.float(), dim=-1)
        return dict(local=local, context=context, fused=fused, fused_pre_l2=pre_l2,
                    token_valid=valid, physical_window_indices=physical)

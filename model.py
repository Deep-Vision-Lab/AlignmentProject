"""Line -> explicit windows -> CNN -> Transformer -> selectable fusion -> unit vectors."""
import torch
from torch import nn
from torch.nn import functional as F

from cnn_encoder import CNNEncoder
from transformer_encoder import TransformerEncoder
from window_geometry import window_starts


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
                 transformer_heads=0, cnn_layers=3, variable_length_windows=False):
        super().__init__()
        self.window_width, self.stride, self.rtl = window_width, stride, rtl
        self.variable_length_windows = variable_length_windows
        self.fusion_mode, self.use_gated_fusion = validate_fusion_config(fusion_mode, use_gated_fusion)
        self.cnn = CNNEncoder(cnn_type, embedding_dim, pretrained_cnn, input_channels,
                              local_dropout, num_layers=cnn_layers)
        self.transformer = TransformerEncoder(transformer_type, embedding_dim,
                                               use_positional_encoding, transformer_dropout,
                                               transformer_layers, transformer_heads)
        self.fusion = SelectableFusion(embedding_dim, self.transformer.dim, embedding_dim,
                                       self.fusion_mode, self.use_gated_fusion)
        self.fusion_norm = nn.LayerNorm(embedding_dim)

    def forward(self, images, token_valid=None, valid_widths=None, space_confidence=None,
                window_tensors=None):
        if isinstance(images, (list, tuple)):
            if not self.variable_length_windows:
                raise ValueError('Ragged image batches require variable_length_windows')
            if not images or token_valid is not None or valid_widths is not None:
                raise ValueError('Ragged images carry their own exact valid-window lengths')
            starts = [window_starts(x.shape[-1], self.window_width, self.stride, shift_last=True)
                      for x in images]
            if any(x.ndim != 3 or x.shape[-2] != images[0].shape[-2] for x in images):
                raise ValueError('Ragged lines need [C,H,W] with a common encoder height')
            if window_tensors is not None:
                if len(window_tensors) != len(images) or any(
                        tuple(w.shape) != (len(row), images[0].shape[0], images[0].shape[1], self.window_width)
                        for w, row in zip(window_tensors, starts)):
                    raise ValueError('Pre-extracted windows disagree with image/window geometry')
                windows = torch.cat(window_tensors)
            else:
                windows = torch.cat([torch.stack([x[..., s:s+self.window_width] for s in row])
                                     for x, row in zip(images, starts)])
            # No artificial sequence enters the CNN, including its BatchNorm.
            encoded = self.cnn(windows)
            lengths = [len(row) for row in starts]
            locals_ = list(encoded.split(lengths))
            if self.rtl:
                locals_ = [x.flip(0) for x in locals_]
                starts = [list(reversed(row)) for row in starts]
            local = torch.nn.utils.rnn.pad_sequence(locals_, batch_first=True)
            valid = torch.arange(local.shape[1], device=local.device)[None] < torch.tensor(lengths, device=local.device)[:, None]
            physical_rows = [torch.arange(n-1, -1, -1, device=local.device) if self.rtl
                             else torch.arange(n, device=local.device) for n in lengths]
            physical = torch.nn.utils.rnn.pad_sequence(physical_rows, batch_first=True, padding_value=-1)
            starts_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(row, device=local.device) for row in starts], batch_first=True, padding_value=-1)
            if space_confidence is None:
                confidence = local.new_zeros(valid.shape)
            else:
                if len(space_confidence) != len(images):
                    raise ValueError('One space-confidence row is required per image')
                rows = [torch.as_tensor(row, device=local.device, dtype=local.dtype) for row in space_confidence]
                if any(len(row) != n for row, n in zip(rows, lengths)):
                    raise ValueError('Space-confidence length must match real window count')
                confidence = torch.nn.utils.rnn.pad_sequence(
                    [row.flip(0) if self.rtl else row for row in rows], batch_first=True)
            return self._finish(local, valid, physical, starts_tensor, confidence)
        windows = extract_windows(images, self.window_width, self.stride)
        b, t, c, h, w = windows.shape
        # One projection/dropout draw; the SAME local tensor feeds both branches.
        valid = torch.ones((b, t), dtype=torch.bool, device=images.device) if token_valid is None else token_valid.bool().to(images.device)
        if valid.shape != (b, t):
            raise ValueError('token_valid must match physical [B,T] window grid')
        if valid_widths is not None:
            widths = torch.as_tensor(valid_widths, device=images.device)
            if widths.shape != (b,) or (widths > images.shape[-1]).any() or (widths < self.window_width).any():
                raise ValueError('valid_widths must give an unpadded width >= window_width for each line')
            # Partial boundary windows are excluded, never filled with artificial pixels.
            valid = valid & (torch.arange(t, device=images.device)[None] * self.stride + self.window_width <= widths[:, None])
        if not valid.any(dim=1).all():
            raise ValueError('Each line requires at least one fully valid physical window')
        if valid.all():
            local = self.cnn(windows.reshape(b * t, c, h, w)).reshape(b, t, -1)
        else:
            encoded = self.cnn(windows[valid])  # padding cannot affect CNN BatchNorm statistics
            local = encoded.new_zeros(b, t, encoded.shape[-1])
            local[valid] = encoded
        physical = torch.arange(t, device=images.device).expand(b, -1)
        physical = physical.masked_fill(~valid, -1)
        if self.rtl:
            local, valid, physical = local.flip(1), valid.flip(1), physical.flip(1)
        starts_tensor = physical.clamp_min(0) * self.stride
        starts_tensor = starts_tensor.masked_fill(~valid, -1)
        confidence = local.new_zeros(valid.shape) if space_confidence is None else torch.as_tensor(
            space_confidence, device=local.device, dtype=local.dtype)
        if confidence.shape != valid.shape:
            raise ValueError('Space-confidence shape must match token validity')
        if self.rtl and space_confidence is not None:
            confidence = confidence.flip(1)
        return self._finish(local, valid, physical, starts_tensor, confidence)

    def _finish(self, local, valid, physical, starts_tensor, confidence):
        context = self.transformer(local, valid)
        fused, gate = self.fusion(local, context)
        pre_l2 = self.fusion_norm(fused)
        fused = F.normalize(pre_l2.float(), dim=-1)
        return dict(local=local, context=context, fused=fused, fused_pre_l2=pre_l2,
                    token_valid=valid, physical_window_indices=physical,
                    window_starts=starts_tensor, space_confidence=confidence,
                    fusion_gate=gate, fusion_gate_stats=_gate_stats(gate[valid] if gate is not None else None))

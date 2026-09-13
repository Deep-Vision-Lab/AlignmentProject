"""Shared components for the recommended restoration/alignment pipeline.

The active design deliberately separates three representations:
  local      : one vector from the real 128x32 window pixels;
  contextual : local sequence after positional Transformer context;
  fused      : normalized projection of concat(local, contextual), used by DTW/eval.

Restoration is decoded only from the local vector.  This prevents sequence
context from reconstructing a window on behalf of a collapsed local bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LocalContextFusion(nn.Module):
    """Concatenate local/context vectors and project back to the alignment dim."""

    def __init__(self, dim: int):
        super().__init__()
        dim = int(dim)
        self.projection = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, local: torch.Tensor, contextual: torch.Tensor) -> torch.Tensor:
        if local.shape != contextual.shape:
            raise ValueError(
                f"local/context shape mismatch: {tuple(local.shape)} != {tuple(contextual.shape)}"
            )
        fused = self.projection(torch.cat([local, contextual], dim=-1))
        return F.normalize(fused.float(), p=2, dim=-1).to(dtype=local.dtype)


def denormalize_imagenet_windows(patches: torch.Tensor) -> torch.Tensor:
    """Return the exact RGB window pixels in [0,1] from normalized input windows."""
    if patches.ndim != 5 or int(patches.shape[2]) != 3:
        raise ValueError(f"Expected [B,T,3,H,W], got {tuple(patches.shape)}")
    mean = patches.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = patches.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (patches.float() * std + mean).clamp(0.0, 1.0)


def _foreground_bbox_from_rgb(rgb: torch.Tensor, safety: int = 2):
    """Temporary foreground detector; RGB is never modified by this function."""
    gray = (
        0.2989 * rgb[0].float()
        + 0.5870 * rgb[1].float()
        + 0.1140 * rgb[2].float()
    )
    h, w = int(gray.shape[0]), int(gray.shape[1])
    border_h = max(1, int(round(h * 0.05)))
    border_w = max(1, int(round(w * 0.02)))
    border = torch.cat(
        [
            gray[:border_h, :].reshape(-1),
            gray[-border_h:, :].reshape(-1),
            gray[:, :border_w].reshape(-1),
            gray[:, -border_w:].reshape(-1),
        ]
    )
    background = border.median()
    distance = (gray - background).abs()
    # Robust threshold from the actual line. Keep it conservative so faint dots
    # and diacritics stay inside the detected content rectangle.
    threshold = torch.quantile(distance.reshape(-1), 0.75) * 0.35
    threshold = threshold.clamp_min(0.025)
    mask = distance >= threshold
    ys, xs = torch.where(mask)
    if xs.numel() < 4 or ys.numel() < 4:
        return (0, 0, w, h)
    x0 = max(0, int(xs.min()) - int(safety))
    x1 = min(w, int(xs.max()) + 1 + int(safety))
    y0 = max(0, int(ys.min()) - int(safety))
    y1 = min(h, int(ys.max()) + 1 + int(safety))
    return (x0, y0, x1, y1)


def line_padding_masks(
    normalized_line: torch.Tensor,
    *,
    window_size: int,
    stride: int,
    use_flip: bool = False,
):
    """Find only OUTER artificial padding while preserving internal blank gaps.

    Returns:
      token_valid : [B,T] windows overlapping the outer content rectangle.
      pixel_valid : [B,T,1,H,W] reconstruction pixels inside that rectangle.

    Internal spaces remain valid because the valid region is one rectangle from
    the first to last foreground extent, not an ink-per-window threshold.
    """
    if normalized_line.ndim != 4 or int(normalized_line.shape[1]) != 3:
        raise ValueError("Expected normalized RGB line [B,3,H,W]")
    mean = normalized_line.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = normalized_line.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = (normalized_line.float() * std + mean).clamp(0.0, 1.0)
    b, _, h, w = rgb.shape
    starts = list(range(0, w - int(window_size) + 1, int(stride)))
    token_masks = []
    pixel_masks = []
    for sample in rgb:
        x0, y0, x1, y1 = _foreground_bbox_from_rgb(sample)
        line_mask = torch.zeros((1, h, w), device=sample.device, dtype=torch.float32)
        line_mask[:, y0:y1, x0:x1] = 1.0
        windows = line_mask.unfold(2, int(window_size), int(stride))
        windows = windows.permute(2, 0, 1, 3).contiguous()
        token = []
        for start in starts:
            end = start + int(window_size)
            token.append(end > x0 and start < x1)
        token = torch.tensor(token, device=sample.device, dtype=torch.bool)
        if use_flip:
            windows = torch.flip(windows, dims=[0])
            token = torch.flip(token, dims=[0])
        token_masks.append(token)
        pixel_masks.append(windows)
    return torch.stack(token_masks, dim=0), torch.stack(pixel_masks, dim=0)


def masked_image_restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    pixel_weight: float,
    edge_weight: float,
    structure_weight: float,
):
    """RGB reconstruction loss with artificial-padding exclusion."""
    prediction = prediction.float()
    target = target.float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if valid_mask is None:
        mask = torch.ones_like(target[:, :, :1])
    else:
        if valid_mask.shape[:2] != target.shape[:2] or valid_mask.shape[-2:] != target.shape[-2:]:
            raise ValueError("valid reconstruction mask does not match target windows")
        mask = valid_mask.float()
    mask_rgb = mask.expand(-1, -1, target.shape[2], -1, -1)
    denom = mask_rgb.sum().clamp_min(1.0)
    pixel = ((prediction - target).abs() * mask_rgb).sum() / denom

    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    mask_dx = mask_rgb[..., :, 1:] * mask_rgb[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_dy = mask_rgb[..., 1:, :] * mask_rgb[..., :-1, :]
    edge_x = ((pred_dx - tgt_dx).abs() * mask_dx).sum() / mask_dx.sum().clamp_min(1.0)
    edge_y = ((pred_dy - tgt_dy).abs() * mask_dy).sum() / mask_dy.sum().clamp_min(1.0)
    edge = 0.5 * (edge_x + edge_y)

    pred_flat = (prediction * mask_rgb).flatten(start_dim=2)
    tgt_flat = (target * mask_rgb).flatten(start_dim=2)
    pred_flat = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
    tgt_flat = tgt_flat - tgt_flat.mean(dim=-1, keepdim=True)
    pred_unit = F.normalize(pred_flat, p=2, dim=-1, eps=1e-6)
    tgt_unit = F.normalize(tgt_flat, p=2, dim=-1, eps=1e-6)
    pred_similarity = pred_unit @ pred_unit.transpose(-1, -2)
    tgt_similarity = tgt_unit @ tgt_unit.transpose(-1, -2)
    count = int(prediction.shape[1])
    if count > 1:
        offdiag = ~torch.eye(count, dtype=torch.bool, device=prediction.device).unsqueeze(0)
        structure = (pred_similarity - tgt_similarity).abs().masked_select(
            offdiag.expand_as(pred_similarity)
        ).mean()
    else:
        structure = prediction.sum() * 0.0

    total = (
        float(pixel_weight) * pixel
        + float(edge_weight) * edge
        + float(structure_weight) * structure
    )
    zero = prediction.sum() * 0.0
    return total, pixel, edge, zero, structure


def contrastive_margin_from_costs(
    positive_cost: torch.Tensor,
    negative_costs: list[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    """Require positive DTW cost to be lower than every negative transcript cost."""
    if not negative_costs:
        return positive_cost.sum() * 0.0
    losses = [
        F.relu(float(margin) + positive_cost - negative)
        for negative in negative_costs
    ]
    return torch.stack(losses).mean()

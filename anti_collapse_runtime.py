"""Angular anti-collapse protection for ViT window embeddings.

The historical variance loss operated on raw feature magnitudes. That can miss
cosine collapse: vectors may have different lengths while pointing in almost the
same direction. This runtime patch measures variance after L2 normalization and,
for the Span-DTW path, also regularizes contextual window directions.
"""
from __future__ import annotations

from functools import wraps

import torch
import torch.nn.functional as F

import Parameters as P


def angular_embedding_variance_loss(
    img_emb: torch.Tensor,
    target_std: float = 0.05,
) -> torch.Tensor:
    """VICReg-style variance floor on unit-direction window embeddings."""
    if float(target_std) <= 0:
        return img_emb.new_tensor(0.0)

    z = img_emb.reshape(-1, img_emb.shape[-1]).float()
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)

    # Crucial difference from the previous loss: remove magnitude as an escape
    # route, so collinear windows with different norms still count as collapse.
    z = F.normalize(z, p=2, dim=-1)
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    return torch.relu(float(target_std) - std).mean()


def install(base_module) -> None:
    """Install angular local + contextual anti-collapse losses once."""
    if getattr(base_module, "_angular_anti_collapse_installed", False):
        return

    # Existing trainer calls this for the local representation in Span-DTW and
    # the contextual representation in legacy/non-span modes.
    base_module.image_embedding_variance_loss = angular_embedding_variance_loss

    original_compute = base_module.compute_single_image_text_loss

    @wraps(original_compute)
    def compute_single_image_text_loss(*args, **kwargs):
        loss, stats, cache = original_compute(*args, **kwargs)

        # Span-DTW returns normalized contextual features in cache[0]. The
        # original path already regularizes local features, so add the same
        # angular protection after the Transformer as well.
        if cache is not None:
            norm_context_img = cache[0]
            context_var = angular_embedding_variance_loss(
                norm_context_img,
                P.image_variance_target_std,
            )
            stats["img_context_var_loss"] = float(context_var.detach().item())

            if (
                float(P.image_variance_loss_weight) > 0
                and torch.is_grad_enabled()
            ):
                loss = (
                    loss
                    + float(P.image_variance_loss_weight) * context_var
                )
                stats["total"] = float(loss.detach().item())

        return loss, stats, cache

    base_module.compute_single_image_text_loss = compute_single_image_text_loss
    base_module._angular_anti_collapse_installed = True

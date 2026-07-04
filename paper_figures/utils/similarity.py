"""Compute embeddings and similarity matrices for paper figures."""
import os
import sys
import torch
import numpy as np

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)

from NormalizeFuncs import normalize_func


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_image_embeddings(model, image_tensor, device):
    """
    Run the image through EmbeddingModel and return L2-normalised embeddings.

    Args:
        model:        EmbeddingModel in eval mode.
        image_tensor: [C, H, W] or [B, C, H, W] float tensor.
        device:       torch.device.

    Returns:
        [seq_len, D] for a single image, or [B, seq_len, D] for a batch.
    """
    if image_tensor.dim() == 3:
        img = image_tensor.unsqueeze(0).to(device)
        emb = model(img)           # [1, seq_len, D]
        return normalize_func(emb.squeeze(0))  # [seq_len, D]
    img = image_tensor.to(device)
    emb = model(img)               # [B, seq_len, D]
    return normalize_func(emb)     # [B, seq_len, D]


@torch.no_grad()
def compute_text_embeddings(text_embedder, text):
    """
    Embed a transcript string with the frozen text embedder.

    Args:
        text_embedder: TextEmbedding / FastTextCharEmbedding in eval mode.
        text:          str.

    Returns:
        [text_len, D] L2-normalised tensor.
    """
    emb = text_embedder(text)   # [text_len, D]
    return normalize_func(emb)  # [text_len, D]


# ─────────────────────────────────────────────────────────────────────────────
# Similarity matrices
# ─────────────────────────────────────────────────────────────────────────────

def compute_text_image_similarity(text_emb, img_emb):
    """
    Cosine similarity matrix between text and image embeddings.

    Both inputs must already be L2-normalised.

    Args:
        text_emb: [T, D]
        img_emb:  [S, D]

    Returns:
        [T, S] cosine similarity matrix.
    """
    return torch.einsum("td,sd->ts", text_emb, img_emb)


def compute_image_image_similarity(img_emb_a, img_emb_b):
    """
    Cosine similarity matrix between two sets of image window embeddings.

    Both inputs must already be L2-normalised.

    Args:
        img_emb_a: [S_a, D]
        img_emb_b: [S_b, D]

    Returns:
        [S_a, S_b] cosine similarity matrix.
    """
    return torch.einsum("ad,bd->ab", img_emb_a, img_emb_b)


# ─────────────────────────────────────────────────────────────────────────────
# DTW cost (using the same criterion as training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dtw_cost(sim_matrix, device="cpu"):
    """
    Compute the (length-normalised) Soft-DTW cost on a [T, S] similarity matrix.

    Uses the same ContrastiveSoftDTW._compute_dtw_on_similarity as training so
    the cost numbers are directly comparable to training logs.

    Args:
        sim_matrix: Tensor [T, S] or ndarray [T, S].
        device:     Device for the DTW computation (CPU is fine for figures).

    Returns:
        Scalar Python float — the normalised cost.
    """
    from LossFunctionWithHelpers import ContrastiveSoftDTW
    from Parameters import (
        contrastive_soft_dtw_gamma, sakoe_chiba_bandwidth_ratio,
        contrastive_temperature, diagonal_prior_weight,
        diagonal_prior_sigma_ratio, dtw_band_penalty_weight,
    )

    if isinstance(sim_matrix, np.ndarray):
        sim_matrix = torch.from_numpy(sim_matrix).float()

    criterion = ContrastiveSoftDTW(
        gamma=contrastive_soft_dtw_gamma,
        use_cuda=False,
        bandwidth=sakoe_chiba_bandwidth_ratio,
        temperature=contrastive_temperature,
        diagonal_prior_weight=0.0,   # exclude diagonal prior from cost comparison
        diagonal_prior_sigma_ratio=diagonal_prior_sigma_ratio,
        dtw_band_penalty_weight=dtw_band_penalty_weight,
    )

    sm = sim_matrix.to(device)
    with torch.no_grad():
        cost = criterion._compute_dtw_on_similarity(sm.unsqueeze(0))[0]
        norm = max(sm.shape[0], sm.shape[1])
        return (cost / norm).item()

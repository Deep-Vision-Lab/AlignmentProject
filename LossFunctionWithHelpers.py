import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

# Import CUDA-accelerated Soft-DTW
# Note: The file soft-dtw-cuda.py needs to be renamed to soft_dtw_cuda.py for import
# Or use importlib to handle the hyphenated name
import importlib.util
import sys
import os
import soft_dtw_cuda as soft_dtw_cuda
from Parameters import *


# ============================================================================
# CONTRASTIVE SOFT-DTW LOSS (InfoNCE-style Contrastive Learning)
# Uses CUDA-accelerated Soft-DTW from soft_dtw_cuda.py
# ============================================================================

class ContrastiveSoftDTW(nn.Module):
    """
    Contrastive Soft-DTW Loss that takes pre-computed similarity matrices.
    
    This loss computes Soft-DTW alignment on the similarity matrices between
    image and text pairs, encouraging proper alignment structure (staircase pattern).
    
    Label-Aware Design:
        Since text embeddings are context-free (same letter = same embedding),
        repeated letters in text will have identical columns in the similarity matrix.
        Soft-DTW naturally handles this by finding optimal alignment paths that
        can match image patches to ANY of the matching text positions.
    
    Args:
        gamma (float): Soft-DTW smoothing parameter (gamma -> 0: hard DTW)ss
        use_cuda (bool): Use CUDA acceleration
        bandwidth (int): Sakoe-Chiba bandwidth for pruning
    """
    
    def __init__(self, gamma=0.1, use_cuda=True, bandwidth=0.0, margin=10.0,
                 temperature=0.07, diagonal_prior_weight=1.0,
                 diagonal_prior_sigma_ratio=0.15,
                 dtw_band_penalty_weight=2.0):
        super(ContrastiveSoftDTW, self).__init__()
        self.gamma = gamma
        self.use_cuda = use_cuda
        # NOTE: The CUDA kernel's Sakoe-Chiba band uses `abs(i - j) <= bw`, which
        # only makes sense for square matrices. Our matrices are rectangular
        # (N text rows ≈ 30, M image cols ≈ 100-200), so the natural diagonal
        # path runs along j ≈ i·M/N and its endpoint is FAR outside any
        # |i-j|<=bw band — the kernel marks it unreachable and returns inf.
        # We therefore keep `bandwidth` for backwards-compat but force it to 0
        # at the kernel and instead bake a SOFT band penalty into the distance
        # matrix (see _compute_dtw_on_similarity).
        self.bandwidth = bandwidth
        self.margin = margin
        # Temperature for softmax over cosine similarities.
        # Cosine sims live in [-1, 1]; without dividing by a small temperature
        # the softmax is nearly uniform across image patches and -log_softmax
        # saturates at log(seq_len), giving no gradient signal.
        self.temperature = temperature
        # Diagonal prior: a Gaussian-weighted bonus along the expected diagonal
        # stripe that directly rewards staircase-shaped alignments. The
        # contrastive margin alone is too weak — DTW can satisfy it with any
        # smeared similarity matrix. This term pulls similarity UP on-diagonal
        # and DOWN off-diagonal so the heatmap becomes sharp.
        self.diagonal_prior_weight = diagonal_prior_weight
        self.diagonal_prior_sigma_ratio = diagonal_prior_sigma_ratio
        # Soft diagonal band: a quadratic penalty added to the DTW distance
        # matrix that grows with squared distance from the expected diagonal
        # j ≈ i·(M-1)/(N-1). Replaces the broken kernel-level Sakoe-Chiba band
        # for rectangular matrices. 0 disables it.
        self.dtw_band_penalty_weight = dtw_band_penalty_weight

    def _soft_band_penalty(self, N: int, M: int, device, dtype) -> "torch.Tensor":
        """Quadratic distance-from-expected-diagonal penalty, shape [N, M].

        Zero on the diagonal j = i·(M-1)/(N-1); grows as (j_norm - i_norm)^2
        scaled by `dtw_band_penalty_weight`. Stays finite everywhere so no DTW
        cell becomes unreachable.
        """
        if self.dtw_band_penalty_weight <= 0.0:
            return torch.zeros(N, M, device=device, dtype=dtype)
        i_n = torch.arange(N, device=device, dtype=dtype) / max(N - 1, 1)
        j_n = torch.arange(M, device=device, dtype=dtype) / max(M - 1, 1)
        diff = j_n.unsqueeze(0) - i_n.unsqueeze(1)            # [N, M]
        return self.dtw_band_penalty_weight * diff * diff

    def _compute_dtw_on_similarity(self, sim_matrix):
        """
        Compute Soft-DTW directly on a pre-computed similarity matrix.
        Args:
            sim_matrix: [B, N, M] cosine similarity matrices, values in [-1, 1].
        Returns:
            DTW costs [B] as torch.Tensor
        """
        # 1. NEW: Column Centering
        # Compute the mean similarity score for each frame across all characters
        column_mean = sim_matrix.mean(dim=1, keepdim=True)  # Shape: [Batch, 1, M]
        sim_matrix_centered = sim_matrix - column_mean      # Zero-out global frame bias
        
        # 2. Compute distances using the centered matrix
        dist_matrix = -F.log_softmax(sim_matrix_centered / self.temperature, dim=-1)
        
        # 3. Apply your soft band penalty as normal
        N, M = sim_matrix.shape[1], sim_matrix.shape[2]
        dist_matrix = dist_matrix + self._soft_band_penalty(
            N, M, dist_matrix.device, dist_matrix.dtype
        )
        
        # 4. Pass to CUDA kernel
        _SoftDTW = soft_dtw_cuda._SoftDTW
        _SoftDTWCUDA = soft_dtw_cuda._SoftDTWCUDA

        if self.use_cuda and dist_matrix.is_cuda:
            dtw_costs = _SoftDTWCUDA.apply(dist_matrix, self.gamma, 0.0)
        else:
            dtw_costs = _SoftDTW.apply(dist_matrix, self.gamma, 0.0)

        # Cast to Tensor for type checker (apply() returns Tensor at runtime)
        assert isinstance(dtw_costs, torch.Tensor)
        return dtw_costs  # [B]

    def _diagonal_prior_loss(self, sim_pos: "torch.Tensor") -> "torch.Tensor":
        """
        Encourage a monotonic staircase in the positive similarity matrix.

        Builds a Gaussian "diagonal" weight W in [-1, 1]: +1 on the expected
        diagonal stripe j ≈ i * (M-1)/(N-1), smoothly transitioning to -1 far
        from it. Maximises <sim, W>, which simultaneously pulls similarity UP
        on-diagonal and DOWN off-diagonal, producing the staircase shape.
        """
        B, N, M = sim_pos.shape
        device = sim_pos.device
        i_idx = torch.arange(N, device=device, dtype=sim_pos.dtype)
        j_idx = torch.arange(M, device=device, dtype=sim_pos.dtype)
        slope = (M - 1) / max(N - 1, 1)
        expected_j = i_idx * slope                              # [N]
        dist = (j_idx.unsqueeze(0) - expected_j.unsqueeze(1)).abs()  # [N, M]
        sigma = max(1.0, self.diagonal_prior_sigma_ratio * max(N, M))
        gauss = torch.exp(-0.5 * (dist / sigma) ** 2)            # [N, M], in (0, 1]
        weight = 2.0 * gauss - 1.0                               # [N, M], in [-1, 1]
        return -(sim_pos * weight.unsqueeze(0)).mean()
    

    def forward(self, sim_pos, sim_neg_all):
        # sim_pos: [B, text_len, seq_len]
        # sim_neg_all: [B, K, neg_text_len, seq_len]
        
        B, K = sim_neg_all.shape[0], sim_neg_all.shape[1]
        
        # 1. Calculate positive DTW cost
        # Returns pos_cost: [B]
        pos_cost = self._compute_dtw_on_similarity(sim_pos) 
        
        # 2. Calculate DTW cost for ALL K negatives
        # We must reshape sim_neg_all to merge B and K so the DTW function can process them in parallel
        sim_neg_flat = sim_neg_all.view(B * K, sim_neg_all.shape[2], sim_neg_all.shape[3])
        
        # Returns neg_cost_flat: [B * K]
        neg_cost_flat = self._compute_dtw_on_similarity(sim_neg_flat)
        
        # Reshape back to [B, K]
        neg_costs = neg_cost_flat.view(B, K)
        
        # 3. SEQUENCE LENGTH NORMALIZATION
        seq_len = max(sim_pos.size(1), sim_pos.size(2))
        norm_pos_cost = pos_cost / seq_len          # [B]
        norm_neg_costs = neg_costs / seq_len         # [B, K]

        # 4. SUM-ALL MARGIN LOSS over all K negatives
        # For each sample, sum max(norm_pos - norm_neg_k + margin, 0) over all K
        raw_loss = norm_pos_cost.unsqueeze(1) - norm_neg_costs + self.margin  # [B, K]
        per_neg_loss = torch.clamp(raw_loss, min=0.0)                # [B, K]
        contrastive_loss = per_neg_loss.mean(dim=1).mean()                  # scalar

        # 5. DIAGONAL PRIOR on positive similarity — directly rewards staircase.
        diag_loss = self._diagonal_prior_loss(sim_pos)
        total_loss = contrastive_loss + self.diagonal_prior_weight * diag_loss

        avg_norm_gap = (norm_pos_cost.unsqueeze(1) - norm_neg_costs).mean().item()
        loss_dict = {
            'cost_pos': pos_cost.mean().item(),
            'cost_neg': neg_costs.mean().item(),
            'gap': avg_norm_gap,
            'norm_pos': norm_pos_cost.mean().item(),
            'norm_neg': norm_neg_costs.mean().item(),
            'contrastive': contrastive_loss.item(),
            'diag_prior': diag_loss.item(),
            # 'active_triplets': (per_neg_loss > 0).float().mean().item(),
        }
        return total_loss, loss_dict


# ============================================================================
# MULTI-SCALE CONTRASTIVE SOFT-DTW LOSS
# ============================================================================

class MultiScaleContrastiveSoftDTW(nn.Module):
    """
    Multi-Scale (Multi-Resolution) Contrastive Soft-DTW Loss.

    Computes Contrastive Soft-DTW at two sliding-window scales and combines
    them with a weighting factor alpha:

        Loss_total = Loss_norm(macro) + alpha * Loss_norm(micro)

    Both losses are independently sequence-length-normalised before
    combination so the larger micro sequence does not dominate the gradient.

    Args:
        gamma:     Soft-DTW smoothing parameter.
        use_cuda:  Use CUDA acceleration for DTW.
        bandwidth: Sakoe-Chiba bandwidth (0 = unconstrained).
        margin:    Contrastive margin for the triplet-style loss.
        alpha:     Weight for the micro (finer) scale loss.
    """

    def __init__(self, gamma=0.1, use_cuda=True, bandwidth=0.0,
                 margin=10.0, alpha=0.5, temperature=0.07,
                 diagonal_prior_weight=1.0, diagonal_prior_sigma_ratio=0.15,
                 dtw_band_penalty_weight=2.0):
        super(MultiScaleContrastiveSoftDTW, self).__init__()
        self.alpha = alpha
        # A single shared inner loss module (the gamma attribute is mutated
        # by the training loop for annealing, so one copy is fine).
        self.inner = ContrastiveSoftDTW(
            gamma=gamma,
            use_cuda=use_cuda,
            bandwidth=bandwidth,
            margin=margin,
            temperature=temperature,
            diagonal_prior_weight=diagonal_prior_weight,
            diagonal_prior_sigma_ratio=diagonal_prior_sigma_ratio,
            dtw_band_penalty_weight=dtw_band_penalty_weight,
        )

    # Expose gamma so the training-loop annealing code keeps working
    @property
    def gamma(self):
        return self.inner.gamma

    @gamma.setter
    def gamma(self, value):
        self.inner.gamma = value

    def forward(self,
                sim_pos_macro,    sim_neg_all_macro,
                sim_pos_micro,    sim_neg_all_micro):
        """
        Args:
            sim_pos_macro:     [B, text_len, seq_len_macro]
            sim_neg_all_macro: [B, K, neg_text_len, seq_len_macro]
            sim_pos_micro:     [B, text_len, seq_len_micro]
            sim_neg_all_micro: [B, K, neg_text_len, seq_len_micro]

        Returns:
            total_loss (scalar), loss_dict (dict)
        """
        loss_macro, dict_macro = self.inner(sim_pos_macro, sim_neg_all_macro)
        loss_micro, dict_micro = self.inner(sim_pos_micro, sim_neg_all_micro)

        total_loss = (1 - self.alpha) * loss_macro + self.alpha * loss_micro

        # Merge diagnostics from both scales
        loss_dict = {}
        for key, val in dict_macro.items():
            loss_dict[f'macro_{key}'] = val
        for key, val in dict_micro.items():
            loss_dict[f'micro_{key}'] = val
        loss_dict['loss_macro'] = loss_macro.item()
        loss_dict['loss_micro'] = loss_micro.item()

        return total_loss, loss_dict


def contrastive_soft_dtw_alignment_loss(sim_pos, sim_neg,
                                         gamma=0.1, use_cuda=True):
    """
    Functional interface for Contrastive Soft-DTW loss with multiple negatives.
    
    Args:
        sim_pos (torch.Tensor): Positive similarity matrix [B, N_txt, N_img]
        sim_neg (torch.Tensor): Negative similarity matrix [B, N_txt, N_img]
        gamma (float): Soft-DTW smoothing parameter
        use_cuda (bool): Use CUDA acceleration
    
    Returns:
        tuple: (total_loss, loss_dict)
    """
    
    criterion = ContrastiveSoftDTW(
        gamma=gamma,
        use_cuda=use_cuda
    )
    return criterion(sim_pos, sim_neg)


# ============================================================================
# LOSS FUNCTION FACTORY
# ============================================================================

def Loss_choice():
    if multi_scale_enabled:
        # Multi-Scale Contrastive Soft-DTW: two window sizes, one loss
        criterion = MultiScaleContrastiveSoftDTW(
            gamma=contrastive_soft_dtw_gamma,
            use_cuda=True,
            bandwidth=sakoe_chiba_bandwidth_ratio,
            margin=contrastive_margin,
            alpha=multi_scale_alpha,
            temperature=contrastive_temperature,
            diagonal_prior_weight=diagonal_prior_weight,
            diagonal_prior_sigma_ratio=diagonal_prior_sigma_ratio,
            dtw_band_penalty_weight=dtw_band_penalty_weight,
        )
    else:
        # Single-scale fallback
        criterion = ContrastiveSoftDTW(
            gamma=contrastive_soft_dtw_gamma,
            use_cuda=True,
            bandwidth=sakoe_chiba_bandwidth_ratio,
            margin=contrastive_margin,
            temperature=contrastive_temperature,
            diagonal_prior_weight=diagonal_prior_weight,
            diagonal_prior_sigma_ratio=diagonal_prior_sigma_ratio,
            dtw_band_penalty_weight=dtw_band_penalty_weight,
        )
    return criterion


if __name__ == "__main__":
    # Test the ContrastiveSoftDTW loss
    print("Testing ContrastiveSoftDTW loss with similarity matrices...")
    
    batch_size = 4
    N_txt = 15  # Text sequence length
    N_img = 20  # Image sequence length
    
    # Create dummy similarity matrices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_txt_sim1 = torch.randn(batch_size, N_txt, N_img, device=device, requires_grad=True)
    img_txt_sim2 = torch.randn(batch_size, N_txt, N_img, device=device, requires_grad=True)
    
    # Create dummy final prediction and target
    final_pred = torch.randn(batch_size, N_txt, N_txt, device=device, requires_grad=True)
    target = torch.randn(batch_size, N_txt, N_txt, device=device)
    
    # Test the loss with multiple negatives
    criterion = ContrastiveSoftDTW(gamma=0.001, use_cuda=torch.cuda.is_available())
    
    # Create positive similarity matrix
    sim_pos = torch.randn(batch_size, N_txt, N_img, device=device, requires_grad=True)
    
    # Create lists of negative similarity matrices (10 negatives)
    num_neg = 10
    neg_sim_img_list = [torch.randn(batch_size, N_txt, N_img, device=device, requires_grad=True) for _ in range(num_neg)]
    neg_sim_txt_list = [torch.randn(batch_size, N_txt, N_img, device=device, requires_grad=True) for _ in range(num_neg)]
    
    loss, loss_dict = criterion(sim_pos, neg_sim_img_list, neg_sim_txt_list)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test backward pass
    loss.backward()
    print(f"Gradient computed successfully.")
    if sim_pos.grad is not None:
        print(f"  sim_pos grad norm: {sim_pos.grad.norm().item():.4f}")
    if neg_sim_img_list[0].grad is not None:
        print(f"  neg_sim_img_list[0] grad norm: {neg_sim_img_list[0].grad.norm().item():.4f}")

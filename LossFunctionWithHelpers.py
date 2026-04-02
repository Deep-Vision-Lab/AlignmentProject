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
    
    def __init__(self, gamma=0.1, use_cuda=True, bandwidth=0.0, margin=10.0):
        super(ContrastiveSoftDTW, self).__init__()
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.bandwidth = bandwidth
        self.margin = margin
    

    def _compute_dtw_on_similarity(self, sim_matrix: "torch.Tensor") -> "torch.Tensor":
        """
        Compute Soft-DTW directly on a pre-computed similarity matrix.
        
        The SoftDTW expects distance/cost matrices (lower = more similar).
        We convert similarity to negative log-likelihood:
            p(img_m | text_n) = softmax(sim[b, n, :], dim=-1)
            dist = -log(p) = -log_softmax(sim, dim=-1)
        
        This gives a proper probabilistic cost: aligned pairs have low NLL,
        misaligned pairs have high NLL.
        
        Args:
            sim_matrix: [B, N, M] similarity matrices (higher = more similar, typically in [-1, 1])
        
        Returns:
            DTW costs [B] as torch.Tensor
        """
        # Convert similarity to negative log-likelihood
        # log_softmax is numerically stable: -log(softmax(x)) = -x + logsumexp(x)
        # sim_matrix: [B, N, M] -> dist_matrix: [B, N, M]
        dist_matrix = -F.log_softmax(sim_matrix, dim=-1)
        # Use the SoftDTW _SoftDTW function directly with pre-computed distances
        # The _SoftDTW.apply expects [B, N, M] distance matrix
        _SoftDTW = soft_dtw_cuda._SoftDTW
        _SoftDTWCUDA = soft_dtw_cuda._SoftDTWCUDA

        if self.use_cuda and dist_matrix.is_cuda:
            dtw_costs = _SoftDTWCUDA.apply(dist_matrix, self.gamma, self.bandwidth)
        else:
            dtw_costs = _SoftDTW.apply(dist_matrix, self.gamma, self.bandwidth)
        
        # Cast to Tensor for type checker (apply() returns Tensor at runtime)
        assert isinstance(dtw_costs, torch.Tensor)
        return dtw_costs  # [B]
    

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
        per_neg_loss = torch.clamp(raw_loss, min=0.0)           # [B, K]
        loss = per_neg_loss.mean(dim=1)                           # [B]
        
        avg_norm_gap = (norm_pos_cost.unsqueeze(1) - norm_neg_costs).mean().item()
        loss_dict = {
            'cost_pos': pos_cost.mean().item(),
            'cost_neg': neg_costs.mean().item(),
            'gap': avg_norm_gap,
            'norm_pos': norm_pos_cost.mean().item(),
            'norm_neg': norm_neg_costs.mean().item(),
            'active_triplets': (per_neg_loss > 0).float().mean().item(),
        }
        return loss.mean(), loss_dict


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
                 margin=10.0, alpha=0.5):
        super(MultiScaleContrastiveSoftDTW, self).__init__()
        self.alpha = alpha
        # A single shared inner loss module (the gamma attribute is mutated
        # by the training loop for annealing, so one copy is fine).
        self.inner = ContrastiveSoftDTW(
            gamma=gamma,
            use_cuda=use_cuda,
            bandwidth=bandwidth,
            margin=margin,
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

        total_loss = loss_macro + self.alpha * loss_micro

        # Merge diagnostics from both scales
        loss_dict = {}
        for key, val in dict_macro.items():
            loss_dict[f'macro_{key}'] = val
        for key, val in dict_micro.items():
            loss_dict[f'micro_{key}'] = val
        loss_dict['loss_macro'] = loss_macro.item()
        loss_dict['loss_micro'] = loss_micro.item()
        loss_dict['alpha'] = self.alpha

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
        )
    else:
        # Single-scale fallback
        criterion = ContrastiveSoftDTW(
            gamma=contrastive_soft_dtw_gamma,
            use_cuda=True,
            bandwidth=sakoe_chiba_bandwidth_ratio,
            margin=contrastive_margin
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

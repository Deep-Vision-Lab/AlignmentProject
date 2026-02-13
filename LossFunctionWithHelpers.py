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
        gamma (float): Soft-DTW smoothing parameter (gamma -> 0: hard DTW)
        temperature (float): Temperature for contrastive loss
        use_cuda (bool): Use CUDA acceleration
        bandwidth (int): Sakoe-Chiba bandwidth for pruning
    """
    
    def __init__(self, gamma=0.1, temperature=0.1, use_cuda=True, bandwidth=None):
        super(ContrastiveSoftDTW, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.use_cuda = use_cuda
        self.bandwidth = bandwidth
        
        # Initialize the CUDA-accelerated Soft-DTW function
        # IMPORTANT: normalize=False because image and text sequences have different lengths
        self.dtw_func = soft_dtw_cuda.SoftDTW(
            use_cuda=use_cuda, 
            gamma=gamma, 
            normalize=False, 
            bandwidth=bandwidth,
            dist_func=soft_dtw_cuda.SoftDTW._cosine_dist_func
        )
    
    def _compute_dtw_on_similarity(self, sim_matrix: "torch.Tensor") -> "torch.Tensor":
        """
        Compute Soft-DTW directly on a pre-computed similarity matrix.
        
        The SoftDTW expects distance matrices (lower = more similar).
        We convert similarity to distance: dist = 1 - sim
        This ensures non-negative distances (0 = perfect match, 2 = worst match for cosine sim).
        
        Args:
            sim_matrix: [B, N, M] similarity matrices (higher = more similar, typically in [-1, 1])
        
        Returns:
            DTW costs [B] as torch.Tensor
        """
        # Convert similarity to distance (1 - sim)
        # For cosine similarity in [-1, 1]: distance will be in [0, 2]
        # sim_matrix: [B, N, M] -> dist_matrix: [B, N, M]
        dist_matrix = 1.0 - sim_matrix
        
        # Use the SoftDTW _SoftDTW function directly with pre-computed distances
        # The _SoftDTW.apply expects [B, N, M] distance matrix
        _SoftDTW = soft_dtw_cuda._SoftDTW
        _SoftDTWCUDA = soft_dtw_cuda._SoftDTWCUDA
        
        if self.use_cuda and dist_matrix.is_cuda:
            dtw_costs = _SoftDTWCUDA.apply(dist_matrix, self.gamma, self.bandwidth if self.bandwidth else 0)
        else:
            dtw_costs = _SoftDTW.apply(dist_matrix, self.gamma, self.bandwidth if self.bandwidth else 0)
        
        # Cast to Tensor for type checker (apply() returns Tensor at runtime)
        assert isinstance(dtw_costs, torch.Tensor)
        return dtw_costs  # [B]
    

    def forward(self, img_txt_sim1, img_txt_sim2, 
            text1_image2_sim, text2_image1_sim): # <--- You need these mismatched pairs
        
        # 1. POSITIVE PAIRS (We want LOW cost / HIGH similarity)
        # Remember to use the "Cost = -Similarity" fix we discussed!
        cost_pos_1 = self._compute_dtw_on_similarity(img_txt_sim1) 
        cost_pos_2 = self._compute_dtw_on_similarity(img_txt_sim2)
        
        # 2. NEGATIVE PAIRS (We want HIGH cost / LOW similarity)
        cost_neg_1 = self._compute_dtw_on_similarity(text2_image1_sim)
        cost_neg_2 = self._compute_dtw_on_similarity(text1_image2_sim)
        
        # 3. CONTRASTIVE LOSS (Triplet-style or InfoNCE)
        # "Positive cost should be lower than Negative cost by at least 'margin'"
        margin = 1.0 
        
        loss_1 = F.relu(cost_pos_1 - cost_neg_1 + margin)
        loss_2 = F.relu(cost_pos_2 - cost_neg_2 + margin)
        
        total_loss = loss_1.sum() + loss_2.sum()
        
        loss_dict = {
            'cost_pos_1': cost_pos_1.sum().item(),
            'cost_pos_2': cost_pos_2.sum().item(),
            'cost_neg_1': cost_neg_1.sum().item(),
            'cost_neg_2': cost_neg_2.sum().item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def contrastive_soft_dtw_alignment_loss(img_txt_sim1, img_txt_sim2, 
                                         final_pred=None, target=None,
                                         gamma=0.1, temperature=0.1,
                                         mse_weight=1.0, dtw_weight=0.5,
                                         use_cuda=True):
    """
    Functional interface for Contrastive Soft-DTW loss.
    
    This combines MSE reconstruction loss with Soft-DTW alignment loss
    using CUDA-accelerated Soft-DTW.
    
    Args:
        img_txt_sim1 (torch.Tensor): Similarity matrix between text1 and image1 [B, N_txt, N_img]
        img_txt_sim2 (torch.Tensor): Similarity matrix between text2 and image2 [B, M_txt, M_img]
        final_pred (torch.Tensor): Final predicted similarity matrix [B, H, W]
        target (torch.Tensor): Ground truth similarity matrix [B, H, W]
        gamma (float): Soft-DTW smoothing parameter
        temperature (float): Temperature (not used in current implementation)
        mse_weight (float): Weight for MSE loss
        dtw_weight (float): Weight for DTW alignment loss
        use_cuda (bool): Use CUDA acceleration
    
    Returns:
        tuple: (total_loss, loss_dict)
    """
    criterion = ContrastiveSoftDTW(
        gamma=gamma, 
        temperature=temperature, 
        use_cuda=use_cuda
    )
    return criterion(img_txt_sim1, img_txt_sim2, final_pred, target, 
                    mse_weight=mse_weight, dtw_weight=dtw_weight)


# ============================================================================
# LOSS FUNCTION FACTORY
# ============================================================================

def Loss_choice(loss_type):
    """
    Factory function to create the appropriate loss function.
    
    Args:
        loss_type (str): Type of loss function to create
        
    Returns:
        Loss function (criterion)
    """
    if loss_type == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    elif loss_type == 'ContrastiveSoftDTW':
        # Contrastive Soft-DTW: takes similarity matrices directly
        # Computes DTW on both similarity matrices + MSE on final prediction
        from Parameters import contrastive_soft_dtw_gamma, contrastive_soft_dtw_temperature
        criterion = ContrastiveSoftDTW(
            gamma=contrastive_soft_dtw_gamma,
            temperature=contrastive_soft_dtw_temperature,
            use_cuda=True
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: ['MSE', 'ContrastiveSoftDTW']")
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
    
    # Test the loss
    criterion = ContrastiveSoftDTW(gamma=0.1, temperature=0.1, use_cuda=torch.cuda.is_available())
    loss, loss_dict = criterion(img_txt_sim1, img_txt_sim2, final_pred, target)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test backward pass
    loss.backward()
    print(f"Gradient computed successfully.")
    if img_txt_sim1.grad is not None:
        print(f"  img_txt_sim1 grad norm: {img_txt_sim1.grad.norm().item():.4f}")
    if img_txt_sim2.grad is not None:
        print(f"  img_txt_sim2 grad norm: {img_txt_sim2.grad.norm().item():.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from Parameters import *

# Import CUDA-accelerated Soft-DTW
from soft_dtw_cuda import SoftDTW


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
        normalize (bool): Normalize DTW scores
        bandwidth (int): Sakoe-Chiba bandwidth for pruning
    """
    
    def __init__(self, gamma=0.1, temperature=0.1, use_cuda=True, normalize=True, bandwidth=None):
        super(ContrastiveSoftDTW, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.use_cuda = use_cuda
        self.normalize = normalize
        self.bandwidth = bandwidth
        
        # Initialize the CUDA-accelerated Soft-DTW function
        self.dtw_func = SoftDTW(use_cuda=use_cuda, gamma=gamma, normalize=normalize, bandwidth=bandwidth)
    
    def _compute_dtw_on_similarity(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute Soft-DTW directly on a pre-computed similarity matrix.
        
        The SoftDTW expects distance matrices (lower = more similar).
        We convert similarity to distance by negating: dist = -sim
        
        Args:
            sim_matrix: [B, N, M] similarity matrices (higher = more similar)
        
        Returns:
            DTW costs [B] as torch.Tensor
        """
        # Convert similarity to distance (negate)
        # sim_matrix: [B, N, M] -> dist_matrix: [B, N, M]
        dist_matrix = -sim_matrix
        
        # Use the SoftDTW _SoftDTW function directly with pre-computed distances
        # The _SoftDTW.apply expects [B, N, M] distance matrix
        from soft_dtw_cuda import _SoftDTW, _SoftDTWCUDA
        
        if self.use_cuda and dist_matrix.is_cuda:
            dtw_costs = _SoftDTWCUDA.apply(dist_matrix, self.gamma, self.bandwidth if self.bandwidth else 0)
        else:
            dtw_costs = _SoftDTW.apply(dist_matrix, self.gamma, self.bandwidth if self.bandwidth else 0)
        
        # Cast to Tensor for type checker (apply() returns Tensor at runtime)
        assert isinstance(dtw_costs, torch.Tensor)
        return dtw_costs  # [B]
    
    def forward(self, img_txt_sim1, img_txt_sim2, final_pred=None, target=None, 
                mse_weight=1.0, dtw_weight=0.5):
        """
        Compute combined loss using pre-computed similarity matrices.
        
        Args:
            img_txt_sim1 (torch.Tensor): Similarity matrix between text1 and image1 [B, N_txt, N_img]
            img_txt_sim2 (torch.Tensor): Similarity matrix between text2 and image2 [B, M_txt, M_img]
            final_pred (torch.Tensor, optional): Final predicted similarity matrix for MSE [B, H, W]
            target (torch.Tensor, optional): Ground truth similarity matrix for MSE [B, H, W]
            mse_weight (float): Weight for MSE reconstruction loss
            dtw_weight (float): Weight for DTW alignment loss
        
        Returns:
            tuple: (total_loss, loss_dict) with individual loss components
        """
        batch_size = img_txt_sim1.size(0)
        device = img_txt_sim1.device
        
        # ====================================================================
        # Step 1: Compute Soft-DTW costs on both similarity matrices
        # ====================================================================
        # DTW on img_txt_sim1: encourages proper alignment between text1 and image1
        dtw_costs1 = self._compute_dtw_on_similarity(img_txt_sim1)  # [B]
        
        # DTW on img_txt_sim2: encourages proper alignment between text2 and image2
        dtw_costs2 = self._compute_dtw_on_similarity(img_txt_sim2)  # [B]
        
        # Average DTW costs across both pairs
        dtw_loss1 = dtw_costs1.mean()
        dtw_loss2 = dtw_costs2.mean()
        dtw_loss = (dtw_loss1 + dtw_loss2) / 2.0
        
        # ====================================================================
        # Step 2: Compute alignment quality metrics for monitoring
        # ====================================================================
        # Use mean of max similarities as proxy for alignment quality
        with torch.no_grad():
            # For sim1: average max similarity per text position
            max_sim1_per_txt = img_txt_sim1.max(dim=2)[0]  # [B, N_txt]
            alignment_score1 = max_sim1_per_txt.mean()
            
            # For sim2: average max similarity per text position
            max_sim2_per_txt = img_txt_sim2.max(dim=2)[0]  # [B, M_txt]
            alignment_score2 = max_sim2_per_txt.mean()
            
            avg_alignment_score = (alignment_score1 + alignment_score2) / 2.0
        
        # ====================================================================
        # Step 3: MSE loss on final similarity matrix
        # ====================================================================
        if final_pred is not None and target is not None:
            mse_loss = F.mse_loss(final_pred, target)
        else:
            mse_loss = torch.tensor(0.0, device=device)
        
        # ====================================================================
        # Combined Loss: MSE + DTW
        # ====================================================================
        total_loss = mse_weight * mse_loss + dtw_weight * dtw_loss
        
        loss_dict = {
            'mse': mse_loss.item() if torch.is_tensor(mse_loss) else mse_loss,
            'dtw': dtw_loss.item(),
            'dtw1': dtw_loss1.item(),
            'dtw2': dtw_loss2.item(),
            'alignment_score': avg_alignment_score.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def contrastive_soft_dtw_alignment_loss(img_txt_sim1, img_txt_sim2, 
                                         final_pred=None, target=None,
                                         gamma=0.1, temperature=0.1,
                                         mse_weight=1.0, dtw_weight=0.5,
                                         use_cuda=True, normalize=True):
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
        normalize (bool): Normalize DTW scores
    
    Returns:
        tuple: (total_loss, loss_dict)
    """
    criterion = ContrastiveSoftDTW(
        gamma=gamma, 
        temperature=temperature, 
        use_cuda=use_cuda,
        normalize=normalize
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
        criterion = ContrastiveSoftDTW(
            gamma=contrastive_soft_dtw_gamma,
            temperature=contrastive_soft_dtw_temperature,
            use_cuda=True,
            normalize=False  # Don't normalize - sequences have different lengths
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

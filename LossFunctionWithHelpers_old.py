import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

# Import CUDA-accelerated Soft-DTW
from soft_dtw_cuda import SoftDTW


# ============================================================================
# CONTRASTIVE SOFT-DTW LOSS (InfoNCE-style Contrastive Learning)
# Uses CUDA-accelerated Soft-DTW from soft_dtw_cuda.py
# ============================================================================

class ContrastiveSoftDTW(nn.Module):
    """
    Contrastive Soft-DTW Loss with InfoNCE-style contrastive learning.
    
    This loss function combines Soft-DTW alignment with contrastive learning in a single
    unified loss. It computes ALL pairwise alignments within a batch and uses InfoNCE
    (cross-entropy) to enforce that:
    - Correct image-text pairs (diagonal) have low DTW cost (high similarity)
    - Incorrect image-text pairs (off-diagonal) have high DTW cost (low similarity)
    
    The contrastive learning is built-in, so no special handling is needed in the
    training loop. Just pass image and text embeddings, and it handles everything.
    
    Uses CUDA-accelerated Soft-DTW from soft_dtw_cuda.py for efficient computation.
    
    Mathematical formulation:
    1. Compute pairwise similarity matrices: S[i,j] = Image[i] @ Text[j].T for all i,j
    2. Compute Soft-DTW energy for each pair: E[i,j] = -SoftDTW(-S[i,j])
    3. Apply InfoNCE loss: L = CrossEntropy(E / temperature, labels)
       where labels = [0, 1, 2, ..., B-1] (diagonal indices)
    
    Args:
        gamma (float): Soft-DTW smoothing parameter (temperature for DTW)
                       gamma -> 0: hard DTW, gamma -> inf: average pooling
        temperature (float): Temperature for contrastive loss (InfoNCE temperature)
                            Lower = sharper distinctions between pairs
        use_cuda (bool): Whether to use CUDA acceleration for Soft-DTW
        normalize (bool): Whether to use normalized Soft-DTW (divergence form)
        bandwidth (int): Sakoe-Chiba bandwidth for pruning (None for global alignment)
    """
    
    def __init__(self, gamma=0.1, temperature=0.1, use_cuda=True, normalize=True, bandwidth=None):
        super(ContrastiveSoftDTW, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.use_cuda = use_cuda
        self.normalize = normalize
        
        # Initialize the CUDA-accelerated Soft-DTW function
        # Note: We don't use the built-in dist_func since we compute similarity matrices ourselves
        self.dtw_func = SoftDTW(use_cuda=use_cuda, gamma=gamma, normalize=normalize, bandwidth=bandwidth)
    
    def forward(self, image_embeddings, text_embeddings, final_pred=None, target=None, 
                mse_weight=1.0, contrastive_weight=0.5):
        """
        Compute Contrastive Soft-DTW loss.
        
        Args:
            image_embeddings (torch.Tensor): Image patch embeddings [Batch, N, Dim]
            text_embeddings (torch.Tensor): Text token embeddings [Batch, M, Dim]
            final_pred (torch.Tensor, optional): Final similarity matrix for MSE [B, H, W]
            target (torch.Tensor, optional): Ground truth for MSE [B, H, W]
            mse_weight (float): Weight for MSE reconstruction loss
            contrastive_weight (float): Weight for contrastive DTW loss
        
        Returns:
            tuple: (total_loss, loss_dict) with individual loss components
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device
        N = image_embeddings.size(1)  # Image sequence length
        M = text_embeddings.size(1)   # Text sequence length
        D = image_embeddings.size(2)  # Embedding dimension
        
        # ====================================================================
        # Step 1: Compute ALL pairwise similarity matrices at once
        # ====================================================================
        # Broadcast: (Batch, 1, N, D) @ (1, Batch, D, M) -> (Batch, Batch, N, M)
        # This creates a matrix of Image[i] vs Text[j] for ALL i, j pairs
        
        img_expanded = image_embeddings.unsqueeze(1)      # [B, 1, N, D]
        txt_expanded = text_embeddings.unsqueeze(0)       # [1, B, M, D]
        txt_expanded = txt_expanded.transpose(-1, -2)     # [1, B, D, M]
        
        # Giant Matrix Multiplication: all_sim[i,j] = sim(Image_i, Text_j)
        all_sim_matrices = torch.matmul(img_expanded, txt_expanded)  # [B, B, N, M]
        
        # ====================================================================
        # Step 2: Compute Soft-DTW costs for ALL pairs using CUDA-accelerated DTW
        # ====================================================================
        # We need to compute DTW for each (i, j) pair
        # The SoftDTW expects sequences of shape [batch, seq_len, dims]
        # We'll iterate over pairs or reshape cleverly
        
        # Flatten to process all B*B pairs
        # Reshape similarity matrices to be "sequences" for DTW
        # all_sim_matrices: [B, B, N, M] -> we need to compute DTW on each [N, M] matrix
        
        # For DTW, we treat the similarity matrix as distance by negating
        # Then compute DTW cost for each pair
        
        dtw_costs = []
        for i in range(batch_size):
            for j in range(batch_size):
                # Get similarity matrix for pair (i, j): [N, M]
                sim_matrix = all_sim_matrices[i, j]  # [N, M]
                
                # Convert to distance matrix (negate similarity)
                # Reshape for DTW: treat as sequences where each row/col is a "timestep"
                # We'll use the similarity matrix directly as pre-computed distances
                
                # Create dummy sequences that give us the distance matrix we want
                # The SoftDTW.dist_func computes ||x_i - y_j||^2
                # But we want to use our pre-computed similarity as distance
                
                # Trick: Pass the similarity as negative distance directly
                # We create sequences where the pairwise distance IS our cost
                # Actually, let's use the underlying function directly
                
                # The SoftDTW expects X, Y as [batch, seq, dim]
                # We'll pass the image and text embeddings directly for this pair
                img_seq = image_embeddings[i:i+1]  # [1, N, D]
                txt_seq = text_embeddings[j:j+1]   # [1, M, D]
                
                # Compute DTW (returns distance, lower = more similar)
                cost = self.dtw_func(img_seq, txt_seq)  # [1]
                dtw_costs.append(cost)
        
        # Stack all costs: [B*B]
        dtw_costs = torch.cat(dtw_costs, dim=0)
        
        # Reshape to [B, B] energy matrix
        # energy_matrix[i, j] = -DTW_cost(Image_i, Text_j)
        # Negate because lower DTW cost = better match = higher "energy"
        energy_matrix = -dtw_costs.view(batch_size, batch_size)
        
        # ====================================================================
        # Step 3: InfoNCE Contrastive Loss
        # ====================================================================
        # We want diagonal (correct pairs) to have highest energy
        # Labels are simply [0, 1, 2, ...] - the diagonal indices
        labels = torch.arange(batch_size, device=device)
        
        # Scale by temperature
        logits = energy_matrix / self.temperature
        
        # Cross-entropy loss (automatically does log_softmax)
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # ====================================================================
        # Step 4: Optional MSE loss on final similarity matrix
        # ====================================================================
        if final_pred is not None and target is not None:
            mse_loss = F.mse_loss(final_pred, target)
        else:
            mse_loss = torch.tensor(0.0, device=device)
        
        # ====================================================================
        # Combined Loss
        # ====================================================================
        total_loss = mse_weight * mse_loss + contrastive_weight * contrastive_loss
        
        # Compute accuracy for monitoring (how often correct pair has highest energy)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        loss_dict = {
            'mse': mse_loss.item() if torch.is_tensor(mse_loss) else mse_loss,
            'contrastive_dtw': contrastive_loss.item(),
            'contrastive_accuracy': accuracy,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


class ContrastiveSoftDTWFast(nn.Module):
    """
    Fast Contrastive Soft-DTW Loss - Batched computation for efficiency.
    
    This version computes DTW in a more efficient batched manner by
    leveraging the CUDA kernel's ability to process multiple sequences at once.
    
    Uses a custom distance function to pass pre-computed similarity matrices.
    """
    
    def __init__(self, gamma=0.1, temperature=0.1, use_cuda=True, normalize=True, bandwidth=None):
        super(ContrastiveSoftDTWFast, self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.use_cuda = use_cuda
        self.normalize = normalize
        self.bandwidth = bandwidth
        
        # We'll create DTW func on-the-fly with custom distance
    
    def _identity_dist(self, x, y):
        """
        Identity distance function - assumes x already contains the distance/cost matrix.
        x: [B, N, M] - pre-computed distance matrices
        y: [B, N, M] - dummy, same as x
        Returns: x (the pre-computed distances)
        """
        return x
    
    def forward(self, image_embeddings, text_embeddings, final_pred=None, target=None, 
                mse_weight=1.0, contrastive_weight=0.5):
        """
        Compute Contrastive Soft-DTW loss with batched DTW computation.
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device
        N = image_embeddings.size(1)
        M = text_embeddings.size(1)
        D = image_embeddings.size(2)
        
        # ====================================================================
        # Step 1: Compute ALL pairwise distance matrices
        # ====================================================================
        # Using Euclidean distance: ||img_i - txt_j||^2
        # Expand for broadcasting: [B, 1, N, 1, D] vs [1, B, 1, M, D]
        
        img_exp = image_embeddings.view(batch_size, 1, N, 1, D)
        txt_exp = text_embeddings.view(1, batch_size, 1, M, D)
        
        # Pairwise squared Euclidean distance: [B, B, N, M]
        all_dist_matrices = torch.sum((img_exp - txt_exp) ** 2, dim=-1)  # [B, B, N, M]
        
        # ====================================================================
        # Step 2: Compute Soft-DTW for all pairs
        # ====================================================================
        # Flatten to [B*B, N, M] and treat as batch
        flat_dists = all_dist_matrices.view(batch_size * batch_size, N, M)
        
        # Create SoftDTW with identity distance (we already have distances)
        dtw_func = SoftDTW(
            use_cuda=self.use_cuda, 
            gamma=self.gamma, 
            normalize=self.normalize,
            bandwidth=self.bandwidth,
            dist_func=lambda x, y: x  # Identity - x is already the distance matrix
        )
        
        # We need to reshape for DTW input format
        # DTW expects [batch, seq_len, dims] for X and Y
        # We'll create dummy sequences and use custom dist_func
        
        # Create dummy tensors with the right shape
        # The distance function will receive these and should return our pre-computed distances
        dummy_x = flat_dists.unsqueeze(-1)  # [B*B, N, 1] - dummy dim
        dummy_y = torch.zeros(batch_size * batch_size, M, 1, device=device)  # [B*B, M, 1]
        
        # Custom distance function that returns our pre-computed matrix
        def custom_dist(x, y):
            # x: [B*B, N, 1], y: [B*B, M, 1]
            # We want to return flat_dists: [B*B, N, M]
            return flat_dists
        
        # Create a new DTW with our custom distance
        dtw_custom = SoftDTW(
            use_cuda=self.use_cuda,
            gamma=self.gamma,
            normalize=self.normalize,
            bandwidth=self.bandwidth,
            dist_func=custom_dist
        )
        
        # Compute DTW costs
        dtw_costs = dtw_custom(dummy_x, dummy_y)  # [B*B]
        
        # Reshape to [B, B] energy matrix
        energy_matrix = -dtw_costs.view(batch_size, batch_size)
        
        # ====================================================================
        # Step 3: InfoNCE Contrastive Loss
        # ====================================================================
        labels = torch.arange(batch_size, device=device)
        logits = energy_matrix / self.temperature
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # ====================================================================
        # Step 4: Optional MSE loss
        # ====================================================================
        if final_pred is not None and target is not None:
            mse_loss = F.mse_loss(final_pred, target)
        else:
            mse_loss = torch.tensor(0.0, device=device)
        
        # Combined Loss
        total_loss = mse_weight * mse_loss + contrastive_weight * contrastive_loss
        
        # Accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        loss_dict = {
            'mse': mse_loss.item() if torch.is_tensor(mse_loss) else mse_loss,
            'contrastive_dtw': contrastive_loss.item(),
            'contrastive_accuracy': accuracy,
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


def contrastive_soft_dtw_alignment_loss(final_pred, target, 
                                         img_embeddings, txt_embeddings,
                                         gamma=0.1, temperature=0.1,
                                         mse_weight=1.0, contrastive_weight=0.5,
                                         use_cuda=True, normalize=True):
    """
    Functional interface for Contrastive Soft-DTW loss.
    
    This combines MSE reconstruction loss with InfoNCE-style contrastive learning
    using CUDA-accelerated Soft-DTW as the similarity measure.
    
    Args:
        final_pred (torch.Tensor): Final predicted similarity matrix [B, H, W]
        target (torch.Tensor): Ground truth similarity matrix [B, H, W]
        img_embeddings (torch.Tensor): Image patch embeddings [B, N, D]
        txt_embeddings (torch.Tensor): Text token embeddings [B, M, D]
        gamma (float): Soft-DTW smoothing parameter
        temperature (float): Temperature for InfoNCE contrastive loss
        mse_weight (float): Weight for MSE loss
        contrastive_weight (float): Weight for contrastive loss
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
    return criterion(img_embeddings, txt_embeddings, final_pred, target, 
                    mse_weight=mse_weight, contrastive_weight=contrastive_weight)


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
        # Contrastive Soft-DTW: InfoNCE-style contrastive learning with CUDA Soft-DTW
        # All contrastive learning is handled inside the loss function
        # No special handling needed in training loop - just pass embeddings
        from Parameters import (contrastive_soft_dtw_gamma, contrastive_soft_dtw_temperature,
                                contrastive_soft_dtw_mse_weight, contrastive_soft_dtw_contrastive_weight,
                                contrastive_soft_dtw_normalize)
        criterion = partial(contrastive_soft_dtw_alignment_loss,
                          gamma=contrastive_soft_dtw_gamma,
                          temperature=contrastive_soft_dtw_temperature,
                          mse_weight=contrastive_soft_dtw_mse_weight,
                          contrastive_weight=contrastive_soft_dtw_contrastive_weight,
                          use_cuda=True,
                          normalize=contrastive_soft_dtw_normalize)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: ['MSE', 'ContrastiveSoftDTW']")
    return criterion


if __name__ == "__main__":
    # Test the ContrastiveSoftDTW loss
    print("Testing ContrastiveSoftDTW loss...")
    
    batch_size = 4
    N = 20  # Image sequence length
    M = 15  # Text sequence length  
    D = 64  # Embedding dimension
    
    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_embeddings = torch.randn(batch_size, N, D, device=device, requires_grad=True)
    text_embeddings = torch.randn(batch_size, M, D, device=device)
    
    # Test the loss
    criterion = ContrastiveSoftDTW(gamma=0.1, temperature=0.1, use_cuda=torch.cuda.is_available())
    loss, loss_dict = criterion(image_embeddings, text_embeddings)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test backward pass
    loss.backward()
    print(f"Gradient computed successfully. Grad norm: {image_embeddings.grad.norm().item():.4f}")

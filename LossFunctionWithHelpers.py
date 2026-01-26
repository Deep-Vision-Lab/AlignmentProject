import torch
import torch.nn as nn
import torch.nn.functional as F

from saveDATA import *
from functools import partial



def HeightDiff_loss(inputs, targets, lamda=0.5, loss_calc=True):
    """
    Compute the path-based loss using weighted column summation.

    Args:
        targets (torch.Tensor): Smoothed path mask from smith_matrix.
        inputs (torch.Tensor): Smoothed path mask from alignment_output.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Accept inputs/targets as either [H, W] or [B, H, W]
    if targets.dim() == 2:
        # add batch dim
        targets = targets.unsqueeze(0)
        inputs = inputs.unsqueeze(0)
        _squeezed_batch = True
    else:
        _squeezed_batch = False

    if targets.dim() != 3:
        raise ValueError(f"targets must have 2 or 3 dims, got {targets.dim()}")

    B, H, W = targets.shape  # Get batch, height and width


    # Vertical column-wise weighted summation
    height_weights = torch.arange(0, H, device=targets.device).unsqueeze(1).repeat(1, W)  # Column height weights
    height_weights = height_weights.unsqueeze(0).repeat(B, 1, 1)
    height_weights = height_weights.flip(dims=[1])  # Flip to match the original orientation

    # Ensure tensors require gradients
    targets = targets.requires_grad_(True)
    inputs = inputs.requires_grad_(True)

    # Compute column-wise weighted sum
    weighted_Vertical_HN = (targets * height_weights).sum(dim=1)  # Sum over colums (height-weighted)
    weighted_Vertical_HDIFF= (inputs * height_weights).sum(dim=1)

    # Compute absolute column-wise difference
    vertical_loss = torch.abs(weighted_Vertical_HN - weighted_Vertical_HDIFF).mean()
    #######################################################################################################################
    #Horizontal column-wise weighted summation

    height_weights = torch.arange(0, W, device=targets.device).unsqueeze(1).repeat(1, H).T  # Column height weights
    height_weights = height_weights.unsqueeze(0).repeat(B, 1, 1)

    # Ensure tensors require gradients
    targets = targets.requires_grad_(True)
    inputs = inputs.requires_grad_(True)

    # Compute column-wise weighted sum
    weighted_Horizontal_HN = (targets * height_weights).sum(dim=2)  # Sum over rows (height-weighted)
    weighted_Horizontal_HDIFF = (inputs * height_weights).sum(dim=2)

    # Compute absolute column-wise difference
    horizontal_loss = torch.abs(weighted_Horizontal_HN - weighted_Horizontal_HDIFF).mean()
    loss = lamda * vertical_loss + (1 - lamda) * horizontal_loss
    if loss_calc:
        return loss
    else:
        # If original inputs were 2D (single example), return squeezed vectors for convenience
        if _squeezed_batch and B == 1:
            return weighted_Vertical_HN.squeeze(0), weighted_Vertical_HDIFF.squeeze(0), weighted_Horizontal_HN.squeeze(0), weighted_Horizontal_HDIFF.squeeze(0)
        return weighted_Vertical_HN, weighted_Vertical_HDIFF, weighted_Horizontal_HN, weighted_Horizontal_HDIFF


def guided_attention_loss(pred, target, g=0.2, alpha=1.0):
    """
    Guided Attention Loss for sequence alignment tasks.
    Penalizes attention away from the diagonal and encourages prediction to match the ground truth path.

    Args:
        pred (torch.Tensor): Predicted attention/alignment map of shape [B, H, W].
        target (torch.Tensor): Ground truth path mask of shape [B, H, W].
        g (float): Gaussian spread parameter (controls width of diagonal band).
        alpha (float): Weight for the guided attention regularization term.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, H, W = pred.shape
    device = pred.device
    # Create normalized position indices
    i = torch.arange(H, device=device).unsqueeze(1) / H  # [H, 1]
    j = torch.arange(W, device=device).unsqueeze(0) / W  # [1, W]
    # Compute Gaussian mask centered on diagonal
    guided_mask = 1.0 - torch.exp(-((i - j) ** 2) / (2 * g * g))  # [H, W]
    guided_mask = guided_mask.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    # Main loss: encourage prediction to match ground truth path
    main_loss = F.mse_loss(pred, target)
    # Regularization: penalize attention away from diagonal
    reg_loss = (pred * guided_mask).mean()
    return main_loss + alpha * reg_loss


# ============================================================================
# OPTIMIZATION 2: Monotonic Alignment / Diagonal Constraint Loss
# ============================================================================
def diagonal_alignment_loss(pred, target=None, sigma=0.15, monotonic_weight=0.5):
    """
    Diagonal Alignment Constraint Loss for monotonic text alignment.
    
    Since text flows linearly (right-to-left for Arabic, left-to-right for English),
    the alignment must be monotonic. This loss forces the highest connection 
    strength to lie near the diagonal line of the matrix.
    
    Args:
        pred (torch.Tensor): Predicted alignment/similarity map of shape [B, H, W].
        target (torch.Tensor): Optional ground truth path mask of shape [B, H, W].
        sigma (float): Controls the width of the diagonal band (smaller = tighter).
        monotonic_weight (float): Weight for monotonicity constraint (0-1).
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, H, W = pred.shape
    device = pred.device
    
    # Create normalized position indices for diagonal mask
    i_idx = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1) / max(H, 1)  # [H, 1]
    j_idx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0) / max(W, 1)  # [1, W]
    
    # Gaussian diagonal mask: high values near diagonal, low far away
    diagonal_mask = torch.exp(-((i_idx - j_idx) ** 2) / (2 * sigma ** 2))  # [H, W]
    diagonal_mask = diagonal_mask.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # ========================================================================
    # Loss 1: Encourage predictions to follow diagonal band
    # Penalize high predictions far from diagonal (use squared to ensure positive)
    # ========================================================================
    off_diagonal_penalty = (pred ** 2) * (1.0 - diagonal_mask)
    diagonal_loss = off_diagonal_penalty.mean()
    
    # ========================================================================
    # Loss 2: Monotonicity constraint
    # For each row, the "peak" should move to the right as we go down rows
    # This enforces the linear reading order of text
    # ========================================================================
    # Get column index of maximum value per row
    pred_softmax = F.softmax(pred, dim=-1)  # Normalize along columns
    col_indices = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, W]
    
    # Expected column position (soft argmax)
    expected_col = (pred_softmax * col_indices).sum(dim=-1)  # [B, H]
    
    # Monotonicity: expected column should increase (or stay same) as row increases
    col_diff = expected_col[:, 1:] - expected_col[:, :-1]  # [B, H-1]
    
    # Penalize negative differences (going backwards)
    monotonic_loss = F.relu(-col_diff).mean()
    
    # ========================================================================
    # Loss 3: Optional - Match ground truth if provided
    # ========================================================================
    if target is not None:
        reconstruction_loss = F.mse_loss(pred, target)
    else:
        reconstruction_loss = torch.tensor(0.0, device=device)
    
    # Combined loss
    total_loss = diagonal_loss + monotonic_weight * monotonic_loss + reconstruction_loss
    
    return total_loss


def diagonal_regularization_loss(similarity_matrix, sigma=0.15, monotonic_weight=0.3):
    """
    Diagonal regularization loss for text-image similarity matrices.
    Forces the alignment to follow a diagonal pattern (monotonic alignment).
    
    Args:
        similarity_matrix (torch.Tensor): Text-Image similarity matrix [B, H, W]
        sigma (float): Controls width of diagonal band (smaller = tighter)
        monotonic_weight (float): Weight for monotonicity constraint
    
    Returns:
        torch.Tensor: Scalar regularization loss (always non-negative)
    """
    B, H, W = similarity_matrix.shape
    device = similarity_matrix.device
    
    # Create normalized position indices for diagonal mask
    i_idx = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1) / max(H, 1)
    j_idx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0) / max(W, 1)
    
    # Gaussian diagonal mask: high values near diagonal, low far away
    diagonal_mask = torch.exp(-((i_idx - j_idx) ** 2) / (2 * sigma ** 2))
    diagonal_mask = diagonal_mask.unsqueeze(0).expand(B, -1, -1)
    
    # Penalize high ABSOLUTE values far from diagonal (use squared to ensure positive)
    # This ensures the loss is always non-negative regardless of similarity sign
    off_diagonal_penalty = (similarity_matrix ** 2) * (1.0 - diagonal_mask)
    diagonal_loss = off_diagonal_penalty.mean()
    
    # Monotonicity constraint: peaks should move consistently along diagonal
    pred_softmax = F.softmax(similarity_matrix, dim=-1)
    col_indices = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    expected_col = (pred_softmax * col_indices).sum(dim=-1)  # [B, H]
    col_diff = expected_col[:, 1:] - expected_col[:, :-1]
    monotonic_loss = F.relu(-col_diff).mean()
    
    return diagonal_loss + monotonic_weight * monotonic_loss


def mse_with_diagonal_regularization(final_pred, target, 
                                      img_txt_sim1, img_txt_sim2,
                                      mse_weight=1.0, reg_weight=0.3, 
                                      sigma=0.15, monotonic_weight=0.3):
    """
    Combined loss: MSE on final similarity matrix + diagonal regularization on intermediate matrices.
    
    Args:
        final_pred (torch.Tensor): Final similarity matrix from dot product [B, H, W]
        target (torch.Tensor): Ground truth similarity matrix [B, H, W]
        img_txt_sim1 (torch.Tensor): Text1-Image1 similarity matrix [B, T1, I1]
        img_txt_sim2 (torch.Tensor): Text2-Image2 similarity matrix [B, T2, I2]
        mse_weight (float): Weight for MSE loss on final matrix
        reg_weight (float): Weight for diagonal regularization on intermediate matrices
        sigma (float): Diagonal band width for regularization
        monotonic_weight (float): Weight for monotonicity in regularization
    
    Returns:
        tuple: (total_loss, loss_dict)
    """
    # Main MSE loss between final prediction and ground truth
    mse_loss = F.mse_loss(final_pred, target)
    
    # Diagonal regularization on text1-image1 similarity matrix
    diag_reg_1 = diagonal_regularization_loss(img_txt_sim1, sigma=sigma, monotonic_weight=monotonic_weight)
    
    # Diagonal regularization on text2-image2 similarity matrix  
    diag_reg_2 = diagonal_regularization_loss(img_txt_sim2, sigma=sigma, monotonic_weight=monotonic_weight)
    
    # Average the two regularization terms
    diag_reg_total = (diag_reg_1 + diag_reg_2) / 2.0
    
    # Combined loss
    total_loss = mse_weight * mse_loss + reg_weight * diag_reg_total
    
    loss_dict = {
        'mse': mse_loss.item(),
        'diag_reg_1': diag_reg_1.item(),
        'diag_reg_2': diag_reg_2.item(),
        'diag_reg_total': diag_reg_total.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


# ============================================================================
# CONTRASTIVE SOFT-DTW LOSS (Triplet-style Learning)
# ============================================================================
def contrastive_alignment_loss(positive_sim, negative_sims, margin=0.5, 
                                positive_weight=1.0, negative_weight=1.0,
                                diagonal_weight=0.3, sigma=0.15):
    """
    Contrastive loss for text-image alignment with triplet-style learning.
    
    This loss teaches the model to:
    1. Minimize distance between correct image-text pairs (positive)
    2. Maximize distance between incorrect image-text pairs (negative)
    
    Formula: Loss = Positive_Loss + max(0, margin - Negative_Loss)
    
    This forces the "bunch of patches" to be close ONLY to the correct letter
    and far from all other letters in the batch.
    
    Args:
        positive_sim (torch.Tensor): Similarity matrix for correct pairs [B, T, I]
            - Text1 vs Image1, Text2 vs Image2 (diagonal of batch comparison)
        negative_sims (list of torch.Tensor): List of similarity matrices for wrong pairs
            - Text1 vs Image2, Text2 vs Image1, etc. (off-diagonal comparisons)
        margin (float): Margin for contrastive loss (how far apart negatives should be)
        positive_weight (float): Weight for positive pair loss
        negative_weight (float): Weight for negative pair loss  
        diagonal_weight (float): Weight for diagonal constraint on positive pairs
        sigma (float): Diagonal band width for regularization
    
    Returns:
        tuple: (total_loss, loss_dict) with individual loss components
    """
    device = positive_sim.device
    B, H, W = positive_sim.shape
    
    # ========================================================================
    # Positive Loss: Encourage high similarity on diagonal (correct alignment)
    # ========================================================================
    # Create diagonal mask - we want high values near the diagonal for correct pairs
    i_idx = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1) / max(H, 1)
    j_idx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0) / max(W, 1)
    diagonal_mask = torch.exp(-((i_idx - j_idx) ** 2) / (2 * sigma ** 2))
    diagonal_mask = diagonal_mask.unsqueeze(0).expand(B, -1, -1)
    
    # Positive loss: We want HIGH similarity on diagonal
    # Use negative log-likelihood style: -log(sigmoid(similarity))
    # This keeps loss positive and bounded
    positive_on_diagonal = (positive_sim * diagonal_mask).sum() / (diagonal_mask.sum() + 1e-8)
    
    # Use softplus to ensure positive loss: softplus(-x) = log(1 + exp(-x))
    # When similarity is high (positive), loss is low
    # When similarity is low (negative), loss is high
    positive_loss = F.softplus(-positive_on_diagonal)
    
    # Also penalize high values off-diagonal for positive pairs (spurious matches)
    off_diagonal_values = positive_sim * (1.0 - diagonal_mask)
    # Use ReLU to only penalize positive off-diagonal similarities
    off_diagonal_penalty = F.relu(off_diagonal_values).mean()
    positive_loss = positive_loss + diagonal_weight * off_diagonal_penalty
    
    # ========================================================================
    # Negative Loss: Discourage high similarity for wrong pairs
    # ========================================================================
    # For negative pairs, we want LOW similarity everywhere
    # Penalize any positive similarity in negative pairs
    negative_losses = []
    for neg_sim in negative_sims:
        # Penalize positive similarities in wrong pairs
        # ReLU ensures we only penalize when similarity > 0
        neg_penalty = F.relu(neg_sim + margin).mean()  # Push below -margin
        negative_losses.append(neg_penalty)
    
    if negative_losses:
        contrastive_loss = torch.stack(negative_losses).mean()
        avg_negative_sim = torch.stack([ns.mean() for ns in negative_sims]).mean()
    else:
        contrastive_loss = torch.tensor(0.0, device=device)
        avg_negative_sim = torch.tensor(0.0, device=device)
    
    # ========================================================================
    # Combined Loss (guaranteed non-negative)
    # ========================================================================
    total_loss = positive_weight * positive_loss + negative_weight * contrastive_loss
    
    # Ensure total loss is non-negative (safety clamp)
    total_loss = F.relu(total_loss)
    
    loss_dict = {
        'positive': positive_loss.item(),
        'negative': contrastive_loss.item(),
        'avg_neg_sim': avg_negative_sim.item(),
        'off_diagonal': off_diagonal_penalty.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


def contrastive_mse_alignment_loss(final_pred, target, 
                                    positive_sim, negative_sims,
                                    mse_weight=1.0, contrastive_weight=0.5,
                                    margin=0.3, sigma=0.15):
    """
    Combined MSE + Contrastive loss for alignment with classification improvement.
    
    This combines:
    1. MSE loss on final text-text similarity matrix (alignment accuracy)
    2. Contrastive loss on text-image similarity matrices (letter classification)
    
    The contrastive component forces the model to learn that an 'Alif' patch
    should be similar to 'Alif' embeddings and different from 'Lam' embeddings.
    
    Args:
        final_pred (torch.Tensor): Final similarity matrix [B, T1, T2]
        target (torch.Tensor): Ground truth similarity matrix [B, T1, T2]
        positive_sim (torch.Tensor): Correct text-image similarity [B, T, I]
        negative_sims (list): List of wrong text-image similarities
        mse_weight (float): Weight for MSE reconstruction loss
        contrastive_weight (float): Weight for contrastive classification loss
        margin (float): Margin for contrastive loss
        sigma (float): Diagonal band width
    
    Returns:
        tuple: (total_loss, loss_dict)
    """
    # MSE loss on final output
    mse_loss = F.mse_loss(final_pred, target)
    
    # Contrastive loss for letter classification
    contrastive_loss, contrastive_dict = contrastive_alignment_loss(
        positive_sim, negative_sims, 
        margin=margin, sigma=sigma
    )
    
    # Combined loss
    total_loss = mse_weight * mse_loss + contrastive_weight * contrastive_loss
    
    loss_dict = {
        'mse': mse_loss.item(),
        'contrastive': contrastive_loss.item(),
        'positive': contrastive_dict['positive'],
        'negative': contrastive_dict['negative'],
        'avg_neg_sim': contrastive_dict['avg_neg_sim'],
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


def combined_alignment_loss(pred, target, mse_weight=1.0, diagonal_weight=0.3, 
                           monotonic_weight=0.2, guided_weight=0.0, sigma=0.15, g=0.2):
    """
    Combined loss function for alignment training.
    
    Combines:
    - MSE loss for matching ground truth
    - Diagonal constraint to force alignment near diagonal
    - Monotonicity constraint to enforce linear reading order
    - Optional guided attention regularization
    
    Args:
        pred (torch.Tensor): Predicted alignment map [B, H, W]
        target (torch.Tensor): Ground truth alignment map [B, H, W]
        mse_weight (float): Weight for MSE reconstruction loss
        diagonal_weight (float): Weight for diagonal band constraint
        monotonic_weight (float): Weight for monotonicity constraint
        guided_weight (float): Weight for guided attention (0 to disable)
        sigma (float): Diagonal band width
        g (float): Guided attention Gaussian width
    
    Returns:
        tuple: (total_loss, loss_dict) where loss_dict contains individual components
    """
    device = pred.device
    B, H, W = pred.shape
    
    # MSE Loss
    mse_loss = F.mse_loss(pred, target)
    
    # Diagonal constraint
    i_idx = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1) / max(H, 1)
    j_idx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0) / max(W, 1)
    diagonal_mask = torch.exp(-((i_idx - j_idx) ** 2) / (2 * sigma ** 2))
    diagonal_mask = diagonal_mask.unsqueeze(0).expand(B, -1, -1)
    
    off_diagonal = pred * (1.0 - diagonal_mask)
    diagonal_loss = off_diagonal.mean()
    
    # Monotonicity constraint
    pred_softmax = F.softmax(pred, dim=-1)
    col_indices = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    expected_col = (pred_softmax * col_indices).sum(dim=-1)
    col_diff = expected_col[:, 1:] - expected_col[:, :-1]
    monotonic_loss = F.relu(-col_diff).mean()
    
    # Guided attention (optional)
    if guided_weight > 0:
        guided_mask = 1.0 - torch.exp(-((i_idx - j_idx) ** 2) / (2 * g * g))
        guided_mask = guided_mask.unsqueeze(0).expand(B, -1, -1)
        guided_loss = (pred * guided_mask).mean()
    else:
        guided_loss = torch.tensor(0.0, device=device)
    
    # Total loss
    total_loss = (mse_weight * mse_loss + 
                  diagonal_weight * diagonal_loss + 
                  monotonic_weight * monotonic_loss +
                  guided_weight * guided_loss)
    
    loss_dict = {
        'mse': mse_loss.item(),
        'diagonal': diagonal_loss.item(),
        'monotonic': monotonic_loss.item(),
        'guided': guided_loss.item() if isinstance(guided_loss, torch.Tensor) else 0.0,
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict



def dice_loss(pred, target, eps=1e-8):
    """
    Dice Loss for binary segmentation/alignment tasks.
    Args:
        pred (torch.Tensor): Predicted mask [B, H, W] or [B, 1, H, W].
        target (torch.Tensor): Ground truth mask [B, H, W] or [B, 1, H, W].
        eps (float): Smoothing term to avoid division by zero.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Flatten if needed
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2. * intersection + eps) / (union + eps)
    loss = 1 - dice.mean()
    return loss



def kl_divergence_loss(pred, target, eps=1e-8, class_dim=1):
    """
    KL-divergence loss for multi-label classification.
    Args:
        pred (torch.Tensor): Logits or unnormalized predictions, shape [B, C, ...].
        target (torch.Tensor): Target probabilities, shape [B, C, ...].
        eps (float): Small value to avoid division by zero.
    Returns:
        torch.Tensor: Scalar KL-divergence loss.
    """
    # Convert predictions to log-probabilities
    log_probs = torch.nn.functional.log_softmax(pred, dim=class_dim)
    # Convert targets to probabilities
    probs_target = torch.nn.functional.softmax(target, dim=class_dim)

    # Optional: gently avoid exact zeros in target to keep log well-defined
    probs_target = probs_target.clamp_min(eps)  # preserves gradients

    # KL between target and prediction: E_target[log(target) - log(pred)]
    kl = torch.nn.functional.kl_div(log_probs, probs_target, reduction='none')
    # Reduce over class dimension, then mean over batch and remaining dims
    loss = kl.sum(dim=class_dim)
    # If there are extra dims (e.g., H/W), average them too
    if loss.dim() > 1:
        loss = loss.mean(dim=tuple(range(1, loss.dim())))
    loss = loss.mean()  # mean over batch
    return loss


def wasserstein_distance(pred, target, p=1):
    """
    Compute the Wasserstein (Earth Mover's) distance between two distributions using PyTorch.
    Args:
        pred (torch.Tensor): Predicted distribution, shape [B, C, ...]. Should sum to 1 over C.
        target (torch.Tensor): Target distribution, shape [B, C, ...]. Should sum to 1 over C.
        p (int): The norm degree (default 1 for classic Wasserstein).
    Returns:
        torch.Tensor: Scalar Wasserstein distance.
    """
    # Flatten distributions along class/channel dimension
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    # Ensure distributions are normalized
    pred_flat = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + 1e-8)
    target_flat = target_flat / (target_flat.sum(dim=1, keepdim=True) + 1e-8)
    # Compute cumulative distributions
    pred_cdf = torch.cumsum(pred_flat, dim=1)
    target_cdf = torch.cumsum(target_flat, dim=1)
    # Wasserstein distance is the mean Lp distance between CDFs
    wass = torch.mean(torch.abs(pred_cdf - target_cdf) ** p)
    return wass


def Loss_choice(loss_type):
    if loss_type == 'HeightDiff':
        criterion = partial(HeightDiff_loss, lamda=0.5, loss_calc=True)
    elif loss_type == 'MSE':
        criterion = nn.MSELoss(reduction='sum')
    elif loss_type == 'GuidedAttention':
        criterion = guided_attention_loss
    elif loss_type == 'KL-Divergence':
        criterion = kl_divergence_loss
    elif loss_type == 'Dice':
        criterion = dice_loss
    elif loss_type == 'Wasserstein':
        criterion = wasserstein_distance
    elif loss_type == 'SoftCrossEntropy':
        criterion = soft_cross_entropy
    elif loss_type == 'DiagonalAlignment':
        # New: Diagonal constraint with monotonicity
        criterion = partial(diagonal_alignment_loss, sigma=0.15, monotonic_weight=0.5)
    elif loss_type == 'CombinedAlignment':
        # New: Combined MSE + Diagonal + Monotonic loss
        criterion = partial(combined_alignment_loss, 
                          mse_weight=1.0, 
                          diagonal_weight=0.3, 
                          monotonic_weight=0.2,
                          guided_weight=0.0)
    elif loss_type == 'MSEWithDiagonalReg':
        # MSE on final matrix + diagonal regularization on intermediate text-image matrices
        criterion = partial(mse_with_diagonal_regularization,
                          mse_weight=1.0,
                          reg_weight=0.3,
                          sigma=0.15,
                          monotonic_weight=0.3)
    elif loss_type == 'ContrastiveMSE':
        # MSE + Contrastive loss for better letter classification
        # Note: This loss requires special handling in training loop
        # to compute negative pairs from batch
        criterion = partial(contrastive_mse_alignment_loss,
                          mse_weight=1.0,
                          contrastive_weight=0.5,
                          margin=0.3,
                          sigma=0.15)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return criterion



if __name__ == "__main__":
    # Example usage
    inputs = torch.rand(8, 10, 32)  # Simulated input tensor (positive)
    targets = torch.rand(8, 10, 32)  # Simulated target tensor (positive)

    # Apply smoothing
    # smoothed_inputs = smooth_path(inputs)
    # smoothed_targets = smooth_path(targets)

    # Compute loss
    loss = wasserstein_distance(inputs, targets)
    print(f"Computed loss: {loss.item()}")


def soft_cross_entropy(pred, target,eps=1e-8):
    """
    Soft Cross-Entropy Loss for multi-class classification with soft targets.
    Args:
        pred (torch.Tensor): Logits or unnormalized predictions, shape [B, C, ...].
        target (torch.Tensor): Target probabilities, shape [B, C, ...].
    Returns:
        torch.Tensor: Scalar loss value.
    """
    pred = torch.clamp(pred, min=eps, max=1.0)
    log_probs = torch.log(pred)
    loss = -(target * log_probs).sum()
    return loss
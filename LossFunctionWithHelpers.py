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
import torch
import torch.nn.functional as F
from saveDATA import *


def smooth_path(mask, kernel_size=5, sigma=1):
    """
    Smooth the path using a Gaussian filter.

    Args:
        mask (torch.Tensor): Binary mask (1 for path, 0 elsewhere).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian blur.

    Returns:
        torch.Tensor: Smoothed mask.
    """
    device = mask.device

    # Create a Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gaussian_kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()

    # Convert to 2D filter
    kernel_2d = gaussian_kernel[:, None] * gaussian_kernel[None, :]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).to(device) # Shape: (H,W)
    # Apply Gaussian filter
    mask = mask.unsqueeze(1) # Add batch and channel dims
    smoothed_mask = F.conv2d(mask, kernel_2d, padding=kernel_size//2)

    return smoothed_mask.squeeze(1)


def HeightDiff_loss(inputs, targets, lamda=0.5):
    """
    Compute the path-based loss using weighted column summation.

    Args:
        targets (torch.Tensor): Smoothed path mask from smith_matrix.
        inputs (torch.Tensor): Smoothed path mask from alignment_output.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, H, W = targets.shape  # Get height and width


    # Vertical column-wise weighted summation
    height_weights = torch.arange(0, H, device=targets.device).unsqueeze(1).repeat(1, W)  # Column height weights
    height_weights = height_weights.unsqueeze(0).repeat(B, 1, 1)
    height_weights = height_weights.flip(dims=[1])  # Flip to match the original orientation
    # print(f'height_weights: {height_weights}')
    # Ensure tensors require gradients
    targets = targets.requires_grad_(True)
    inputs = inputs.requires_grad_(True)

    # Compute column-wise weighted sum
    weighted_HS = (targets * height_weights).sum(dim=1)  # Sum over colums (height-weighted)
    weighted_HA = (inputs * height_weights).sum(dim=1)
    # print(f'Vertical weighted_HS: {weighted_HS.shape}')
    # print(f'Vertical weighted_HA: {weighted_HA.shape}')

    # Compute absolute column-wise difference
    vertical_loss = torch.abs(weighted_HS - weighted_HA).mean()

    #######################################################################################################################
    #Horizontal column-wise weighted summation

    height_weights = torch.arange(0, W, device=targets.device).unsqueeze(1).repeat(1, H).T  # Column height weights
    height_weights = height_weights.unsqueeze(0).repeat(B, 1, 1)
    # print(f'Horizontal height_weights: {height_weights.shape}')
    # print(f'height_weights: {height_weights}')
    # Ensure tensors require gradients
    targets = targets.requires_grad_(True)
    inputs = inputs.requires_grad_(True)

    # Compute column-wise weighted sum
    weighted_HS = (targets * height_weights).sum(dim=2)  # Sum over rows (height-weighted)
    weighted_HA = (inputs * height_weights).sum(dim=2)
    # print(f'Horizontal weighted_HS: {weighted_HS.shape}')
    # print(f'Horizontal weighted_HA: {weighted_HA.shape}')

    # Compute absolute column-wise difference
    horizontal_loss = torch.abs(weighted_HS - weighted_HA).mean()

    loss = lamda * vertical_loss + (1 - lamda) * horizontal_loss
    return loss
    ################################################################################
    # if plot_vectors:
    #     save_Vectors_plot(weighted_HA.cpu().detach().numpy(), weighted_HS.cpu().detach().numpy(),
    #                     f'Vectors_plot/vetors_{epoch}_{batch_idx}_{i}.png' if epoch is not None
    #                     else f'Vectors_plot/vetors_{batch_idx}_{i}.png',
    #                     f'VectorsSub_plot/vetors_{epoch}_{batch_idx}_{i}.png' if epoch is not None
    #                     else f'VectorsSub_plot/vetors_{batch_idx}_{i}.png'
    #                     )

    #     cosine_sim = F.cosine_similarity(weighted_HS, weighted_HA, dim=0, eps=1e-8)
    #     with open(file_path, 'a') as f:
    #         f.write(f'vectors {batch_idx}_{i} similarity: {cosine_sim}\n')
    ################################################################################


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



def kl_divergence_loss(pred, target, eps=1e-8):
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
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    
    # Normalize target to ensure it sums to 1 along classes
    # target = F.softmax(target, dim=1)
    target = target / (target.sum(dim=1, keepdim=True) + eps)
    
    # Compute KL divergence loss
    kl = torch.nn.functional.kl_div(log_probs, target, reduction='none')
    loss = kl.sum(dim=1).mean()
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



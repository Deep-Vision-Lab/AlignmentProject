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


def compute_path_loss(inputs, targets):
    """
    Compute the path-based loss using weighted column summation.

    Args:
        targets (torch.Tensor): Smoothed path mask from smith_matrix.
        inputs (torch.Tensor): Smoothed path mask from alignment_output.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, H, W = targets.shape  # Get height and width
    height_weights = torch.arange(1, H + 1, device=targets.device).unsqueeze(1).repeat(1, W)  # Column height weights
    height_weights = height_weights.unsqueeze(0).repeat(B, 1, 1)

    # Ensure tensors require gradients
    targets = targets.requires_grad_(True)
    inputs = inputs.requires_grad_(True)

    # Compute column-wise weighted sum
    weighted_HS = (targets * height_weights).sum(dim=2)  # Sum over rows (height-weighted)
    weighted_HA = (inputs * height_weights).sum(dim=2)
    # print(f'weighted_HS: {weighted_HS.shape}')
    # print(f'weighted_HA: {weighted_HA.shape}')
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

    # Compute absolute column-wise difference
    loss = torch.abs(weighted_HS - weighted_HA).mean()

    return loss
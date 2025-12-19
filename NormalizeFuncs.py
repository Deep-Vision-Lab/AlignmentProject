import torch
import torch.nn.functional as F




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



def normalize_func(matrix, normalize_type):
    """
    Optionally smooth and normalize the alignment output tensor.
    Args:
        matrix (torch.Tensor): The alignment output tensor.
        normalize_type (str): Normalization type: 'min_max', 'mean_std', or ''.
    Returns:
        torch.Tensor: Smoothed and normalized alignment output.
    """
    # matrix = smooth_path(matrix)
    if normalize_type == 'min_max':
        matrix = min_max_normalize(matrix)
    elif normalize_type == 'mean_std':
        matrix = mean_std_normalize(matrix)
    elif normalize_type == 'average':
        matrix = average_normalize(matrix)
    return matrix



def min_max_normalize(tensor):
    """
    Min-max normalize the input tensor to the range [0, 1].

    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Min-max normalized tensor.
    """
    tensor_min = tensor.amin(dim=(-2, -1), keepdim=True)
    tensor_max = tensor.amax(dim=(-2, -1), keepdim=True)
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    return normalized_tensor



def mean_std_normalize(tensor):
    """
    Standard score normalize the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Standard score normalized tensor.
    """
    tensor_mean = tensor.mean(dim=(-2, -1), keepdim=True)
    tensor_std = tensor.std(dim=(-2, -1), keepdim=True)
    normalized_tensor = (tensor - tensor_mean) / (tensor_std + 1e-8)
    return normalized_tensor



def average_normalize(tensor, eps=1e-8):
    """
    Average normalize the input tensor so that the sum over H and W equals 1.

    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Average normalized tensor.
    """
    tensor_sum = tensor.sum(dim=(1, 2), keepdim=True)
    normalized_tensor = tensor / (tensor_sum + eps)
    return normalized_tensor




if __name__ == "__main__":

    # Example usage
    input_tensor = torch.rand(4, 32, 32)  # Example tensor

    # normalized_mean_std = mean_std_normalize(input_tensor)
    # print("Mean-std normalized tensor shape:", normalized_mean_std.shape)

    # sum_normalized = normalized_mean_std.sum(dim=(1))
    # print("Sum over H and W dimensions:", sum_normalized)

    normalized_tensor = average_normalize(input_tensor)
    print("Average normalized tensor shape:", normalized_tensor.shape)

    sum_normalized = normalized_tensor.sum(dim=(1, 2))
    print("Sum over H and W dimensions:", sum_normalized)
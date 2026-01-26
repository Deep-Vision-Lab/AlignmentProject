"""
Script to compute similarity matrix between two images using patch embeddings.

This script:
1. Loads two images (img1_{i}.png and img2_{i}.png)
2. Slices them into patches using a sliding window
3. Passes patches through an embedding model (CNN, Transformer, or DINOv2)
4. Computes cosine similarity between all pairs of patch embeddings
5. Returns/saves the similarity matrix
"""

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import argparse
import os

from Parameters import *
from embeddingModel import EmbeddingModel, sliding_window
from DiffNWAlgo import compute_nw_score_matrix
from pathExtractor import compute_traceback_path


def load_image(image_path: str, target_height: int = 128) -> torch.Tensor:
    """
    Load an image and preprocess it.
    
    Args:
        image_path: Path to the image file
        target_height: Target height for the image (width is preserved proportionally or as-is)
    
    Returns:
        Image tensor of shape [1, 3, H, W]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def compute_cosine_similarity_matrix(embeddings_a: torch.Tensor, 
                                      embeddings_b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two sets of embeddings.
    
    Args:
        embeddings_a: Tensor of shape [batch, N, D] - embeddings from image 1
        embeddings_b: Tensor of shape [batch, M, D] - embeddings from image 2
    
    Returns:
        Similarity matrix of shape [batch, N, M] where each element is the
        cosine similarity between the corresponding patch embeddings
    """
    # Normalize embeddings for cosine similarity
    # embeddings_a: [B, N, D], embeddings_b: [B, M, D]
    
    # Transpose embeddings_b for matrix multiplication: [B, D, M]
    embeddings_b_transposed = embeddings_b.transpose(1, 2)
    
    # Compute dot product: [B, N, D] @ [B, D, M] = [B, N, M]
    dot_product = torch.bmm(embeddings_a, embeddings_b_transposed)
    
    # Compute magnitudes
    magnitude_a = torch.norm(embeddings_a, dim=2, keepdim=True)  # [B, N, 1]
    magnitude_b = torch.norm(embeddings_b, dim=2, keepdim=True)  # [B, M, 1]
    
    # Compute denominator: [B, N, 1] * [B, 1, M] = [B, N, M]
    denominator = magnitude_a * magnitude_b.transpose(1, 2) + 1e-8
    
    # Compute cosine similarity
    cosine_similarity = dot_product / denominator
    
    return cosine_similarity


def add_patch_axis_labels(ax, patches, axis='x', zoom=0.3):
    """
    Add patch images as labels on the specified axis.
    
    Args:
        ax: Matplotlib axis
        patches: Tensor of patches [num_patches, C, H, W]
        axis: 'x' or 'y'
        zoom: Zoom factor for the patch images
    """
    patches = patches.cpu()
    for i, patch in enumerate(patches):
        # Format patch for display
        if patch.shape[0] == 3:  # C, H, W
            disp_patch = patch.permute(1, 2, 0).numpy()
        else:  # 1, H, W or H, W
            disp_patch = patch.squeeze().numpy()
        
        # Normalize for display
        disp_patch = (disp_patch - disp_patch.min()) / (disp_patch.max() - disp_patch.min() + 1e-6)
        
        # Rotate for y-axis to align with vertical direction
        if axis == 'y':
            disp_patch = np.rot90(disp_patch, k=1)
            
        imagebox = OffsetImage(disp_patch, zoom=zoom, cmap='gray')
        
        if axis == 'x':
            ab = AnnotationBbox(imagebox, (i, 0),
                                xybox=(0, -25),
                                xycoords=('data', 'axes fraction'),
                                boxcoords="offset points",
                                box_alignment=(0.5, 1.0),
                                frameon=False)
        else:  # y axis
            ab = AnnotationBbox(imagebox, (0, i),
                                xybox=(-25, 0),
                                xycoords=('axes fraction', 'data'),
                                boxcoords="offset points",
                                box_alignment=(1.0, 0.5),
                                frameon=False)
            
        ax.add_artist(ab)


def compute_similarity_matrix_for_pair(
    image1_path: str,
    image2_path: str,
    stride: int = window_size,
    vector_size: int = 128,
    model_arch: str = 'CNN',
    model: EmbeddingModel = None,
    device: str = 'cuda'
) -> tuple:
    """
    Compute similarity matrix between two images.
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        window_size: Size of the sliding window for patches
        stride: Stride for sliding window (defaults to window_size // 2)
        vector_size: Size of the embedding vectors
        model_arch: Architecture to use ('CNN', 'CNN-Transformer', 'dinov2', 'Transformer')
        model: Pre-loaded embedding model (optional, will create one if None)
        device: Device to use for computation
    
    Returns:
        Tuple of (similarity_matrix, patches_a, patches_b)
    """
    if stride is None:
        stride = window_size
    
    # Load images
    print(f"Loading images...")
    image1 = load_image(image1_path).to(device)
    image2 = load_image(image2_path).to(device)
    
    print(f"Image 1 shape: {image1.shape}")
    print(f"Image 2 shape: {image2.shape}")
    
    # Create or use provided embedding model
    if model is None:
        print(f"Creating embedding model with architecture: {model_arch}")
        model = EmbeddingModel(
            window_size=window_size,
            stride=window_size,
            vector_size=vector_size,
            model_arch=model_arch,
            device=device
        ).to(device)
        model.eval()
    
    # Get embeddings for both images
    print(f"Computing embeddings...")
    with torch.no_grad():
        embeddings_a, embeddings_b = model(image1, image2, show_dims=True)
        
        # Extract patches for visualization
        patches_a = sliding_window(image1, window_size, window_size).squeeze(0)
        patches_b = sliding_window(image2, window_size, window_size).squeeze(0)
    
    print(f"Embeddings A shape: {embeddings_a.shape}")
    print(f"Embeddings B shape: {embeddings_b.shape}")
    print(f"Patches A shape: {patches_a.shape}")
    print(f"Patches B shape: {patches_b.shape}")
    
    # Compute cosine similarity matrix
    print(f"Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity_matrix(embeddings_a, embeddings_b)
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Remove batch dimension for single image pair
    similarity_matrix = similarity_matrix.squeeze(0)
    
    return similarity_matrix, patches_a, patches_b


def visualize_similarity_matrix(
    similarity_matrix: torch.Tensor,
    patches_a: torch.Tensor = None,
    patches_b: torch.Tensor = None,
    save_path: str = None,
    title: str = "Cosine Similarity Matrix",
    show: bool = True,
    zoom: float = 0.3
):
    """
    Visualize the similarity matrix as a heatmap with patch images on axes.
    
    Args:
        similarity_matrix: Tensor of shape [N, M]
        patches_a: Tensor of patches from image 1 [N, C, H, W] (optional)
        patches_b: Tensor of patches from image 2 [M, C, H, W] (optional)
        save_path: Path to save the visualization (optional)
        title: Title for the plot
        show: Whether to display the plot
        zoom: Zoom factor for patch images on axes
    """
    # Convert to numpy
    sim_np = similarity_matrix.cpu().numpy()
    
    num_patches_y, num_patches_x = sim_np.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Apply threshold if requested (e.g., > 0.95)
    # We can mask values below threshold
    threshold = 0.95
    sim_np_thresholded = np.where(sim_np > threshold, sim_np, 0.0)
    
    # Plot similarity matrix
    im = ax.imshow(sim_np_thresholded, cmap='hot', aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Set up ticks
    ax.set_xticks(range(num_patches_x))
    ax.set_yticks(range(num_patches_y))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add grid lines
    y_positions = np.arange(num_patches_y + 1) - 0.5
    ax.hlines(y_positions, -0.5, num_patches_x - 0.5, colors='gray', linewidths=0.5, alpha=0.5)
    x_positions = np.arange(num_patches_x + 1) - 0.5
    ax.vlines(x_positions, -0.5, num_patches_y - 0.5, colors='gray', linewidths=0.5, alpha=0.5)
    
    # Add patch images to axes if provided
    if patches_b is not None:
        add_patch_axis_labels(ax, patches_b, axis='x', zoom=zoom)
    if patches_a is not None:
        add_patch_axis_labels(ax, patches_a, axis='y', zoom=zoom)
    
    ax.set_xlabel('Image 2 Patches')
    ax.set_ylabel('Image 1 Patches')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_score_matrix_with_path(
    score_matrix: torch.Tensor,
    path: list,
    patches_a: torch.Tensor = None,
    patches_b: torch.Tensor = None,
    save_path: str = None,
    title: str = "NW Score Matrix & Path",
    show: bool = True,
    zoom: float = 0.3
):
    """
    Visualize the NW score matrix with the alignment path overlaid.
    
    Args:
        score_matrix: Tensor of shape [N, M]
        path: List of (y, x) tuples representing the alignment path
        patches_a: Tensor of patches from image 1 [N, C, H, W]
        patches_b: Tensor of patches from image 2 [M, C, H, W]
        save_path: Path to save result
        title: Plot title
        show: Whether to show plot
        zoom: Zoom factor for patch axis labels
    """
    score_np = score_matrix.cpu().numpy()
    num_patches_y, num_patches_x = score_np.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Plot score matrix
    im = ax.imshow(score_np, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='Alignment Score')
    
    # Overlay path
    if path and len(path) > 0:
        path_y, path_x = zip(*path)
        # Use origin='lower' coordinates
        ax.plot(path_x, path_y, color='red', linewidth=3, alpha=0.8, label='Optimal Path')
        ax.scatter(path_x, path_y, color='white', s=20, edgecolors='red', zorder=5)
        ax.legend()
    
    # Set up ticks
    ax.set_xticks(range(num_patches_x))
    ax.set_yticks(range(num_patches_y))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add grid lines
    y_positions = np.arange(num_patches_y + 1) - 0.5
    ax.hlines(y_positions, -0.5, num_patches_x - 0.5, colors='gray', linewidths=0.5, alpha=0.3)
    x_positions = np.arange(num_patches_x + 1) - 0.5
    ax.vlines(x_positions, -0.5, num_patches_y - 0.5, colors='gray', linewidths=0.5, alpha=0.3)

    # Add patch images to axes
    if patches_b is not None:
        add_patch_axis_labels(ax, patches_b, axis='x', zoom=zoom)
    if patches_a is not None:
        add_patch_axis_labels(ax, patches_a, axis='y', zoom=zoom)
        
    ax.set_xlabel('Image 2 Patches')
    ax.set_ylabel('Image 1 Patches')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved score matrix visualization to {save_path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute similarity matrix between two images')
    parser.add_argument('--index', '-i', type=int, required=True,
                        help='Index of the image pair (e.g., 1 for img1_1.png and img2_1.png)')
    parser.add_argument('--data-dir', '-d', type=str, 
                        default='../DataSet/Synthetic_Arabic/images',
                        help='Directory containing the images')
    parser.add_argument('--window-size', '-w', type=int, default=16,
                        help='Size of the sliding window for patches')
    parser.add_argument('--stride', '-s', type=int, default=None,
                        help='Stride for sliding window (defaults to window_size // 2)')
    parser.add_argument('--vector-size', '-v', type=int, default=128,
                        help='Size of the embedding vectors')
    parser.add_argument('--model-arch', '-m', type=str, default='CNN',
                        choices=['CNN', 'CNN-Transformer', 'dinov2', 'Transformer'],
                        help='Model architecture to use')
    parser.add_argument('--output-dir', '-o', type=str, default='../Results/SimilarityMatrices',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the similarity matrix')
    parser.add_argument('--save-matrix', action='store_true',
                        help='Save the similarity matrix as a numpy file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Construct image paths
    image1_path = os.path.join(args.data_dir, f'img1_{args.index}.png')
    image2_path = os.path.join(args.data_dir, f'img2_{args.index}.png')
    
    # Check if images exist
    if not os.path.exists(image1_path):
        raise FileNotFoundError(f"Image not found: {image1_path}")
    if not os.path.exists(image2_path):
        raise FileNotFoundError(f"Image not found: {image2_path}")
    
    print(f"Processing image pair {args.index}:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print(f"  Window size: {args.window_size}")
    print(f"  Stride: {window_size}")
    print(f"  Model architecture: {args.model_arch}")
    
    # Compute similarity matrix
    similarity_matrix, patches_a, patches_b = compute_similarity_matrix_for_pair(
        image1_path=image1_path,
        image2_path=image2_path,
        window_size=window_size,
        stride=window_size,
        vector_size=vector_size,
        model_arch=model_arch,
        device=device
    )
    
    num_patches_1 = patches_a.shape[0]
    num_patches_2 = patches_b.shape[0]
    
    print(f"\nResults:")
    print(f"  Number of patches in image 1: {num_patches_1}")
    print(f"  Number of patches in image 2: {num_patches_2}")
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  Min similarity: {similarity_matrix.min().item():.4f}")
    print(f"  Max similarity: {similarity_matrix.max().item():.4f}")
    print(f"  Mean similarity: {similarity_matrix.mean().item():.4f}")
    
    # Create output directory if needed
    if args.visualize or args.save_matrix:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Save similarity matrix
    if args.save_matrix:
        matrix_path = os.path.join(args.output_dir, f'similarity_matrix_{args.index}.npy')
        np.save(matrix_path, similarity_matrix.cpu().numpy())
        print(f"\nSaved similarity matrix to {matrix_path}")
    
    # Visualize
    if args.visualize:
        viz_path = os.path.join(args.output_dir, f'similarity_matrix_{args.index}.png')
        visualize_similarity_matrix(
            similarity_matrix,
            patches_a=patches_a,
            patches_b=patches_b,
            save_path=viz_path,
            title=f'Cosine Similarity Matrix (Image Pair {args.index})',
            show=True
        )

        # Compute Score Matrix and Path
        # Apply threshold 0.95 (to match visualization)
        threshold = 0.95
        sim_thresholded = torch.where(similarity_matrix > threshold, 
                                     similarity_matrix, 
                                     torch.tensor(0.0).to(similarity_matrix.device))
        
        print("Computing NW score matrix...")
        # compute_nw_score_matrix returns [N+1, M+1]
        H_full = compute_nw_score_matrix(sim_thresholded, gap_penalty=gapScore)
        # Get the matrix used for alignment visualization and path extraction (N x M)
        H_vis = H_full[1:, 1:]
        
        print("Computing alignment path...")
        # compute_traceback_path expects (score_matrix, similarity_matrix, ...)
        path, _ = compute_traceback_path(
            H_vis, 
            sim_thresholded, 
            match_score=matchScore, 
            miss_score=mismatchScore, 
            gap_penalty=gapScore
        )
        
        # Visualize Score Matrix and Path
        viz_path_score = os.path.join(args.output_dir, f'score_matrix_{args.index}.png')
        visualize_score_matrix_with_path(
            H_vis,
            path=path,
            patches_a=patches_a,
            patches_b=patches_b,
            save_path=viz_path_score,
            title=f'NW Score Matrix and Path (Image Pair {args.index})',
            show=True
        )
    
    return similarity_matrix


if __name__ == "__main__":
    main()

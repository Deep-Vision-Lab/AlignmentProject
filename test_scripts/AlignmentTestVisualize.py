"""
Script to compute similarity matrix between two images, apply NW alignment,
and visualize heatmaps with input images on axes.

This script:
1. Loads two input images
2. Computes patch embeddings using the EmbeddingModel
3. Computes cosine similarity matrix between patches
4. Applies Needleman-Wunsch alignment algorithm
5. Visualizes the score matrix heatmap with input images on axes
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision import transforms

from Parameters import window_size, vector_size, device
from embeddingModel import *
from DiffNWAlgo import compute_nw_score_matrix
from pathExtractor import compute_diff_traceback_path, compute_traceback_path


# Default alignment parameters
DEFAULT_GAP_PENALTY = -1.0
DEFAULT_MATCH_SCORE = 1.0
DEFAULT_MISMATCH_SCORE = -1.0


def load_image(image_path: str) -> torch.Tensor:
    """
    Load an image and preprocess it.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image tensor of shape [1, 3, H, W]
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def compute_cosine_similarity_matrix(embeddings_a: torch.Tensor, 
                                      embeddings_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between two sets of embeddings.
    
    Args:
        embeddings_a: Tensor of shape [B, N, D]
        embeddings_b: Tensor of shape [B, M, D]
        
    Returns:
        similarity_matrix: Tensor of shape [B, N, M]
    """
    # Unsqueeze to allow broadcasting for pairwise comparison
    # embeddings_a: [B, N, D] -> [B, N, 1, D]
    # embeddings_b: [B, M, D] -> [B, 1, M, D]
    return F.cosine_similarity(embeddings_a.unsqueeze(2), embeddings_b.unsqueeze(1), dim=-1)


def patch_to_image(patch: torch.Tensor) -> np.ndarray:
    """
    Convert a patch tensor to a displayable numpy image.
    
    Args:
        patch: Tensor of shape [C, H, W]
    
    Returns:
        Numpy array of shape [H, W, C] normalized to [0, 1]
    """
    patch = patch.cpu()
    if patch.shape[0] == 3:  # RGB
        img = patch.permute(1, 2, 0).numpy()
    elif patch.shape[0] == 1:  # Grayscale
        img = patch.squeeze(0).numpy()
        img = np.stack([img, img, img], axis=-1)
    else:
        img = patch.numpy()
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def add_patch_images_to_axis(ax, patches: torch.Tensor, axis: str, 
                              zoom: float, highlighted_indices: set):
    """
    Add patch images as labels along an axis.
    
    Args:
        ax: Matplotlib axis
        patches: Tensor of patches [num_patches, C, H, W]
        axis: 'x' or 'y'
        zoom: Zoom factor for patch thumbnails
        highlighted_indices: Set of indices to highlight (on the alignment path)
    """
    highlighted_indices = highlighted_indices or set()
    
    for i, patch in enumerate(patches):
        img = patch_to_image(patch)
        
        # Rotate 90 degrees for y-axis to align with vertical direction
        if axis == 'y':
            img = np.rot90(img, k=1)
        
        # Add border for highlighted patches
        if i in highlighted_indices:
            # Add red border by padding
            border_width = 2
            bordered = np.ones((img.shape[0] + 2*border_width, 
                               img.shape[1] + 2*border_width, 3), dtype=np.float32)
            bordered[:, :, 0] = 1.0  # Red
            bordered[:, :, 1] = 0.0  # Green
            bordered[:, :, 2] = 0.0  # Blue
            bordered[border_width:-border_width, border_width:-border_width] = img
            img = bordered
            current_zoom = zoom * 1.2  # Make highlighted patches slightly larger
        else:
            current_zoom = zoom
        
        imagebox = OffsetImage(img, zoom=current_zoom)
        
        if axis == 'x':
            ab = AnnotationBbox(imagebox, (i, 0),
                                xybox=(0, -20),
                                xycoords=('data', 'axes fraction'),
                                boxcoords="offset points",
                                box_alignment=(0.5, 1.0),
                                frameon=False)
        else:  # y axis
            ab = AnnotationBbox(imagebox, (0, i),
                                xybox=(-20, 0),
                                xycoords=('axes fraction', 'data'),
                                boxcoords="offset points",
                                box_alignment=(1.0, 0.5),
                                frameon=False)
        
        ax.add_artist(ab)


def visualize_similarity_with_patches(
    similarity_matrix: torch.Tensor,
    image1_path: str,
    image2_path: str,
    patches_a: torch.Tensor,
    patches_b: torch.Tensor,
    path: list,
    save_path: str,
    title: str = "Cosine Similarity Matrix",
    show: bool = True,
    cmap: str = 'hot',
    patch_zoom: float = 0.3
):
    """
    Visualize similarity matrix heatmap with sliding window patches on axes.
    
    Args:
        similarity_matrix: Tensor of shape [N, M]
        image1_path: Path to first image (for title)
        image2_path: Path to second image (for title)
        patches_a: Patches from image 1 [N, C, H, W]
        patches_b: Patches from image 2 [M, C, H, W]
        path: Optional alignment path to highlight patches
        save_path: Path to save the figure
        title: Plot title
        show: Whether to show the plot
        cmap: Colormap for heatmap
        patch_zoom: Zoom factor for patch images on axes
    """
    sim_np = similarity_matrix.cpu().numpy()
    num_patches_y, num_patches_x = sim_np.shape
    
    # Extract indices on the alignment path
    path_y_indices = set()
    path_x_indices = set()
    if path:
        for (y, x) in path:
            path_y_indices.add(y)
            path_x_indices.add(x)
    
    # Calculate figure size based on number of patches
    fig_width = max(14, num_patches_x * 0.8 + 4)
    fig_height = max(12, num_patches_y * 0.8 + 4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot heatmap
    im = ax.imshow(sim_np, cmap=cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    
    # Add grid lines
    y_positions = np.arange(num_patches_y + 1) - 0.5
    ax.hlines(y_positions, -0.5, num_patches_x - 0.5, colors='gray', 
              linewidths=0.5, alpha=0.3)
    x_positions = np.arange(num_patches_x + 1) - 0.5
    ax.vlines(x_positions, -0.5, num_patches_y - 0.5, colors='gray', 
              linewidths=0.5, alpha=0.3)
    
    # Add patch images to axes
    add_patch_images_to_axis(ax, patches_b, axis='x', zoom=patch_zoom, 
                             highlighted_indices=path_x_indices)
    add_patch_images_to_axis(ax, patches_a, axis='y', zoom=patch_zoom,
                             highlighted_indices=path_y_indices)
    
    # Set labels
    ax.set_xticks(range(num_patches_x))
    ax.set_yticks(range(num_patches_y))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_xlabel(f'Image 2 Patches ({os.path.basename(image2_path)})', fontsize=10)
    ax.set_ylabel(f'Image 1 Patches ({os.path.basename(image1_path)})', fontsize=10)
    ax.set_title(title, fontsize=12)
    
    # Adjust layout to make room for patches
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_score_matrix_with_patches_and_path(
    score_matrix: torch.Tensor,
    similarity_matrix: torch.Tensor,
    path: list,
    image1_path: str,
    image2_path: str,
    patches_a: torch.Tensor,
    patches_b: torch.Tensor,
    save_path: str,
    title: str = "NW Alignment Score Matrix",
    show: bool = True,
    cmap: str = 'viridis',
    patch_zoom: float = 0.3
):
    """
    Visualize NW score matrix with alignment path and sliding window patches on axes.
    Patches on the alignment path are highlighted with red borders.
    
    Args:
        score_matrix: NW score matrix of shape [N, M]
        similarity_matrix: Original similarity matrix [N, M]
        path: List of (y, x) tuples representing alignment path
        image1_path: Path to first image (for title)
        image2_path: Path to second image (for title)
        patches_a: Patches from image 1 [N, C, H, W]
        patches_b: Patches from image 2 [M, C, H, W]
        save_path: Path to save figure
        title: Plot title
        show: Whether to show the plot
        cmap: Colormap
        patch_zoom: Zoom factor for patch images
    """
    score_np = score_matrix.cpu().numpy()
    num_patches_y, num_patches_x = score_np.shape
    
    # Extract indices on the alignment path
    path_y_indices = set()
    path_x_indices = set()
    if path:
        for (y, x) in path:
            path_y_indices.add(y)
            path_x_indices.add(x)
    
    # Calculate figure size based on number of patches
    fig_width = max(14, num_patches_x * 0.8 + 4)
    fig_height = max(12, num_patches_y * 0.8 + 4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot heatmap
    im = ax.imshow(score_np, cmap=cmap, aspect='auto', origin='lower')
    
    # Overlay alignment path
    if path and len(path) > 0:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color='red', linewidth=3, alpha=0.8, 
                label='Alignment Path')
        ax.scatter(path_x, path_y, color='white', s=40, edgecolors='red', 
                   linewidths=2, zorder=5)
        ax.legend(loc='lower right', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Alignment Score', shrink=0.8)
    
    # Add grid lines
    y_positions = np.arange(num_patches_y + 1) - 0.5
    ax.hlines(y_positions, -0.5, num_patches_x - 0.5, colors='gray', 
              linewidths=0.5, alpha=0.3)
    x_positions = np.arange(num_patches_x + 1) - 0.5
    ax.vlines(x_positions, -0.5, num_patches_y - 0.5, colors='gray', 
              linewidths=0.5, alpha=0.3)
    
    # Add patch images to axes (highlighted patches on path have red borders)
    add_patch_images_to_axis(ax, patches_b, axis='x', zoom=patch_zoom,
                             highlighted_indices=path_x_indices)
    add_patch_images_to_axis(ax, patches_a, axis='y', zoom=patch_zoom,
                             highlighted_indices=path_y_indices)
    
    # Set labels
    ax.set_xticks(range(num_patches_x))
    ax.set_yticks(range(num_patches_y))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    ax.set_xlabel(f'Image 2 Patches ({os.path.basename(image2_path)})', fontsize=10)
    ax.set_ylabel(f'Image 1 Patches ({os.path.basename(image1_path)})', fontsize=10)
    ax.set_title(title, fontsize=12)
    
    # Adjust layout to make room for patches
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved score matrix visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_aligned_patches(
    path: list,
    patches_a: torch.Tensor,
    patches_b: torch.Tensor,
    similarity_matrix: torch.Tensor,
    image1_path: str,
    image2_path: str,
    save_path: str,
    title: str = "Aligned Patch Pairs",
    show: bool = True,
    max_pairs: int = 200
):
    """
    Visualize the aligned patch pairs side by side.
    Shows which patches from image1 are aligned with which patches from image2.
    
    Args:
        path: List of (y, x) tuples representing alignment path
        patches_a: Patches from image 1 [N, C, H, W]
        patches_b: Patches from image 2 [M, C, H, W]
        similarity_matrix: Similarity scores [N, M]
        image1_path: Path to first image (for title)
        image2_path: Path to second image (for title)
        save_path: Path to save figure
        title: Plot title
        show: Whether to show the plot
        max_pairs: Maximum number of pairs to show
    """
    if not path:
        print("No alignment path to visualize")
        return
    
    # Filter path to get unique patch pairs (remove "doubled" patches)
    # We only keep steps where BOTH indices increase (diagonal moves)
    # This prevents showing the same patch index multiple times
    aligned_pairs = []
    last_y, last_x = -1, -1
    
    for y, x in path:
        # Check if both indices are strictly greater than the last included ones
        # This filters out vertical/horizontal moves (gaps) where one index repeats
        if y > last_y and x > last_x:
            similarity = similarity_matrix[y, x].item()
            aligned_pairs.append((y, x, similarity))
            last_y, last_x = y, x
    
    # Limit to max_pairs
    if len(aligned_pairs) > max_pairs:
        print(f"Path length {len(aligned_pairs)} exceeds max_pairs {max_pairs}, truncating visualization.")
        aligned_pairs = aligned_pairs[:max_pairs]
    
    num_pairs = len(aligned_pairs)
    
    # Create figure: 2 rows (image1 patches, image2 patches) x num_pairs columns
    # Dynamically adjust width
    fig_width = max(10, num_pairs * 1.5)
    fig, axes = plt.subplots(2, num_pairs, figsize=(fig_width, 4))
    
    # Consistency for single pair case
    if num_pairs == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    fig.suptitle(f'{title}\n{os.path.basename(image1_path)} ↔ {os.path.basename(image2_path)}', 
                 fontsize=12)
    
    for i, (y_idx, x_idx, sim) in enumerate(aligned_pairs):
        # Image 1 patch (top row)
        img1_patch = patch_to_image(patches_a[y_idx])
        
        axes[0, i].imshow(img1_patch)
        axes[0, i].axis('off')
        # Always show index for P1
        axes[0, i].set_title(f'P1[{y_idx}]', fontsize=8)
        
        # Image 2 patch (bottom row)
        img2_patch = patch_to_image(patches_b[x_idx])
        
        # Always show index and similarity for P2
        axes[1, i].imshow(img2_patch)
        axes[1, i].axis('off')
        # Display similarity score clearly
        axes[1, i].set_title(f'P2[{x_idx}]\nsim: {sim:.2f}', fontsize=8)
    
    # Add row labels
    axes[0, 0].set_ylabel('Image 1', fontsize=10)
    axes[1, 0].set_ylabel('Image 2', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved aligned patches visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute similarity matrix and NW alignment between two images'
    )
    parser.add_argument('--image1', '-i1', type=str, required=True,
                        help='Path to the first image')
    parser.add_argument('--image2', '-i2', type=str, required=True,
                        help='Path to the second image')
    parser.add_argument('--output-dir', '-o', type=str, 
                        default='../Results/SimilarityMatrices',
                        help='Directory to save results')
    parser.add_argument('--gap-penalty', '-g', type=float, default=DEFAULT_GAP_PENALTY,
                        help=f'Gap penalty for NW alignment (default: {DEFAULT_GAP_PENALTY})')
    parser.add_argument('--match-score', type=float, default=DEFAULT_MATCH_SCORE,
                        help=f'Match score for traceback (default: {DEFAULT_MATCH_SCORE})')
    parser.add_argument('--mismatch-score', type=float, default=DEFAULT_MISMATCH_SCORE,
                        help=f'Mismatch score for traceback (default: {DEFAULT_MISMATCH_SCORE})')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Similarity threshold (values below are set to 0)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save)')
    parser.add_argument('--device', type=str, default=device,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--weights', '-w', type=str, default=None,
                        help='Path to model weights file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image1):
        raise FileNotFoundError(f"Image not found: {args.image1}")
    if not os.path.exists(args.image2):
        raise FileNotFoundError(f"Image not found: {args.image2}")
    
    # Set device
    use_device = args.device
    if use_device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        use_device = 'cpu'
    
    print("=" * 60)
    print("Similarity Matrix and NW Alignment Visualization")
    print("=" * 60)
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print(f"Device: {use_device}")
    print(f"Window size: {window_size}")
    print(f"Vector size: {vector_size}")
    print(f"Gap penalty: {args.gap_penalty}")
    print("=" * 60)
    
    # Load images
    print("\n[1/5] Loading images...")
    image1 = load_image(args.image1).to(use_device)
    image2 = load_image(args.image2).to(use_device)
    print(f"  Image 1 shape: {image1.shape}")
    print(f"  Image 2 shape: {image2.shape}")
    
    # Create embedding model
    print("\n[2/5] Creating embedding model...")
    model = EmbeddingModel(
        window_size=window_size,
        stride=window_size,  # Now uses calculated overlap stride
        vector_size=vector_size,
        device=device,
        # OPTIMIZATION 1 & 3: Enable BiLSTM and Positional Encoding
        use_bilstm=use_bilstm,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_type=positional_encoding_type,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout
    ).to(use_device)
    
    # Load weights if provided
    if args.weights:
        if os.path.exists(args.weights):
            print(f"  Loading weights from: {args.weights}")
            state_dict = torch.load(args.weights, map_location=use_device)
            # Handle different state dict formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            # Filter out incompatible keys
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() 
                         if k in model_dict and model_dict[k].shape == v.shape}
            model.load_state_dict(state_dict, strict=False)
            print("  Weights loaded successfully")
        else:
            print(f"  Warning: Weights file not found: {args.weights}")
    
    model.eval()
    
    # Compute embeddings
    print("\n[3/5] Computing patch embeddings...")
    with torch.no_grad():
        embeddings_a = model(image1, show_dims=True)
        embeddings_b = model(image2, show_dims=True)
        patches_a = sliding_window(image1, window_size, window_size).squeeze(0)
        patches_b = sliding_window(image2, window_size, window_size).squeeze(0)
    
    # Flip patches vertically to match heatmap orientation (origin='lower')
    patches_a = torch.flip(patches_a, dims=[1])
    patches_b = torch.flip(patches_b, dims=[1])

    #normalize embeddings for better similarity visualization
    embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

    print(f"  Embeddings A shape: {embeddings_a.shape}")
    print(f"  Embeddings B shape: {embeddings_b.shape}")
    print(f"  Patches A shape: {patches_a.shape}")
    print(f"  Patches B shape: {patches_b.shape}")
    
    # Compute similarity matrix
    print("\n[4/5] Computing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity_matrix(embeddings_a, embeddings_b)
    similarity_matrix = similarity_matrix.squeeze(0)  # Remove batch dimension
    
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  Min similarity: {similarity_matrix.min().item():.4f}")
    print(f"  Max similarity: {similarity_matrix.max().item():.4f}")
    print(f"  Mean similarity: {similarity_matrix.mean().item():.4f}")
    
    # Apply threshold if specified
    if args.threshold is not None:
        print(f"  Applying threshold: {args.threshold}")
        sim_for_alignment = torch.where(
            similarity_matrix > args.threshold,
            1.0,
            torch.tensor(0.0, device=use_device)
        )
    else:
        sim_for_alignment = similarity_matrix
    
    # Compute NW score matrix and alignment path
    print("\n[5/5] Computing NW alignment...")
    H_full = compute_nw_score_matrix(sim_for_alignment, gap_penalty=args.gap_penalty)
    H_vis = H_full[1:, 1:]  # Remove boundary row/column for visualization
    
    print(f"  Score matrix shape: {H_vis.shape}")
    print(f"  Final alignment score: {H_full[-1, -1].item():.4f}")
    
    # Compute traceback path
    path, _ = compute_traceback_path(
        H_vis,
        sim_for_alignment,
        match_score=3,
        miss_score=-27,
        gap_penalty=-10
    )
    print(f"  Alignment path length: {len(path)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    base1 = os.path.splitext(os.path.basename(args.image1))[0]
    base2 = os.path.splitext(os.path.basename(args.image2))[0]
    
    # Visualize raw similarity matrix with patches on axes
    print("\nGenerating visualizations...")
    sim_save_path = os.path.join(args.output_dir, f'similarity_{base1}_vs_{base2}.png')
    visualize_similarity_with_patches(
        similarity_matrix,
        args.image1,
        args.image2,
        patches_a,
        patches_b,
        path=path,  # Pass path to highlight aligned patches
        save_path=sim_save_path,
        title=f'Cosine Similarity (Raw): {base1} vs {base2}',
        show=not args.no_show
    )
    
    # Visualize thresholded similarity matrix (if threshold was applied)
    if args.threshold is not None:
        sim_thresh_save_path = os.path.join(args.output_dir, f'similarity_thresholded_{base1}_vs_{base2}.png')
        visualize_similarity_with_patches(
            sim_for_alignment,
            args.image1,
            args.image2,
            patches_a,
            patches_b,
            path=path,
            save_path=sim_thresh_save_path,
            title=f'Cosine Similarity (Threshold > {args.threshold}): {base1} vs {base2}',
            show=not args.no_show
        )
    
    # Visualize score matrix with alignment path and patches
    score_save_path = os.path.join(args.output_dir, f'alignment_{base1}_vs_{base2}.png')
    visualize_score_matrix_with_patches_and_path(
        H_vis,
        similarity_matrix,
        path,
        args.image1,
        args.image2,
        patches_a,
        patches_b,
        save_path=score_save_path,
        title=f'NW Alignment: {base1} vs {base2} (Gap={args.gap_penalty})',
        show=not args.no_show
    )
    
    # Visualize aligned patch pairs
    aligned_save_path = os.path.join(args.output_dir, f'aligned_patches_{base1}_vs_{base2}.png')
    visualize_aligned_patches(
        path,
        patches_a,
        patches_b,
        similarity_matrix,
        args.image1,
        args.image2,
        save_path=aligned_save_path,
        title='Aligned Patch Pairs',
        show=not args.no_show
    )
    
    
    print("\n" + "=" * 60)
    print("Done! Results saved to:", args.output_dir)
    print("=" * 60)
    
    return similarity_matrix, H_vis, path


if __name__ == "__main__":
    main()


# python test_scripts/AlignmentTestVisualize.py --image1 DataSet/Synthetic_Arabic/images/img1_1.png --image2 DataSet/Synthetic_Arabic/images/img2_1.png --output-dir Results/AlignmentVisualize --threshold 0.5
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathExtractor import extract_traceback_path



def visualize_paths(alignment_path, smith_path=None, epoch=0, batch_idx=0):
    """
    Visualize alignment path and smith path side by side.

    Args:
        alignment_path (torch.Tensor): Smoothed path mask from alignment output.
        smith_path (torch.Tensor): Smoothed path mask from Smith-Waterman alignment.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
    """
    alignment_np = alignment_path.cpu().detach().numpy()
    if smith_path is not None:
        smith_np = smith_path.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(alignment_np, interpolation='nearest')
    axes[0].set_title(f"Alignment Path (Epoch {epoch}, Batch {batch_idx})")
    axes[0].axis("off")

    if smith_path is not None:
        axes[1].imshow(smith_np, interpolation='nearest')
        axes[1].set_title(f"Smith-Waterman Path (Epoch {epoch}, Batch {batch_idx})")
        axes[1].axis("off")

    plt.show()


def visualize_heatmap(smith_path, image_path):
    """
    Visualizes the Smith-Waterman alignment path as a heatmap.

    Args:
        smith_path (torch.Tensor): Smoothed path mask from Smith-Waterman alignment.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index.
    """
    smith_np = smith_path.cpu().detach().numpy()

    # Normalize the path for heatmap visualization
    # smith_np = (smith_np - smith_np.min()) / (smith_np.max() - smith_np.min() + 1e-6)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(smith_np, cmap='jet', linewidths=0.1, linecolor='black')
    plt.title(f"Smith-Waterman Path Heatmap")
    plt.axis("off")

    # Save heatmap
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_heatmaps(alignment_matrix_data, smith_matrix_data, image_path, patches_y=None, patches_x=None, zoom_factor=0.2, y_axis_x_offset=-30, x_axis_y_offset=25):
    """
    Visualizes heatmaps for both alignment and Smith-Waterman paths.
    Optionally displays image patches along the axes if provided.

    Args:
        alignment_matrix_data (torch.Tensor): Alignment output data (before path extraction).
        smith_matrix_data (torch.Tensor): Smith-Waterman data (before path extraction).
        image_path (str): Path to save the heatmap image.
        patches_y (list or torch.Tensor, optional): List of image patches for the Y-axis.
        patches_x (list or torch.Tensor, optional): List of image patches for the X-axis.
        zoom_factor (float): Zoom factor for the axis image patches.
        y_axis_x_offset (float): Offset for Y-axis patch images from the axis.
        x_axis_y_offset (float): Offset for X-axis patch images from the axis.
    """
    # Process matrices to get paths for heatmap display
    alignment_path_np = alignment_matrix_data.cpu().numpy()
    smith_path_np = smith_matrix_data.cpu().numpy()

    # Plot heatmaps side by side
    # Reduced figsize from (160, 80) to a more manageable size.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    matrices_to_plot = [alignment_path_np, smith_path_np]
    titles = ["Alignment Path Heatmap", "Smith-Waterman Path Heatmap"]

    for idx, ax in enumerate(axes):
        matrix_data = matrices_to_plot[idx]
        use_img_labels = patches_y is not None and patches_x is not None

        # annot=True is very slow for large matrices. It's disabled for performance.
        # If your matrices are small (e.g., < 20x20), you can re-enable it.
        sns.heatmap(matrix_data, cmap='jet', linewidths=0.1, linecolor='black',
                     ax=ax, annot=False, fmt=".3f",
                     xticklabels=False if use_img_labels else list(range(matrix_data.shape[1])),
                     yticklabels=False if use_img_labels else list(range(matrix_data.shape[0])))
        # ax.set_title(titles[idx]) # Removed: Title will be placed at the bottom

        if use_img_labels:
            # Y-axis image labels (patches_y)
            if len(patches_y) == matrix_data.shape[0]:
                for k in range(matrix_data.shape[0]):
                    patch_tensor = patches_y[k]
                    img = patch_tensor.cpu().permute(1, 2, 0).numpy()
                    # Ensure image is in displayable range (e.g., 0-1 for float)
                    img = np.clip(img, 0, 1) if img.max() > 1.0 and img.dtype == np.float32 else img
                    oi = OffsetImage(img, zoom=zoom_factor)
                    # Position to the left of the y-axis ticks
                    ab = AnnotationBbox(oi, (0, k + 0.5),
                                        xybox=(y_axis_x_offset, 0), frameon=False,
                                        xycoords='data', boxcoords="offset points", pad=0.1)
                    ax.add_artist(ab)
            # X-axis image labels (patches_x)
            if len(patches_x) == matrix_data.shape[1]:
                for k in range(matrix_data.shape[1]):
                    patch_tensor = patches_x[k]
                    img = patch_tensor.cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1) if img.max() > 1.0 and img.dtype == np.float32 else img
                    oi = OffsetImage(img, zoom=zoom_factor)
                    # Position below the x-axis ticks
                    ab = AnnotationBbox(oi, (k + 0.5, 0),
                                        xybox=(0, x_axis_y_offset), frameon=False,
                                        xycoords='data', boxcoords="offset points", pad=0.1)
                    ax.add_artist(ab)
        # Place title at the bottom of the subplot
        ax.text(0.5, -0.05, titles[idx], transform=ax.transAxes,
                ha="center", va="top", fontsize=10) # Reverted y-coordinate

    # The following lines are redundant as titles are set in the loop
    # axes[0].set_title(f"Alignment Path Heatmap")
    # axes[1].set_title(f"Smith-Waterman Path Heatmap")
    if use_img_labels:
        fig.subplots_adjust(left=0.4, bottom=0.35) # Increased bottom margin for X-axis patches

    # Save heatmap
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    # plt.show()

def visualize_single_heatmap_with_text_labels(matrix_data, x_labels, y_labels, image_path, title):
    """
    Visualizes a single heatmap with text labels on axes.

    Args:
        matrix_data (torch.Tensor or np.array): The matrix to plot.
        x_labels (list of str): List of strings for x-axis ticks.
        y_labels (list of str): List of strings for y-axis ticks.
        image_path (str): Path to save the heatmap image.
        title (str): Title for the heatmap.
    """
    if torch.is_tensor(matrix_data):
        matrix_data = matrix_data.cpu().numpy()

    # Dynamically adjust figsize based on label count; ensure minimum size
    fig_width = max(12, len(x_labels) * 0.35)
    fig_height = max(10, len(y_labels) * 0.35)
    plt.figure(figsize=(fig_width, fig_height))

    ax = sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap='viridis',
                     xticklabels=x_labels, yticklabels=y_labels,
                     linewidths=.5, cbar=True)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='x', labelrotation=90, labelsize=8) # Rotate x-labels, adjust size
    ax.tick_params(axis='y', labelrotation=0, labelsize=8)  # Adjust y-label size
    plt.tight_layout()
    plt.savefig(image_path, dpi=150) # DPI can be adjusted based on needs
    plt.close()


def visualize_dual_char_heatmaps(matrix1_data, matrix2_data, x_labels, y_labels, image_path, title1, title2):
    """
    Visualizes two heatmaps side-by-side with text labels on axes.

    Args:
        matrix1_data (torch.Tensor or np.array): The first matrix to plot (e.g., scores).
        matrix2_data (torch.Tensor or np.array): The second matrix to plot (e.g., path).
        x_labels (list of str): List of strings for x-axis ticks.
        y_labels (list of str): List of strings for y-axis ticks.
        image_path (str): Path to save the heatmap image.
        title1 (str): Title for the first heatmap.
        title2 (str): Title for the second heatmap.
    """
    if torch.is_tensor(matrix1_data):
        matrix1_data = matrix1_data.cpu().numpy()
    if torch.is_tensor(matrix2_data):
        matrix2_data = matrix2_data.cpu().numpy()

    # Dynamically adjust figsize based on label count; ensure minimum size
    # Consider width for two plots
    fig_width = max(24, len(x_labels) * 0.35 * 2) # Double width for two plots
    fig_height = max(10, len(y_labels) * 0.35)
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    matrices_to_plot = [matrix1_data, matrix2_data]
    titles = [title1, title2]
    # Use fmt=".0f" for both, assuming integer scores and binary path
    fmts = [".0f", ".0f"] 

    for idx, ax in enumerate(axes):
        # annot=True is very slow for large matrices (e.g., >30x30).
        # It is disabled here for performance.
        sns.heatmap(matrices_to_plot[idx], annot=False, fmt=fmts[idx], cmap='viridis',
                     xticklabels=x_labels, yticklabels=y_labels,
                     linewidths=.5, cbar=True, ax=ax)
        ax.set_title(titles[idx], fontsize=16)
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
        ax.tick_params(axis='y', labelrotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()
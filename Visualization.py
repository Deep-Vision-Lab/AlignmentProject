import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathExtractor import *
from saveDATA import *
from embeddingModel import sliding_window
import os


def save_debug_visualizations(model, image1, image2, tokens_a, tokens_b, NWTextTensor, diffNWimage, 
                               DiffNWAlgo, loss_type, epoch, batch_idx):
    """
    Save debug visualizations for the training process.
    
    Args:
        model: The embedding model
        image1, image2: Input images
        tokens_a, tokens_b: Token embeddings
        NWTextTensor: Text-based NW tensor
        diffNWimage: Image-based NW tensor
        DiffNWAlgo: The DiffNW algorithm instance
        loss_type: Type of loss function used
        epoch: Current epoch number
        batch_idx: Current batch index
    """
    # Prepare directories for saving visualizations
    vectors_epoch_dir = f'TrainResults/{loss_type}/VectorsPerEpoch/{model.model_arch}/Epoch_{epoch+1}'
    matrices_epoch_dir = f'TrainResults/{loss_type}/ScoreMatricesPerEpoch/{model.model_arch}/Epoch_{epoch+1}'
    os.makedirs(vectors_epoch_dir, exist_ok=True)
    os.makedirs(matrices_epoch_dir, exist_ok=True)

    # Clone tensors for visualization to avoid affecting gradients
    debug_image1 = image1.detach().cpu()
    debug_image2 = image2.detach().cpu()
    debug_tokens_a = tokens_a.detach().cpu()
    debug_tokens_b = tokens_b.detach().cpu()
    debug_diffNWText = NWTextTensor.detach().cpu()
    debug_diffNWimage = diffNWimage.detach().cpu()

    # debug_diffNWText is only available during training if calc_cosine=False in Alignment
    if hasattr(DiffNWAlgo, 'similarity_matrix'):
        debug_NWTextSimilar = DiffNWAlgo.similarity_matrix.clone().detach()
    else:
        debug_NWTextSimilar = None

    # original_text1_batch and original_text2_batch are only available during training if calc_cosine=False in Alignment
    if hasattr(DiffNWAlgo, 'original_text1_batch') and hasattr(DiffNWAlgo, 'original_text2_batch'):
        original_text1_batch = DiffNWAlgo.original_text1_batch
        original_text2_batch = DiffNWAlgo.original_text2_batch
    else:
        original_text1_batch = None
        original_text2_batch = None

    # Save heatmap visualizations
    saveHeatmapPlots(model, debug_image1, debug_image2, 
                    debug_tokens_a, debug_tokens_b, 
                    vectors_epoch_dir, epoch, batch_idx,
                    debug_diffNWimage, 
                    debug_diffNWText, 
                    debug_NWTextSimilar, 
                    matrices_epoch_dir, original_text1_batch,
                    original_text2_batch)

    del debug_image1, debug_image2
    del debug_tokens_a, debug_tokens_b
    del debug_diffNWimage, debug_diffNWText
    del vectors_epoch_dir, matrices_epoch_dir
    del original_text1_batch, original_text2_batch
    if debug_NWTextSimilar is not None:
        del debug_NWTextSimilar


def saveHeatmapPlots(model, image1, image2, tokens_a, tokens_b, vectors_epoch_dir, epoch, batch_idx, 
                    debug_diffNWimage, debug_diffNWText, debug_NWTextSimilar,
                    matrices_epoch_dir, original_text1_batch, original_text2_batch):
    for i in range(image1.size(0)): # Iterate through items in the current batch
        # Save token vectors
        # tokens_a and tokens_b are already flipped and on the correct device
        print_elements(tokens_a[i], f'{vectors_epoch_dir}/tokens_a_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.xlsx')
        print_elements(tokens_b[i], f'{vectors_epoch_dir}/tokens_b_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.xlsx')
        # Generate patches for visualization
        image1_i = image1[i].unsqueeze(0)
        image2_i = image2[i].unsqueeze(0)
        model_window_size = model.window_size
        model_stride = model.stride
        windows_img1 = sliding_window(image1_i, model_window_size, model_stride).squeeze(0)
        windows_img2 = sliding_window(image2_i, model_window_size, model_stride).squeeze(0)

        y_heatmap = torch.flip(windows_img1, dims=[0]) # From image1, for Y-axis
        x_heatmap = torch.flip(windows_img2, dims=[0]) # From image2, for X-axis

        # Extract paths using diff_NW_Path for visualization
        textPath, _ = NW_Path(debug_diffNWText[i:i+1], debug_diffNWText[i:i+1], match_score=7, miss_score=-3, gap_penalty=-1)
        imagePath, _ = diff_NW_Path(debug_diffNWimage[i:i+1], debug_diffNWimage[i:i+1], match_score=7, miss_score=-3, gap_penalty=-1)
        
        # Visualize paths instead of raw matrices
        visualize_heatmaps(
            imagePath[0],  # Use extracted path
            textPath[0],   # Use extracted path
            f"{matrices_epoch_dir}/heatmaps_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.png",
            patches_y=y_heatmap,
            patches_x=x_heatmap
        )

        # Only run the following if debug_NWTextSimilar is not None
        if debug_NWTextSimilar is not None:
            # New visualization: Raw Smith-Waterman matrix (before interpolation)
            # with original character sequences as axes. seq1 on X, seq2 on Y.
            debug_NWTextSimilar_i = debug_NWTextSimilar[i] # Shape [H_orig, W_orig]
            original_text1_batch_i = original_text1_batch[i] # string for seq1
            original_text2_batch_i = original_text2_batch[i] # string for seq2

            # Use NW_Path instead of makeTracerouteMatrixBinary
            smith_original_path_tensor, _ = NW_Path(debug_NWTextSimilar_i.unsqueeze(0), 
                                                   debug_NWTextSimilar_i.unsqueeze(0),
                                                   match_score=7, miss_score=-3, gap_penalty=-1)
            smith_original_path_np = smith_original_path_tensor[0].cpu().numpy()

            # NW matrix has text1 along rows, text2 along columns.
            # To put text1 (seq1) on X-axis and text2 (seq2) on Y-axis, we need to transpose.
            scores_matrix_to_plot = debug_NWTextSimilar_i.T
            path_matrix_to_plot = torch.tensor(smith_original_path_np).T

            # Labels for axes: X-axis for seq1 (original_text1), Y-axis for seq2 (original_text2)
            x_char_labels = ['ø'] + list(original_text1_batch_i.replace(" ", "")) # 'ø' for the initial empty string state
            y_char_labels = ['ø'] + list(original_text2_batch_i.replace(" ", ""))
            filename_char_level_smith_dual = f"{matrices_epoch_dir}/raw_char_smith_scores_path_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.png"
            visualize_dual_char_heatmaps(
                scores_matrix_to_plot,
                path_matrix_to_plot,
                x_labels=x_char_labels,
                y_labels=y_char_labels,
                image_path=filename_char_level_smith_dual,
                title1=f"Raw NW Scores (Seq1 vs Seq2) - Item {i}",
                title2=f"Raw NW Path (Seq1 vs Seq2) - Item {i}"
            )
            del debug_NWTextSimilar_i, original_text1_batch_i, original_text2_batch_i
            del smith_original_path_tensor, smith_original_path_np
            del scores_matrix_to_plot, path_matrix_to_plot
            del x_char_labels, y_char_labels, filename_char_level_smith_dual
        del windows_img1, windows_img2, y_heatmap, x_heatmap
        del textPath, imagePath
        del image1_i, image2_i




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
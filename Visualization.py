import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch
import torch.nn.functional as F

from saveDATA import *
from Parameters import *
from pathExtractor import *
from embeddingModel import *



def save_debug_visualizations(model, image1, image2, tokens_a, tokens_b, 
                            NWTextTensor, diffNWimage,
                            original_diffNWImageSimilar, interpolated_diffNWImageSimilar,
                            original_NWTextSimilar, interpolated_NWTextSimilar,
                            loss_type, epoch, batch_idx):
    # Prepare directories for saving visualizations
    loss_dir = f'TrainResults/{loss_type}'
    os.makedirs(loss_dir, exist_ok=True)

    vectors_epoch_dir = f'{loss_dir}/SimilarityMatricesPerEpoch/{model.model_arch}/Epoch_{epoch}'
    matrices_epoch_dir = f'{loss_dir}/ScoreMatricesPerEpoch/{model.model_arch}/Epoch_{epoch}'
    os.makedirs(vectors_epoch_dir, exist_ok=True)
    os.makedirs(matrices_epoch_dir, exist_ok=True)

    # Clone tensors for visualization to avoid affecting gradients
    debug_image1 = image1.detach().cpu()
    debug_image2 = image2.detach().cpu()
    debug_NWText = NWTextTensor.detach().cpu()
    debug_diffNWimage = diffNWimage.detach().cpu()
    debug_original_diffNWImageSimilar = original_diffNWImageSimilar.detach().cpu()
    debug_interpolated_diffNWImageSimilar = interpolated_diffNWImageSimilar.detach().cpu()
    debug_original_NWTextSimilar = original_NWTextSimilar.detach().cpu()
    debug_interpolated_NWTextSimilar = interpolated_NWTextSimilar.detach().cpu()

    # Save heatmap visualizations
    saveHeatmapPlots(
        model, debug_image1, debug_image2, vectors_epoch_dir, epoch, batch_idx,
        debug_diffNWimage, debug_NWText, 
        debug_original_diffNWImageSimilar, debug_interpolated_diffNWImageSimilar,
        debug_original_NWTextSimilar, debug_interpolated_NWTextSimilar, 
        matrices_epoch_dir
    )

    del debug_image1, debug_image2
    del debug_diffNWimage, debug_NWText
    del vectors_epoch_dir, matrices_epoch_dir
    del debug_original_diffNWImageSimilar, debug_interpolated_diffNWImageSimilar
    del debug_original_NWTextSimilar, debug_interpolated_NWTextSimilar



def saveHeatmapPlots(model, image1, image2, similarity_epoch_dir, epoch, batch_idx, 
                    debug_diffNWimage, debug_NWTextMatrix, 
                    debug_original_diffNWImageSimilar, debug_interpolated_diffNWImageSimilar,
                    debug_original_NWTextSimilar, debug_interpolated_NWTextSimilar, 
                    matrices_epoch_dir):
    for i in range(image1.size(0)): # Iterate through items in the current batch
        similarity_dir_per_batch = f'{similarity_epoch_dir}/{i}'
        os.makedirs(similarity_dir_per_batch, exist_ok=True)
        # Save Image similarity matrix heatmap
        image_similarity = f'{similarity_epoch_dir}/{i}/ImageDomain'
        os.makedirs(image_similarity, exist_ok=True)
        visualize_heatmap(
            debug_original_diffNWImageSimilar[i],
            f"Original Image Similarity Matrix Heatmap",
            f'{image_similarity}/OriginalImageSimilarityMatrixEpoch_{epoch+1}_batch_{batch_idx}_item_{i}.png'
        )
        # Save Image similarity matrix heatmap
        visualize_heatmap(
            debug_interpolated_diffNWImageSimilar[i],
            f"Interpolated Image Similarity Matrix Heatmap",
            f'{image_similarity}/InterpolatedImageSimilarityMatrixEpoch_{epoch+1}_batch_{batch_idx}_item_{i}.png'
        )

        text_similarity = f'{similarity_epoch_dir}/{i}/TextDomain'
        os.makedirs(text_similarity, exist_ok=True)
        # Save Original Text similarity matrix heatmap
        visualize_heatmap(
            debug_original_NWTextSimilar[i],
            f"Original Text Similarity Matrix Heatmap",
            f'{text_similarity}/OriginalTextSimilarityMatrixEpoch_{epoch+1}_batch_{batch_idx}_item_{i}.png'
        )
        # Save Interpolated Text similarity matrix heatmap
        visualize_heatmap(
            debug_interpolated_NWTextSimilar[i],
            f"Interpolated Text Similarity Matrix Heatmap",
            f'{text_similarity}/InterpolatedTextSimilarityMatrixEpoch_{epoch+1}_batch_{batch_idx}_item_{i}.png'
        )

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
        textPath, _ = NW_Path(
            debug_NWTextMatrix[i:i+1], 
            debug_interpolated_NWTextSimilar[i:i+1], 
            match_score=matchScore, 
            miss_score=mismatchScore, 
            gap_penalty=gapScore
        )
        imagePath, _ = diff_NW_Path(
            debug_diffNWimage[i:i+1], 
            debug_interpolated_diffNWImageSimilar[i:i+1], 
            match_score=matchScore, 
            miss_score=mismatchScore, 
            gap_penalty=gapScore
        )
        
        # Visualize paths instead of raw matrices
        VisualizeWithImageAxis(
            ImageScoreMatrix=imagePath[0],
            TextScoreMatrix=textPath[0],
            image_path=f"{matrices_epoch_dir}/HeatmapsEpoch_{epoch+1}_batch_{batch_idx}_item_{i}.png",
            title1="Image NW Path Heatmap",
            title2="Text NW Path Heatmap",
            patches_y=y_heatmap,
            patches_x=x_heatmap
        )
        del windows_img1, windows_img2, y_heatmap, x_heatmap
        del textPath, imagePath
        del image1_i, image2_i



def visualize_paths(ImagePathMatrix, TextPathMatrix=None, epoch=0, batch_idx=0):
    ImagePathMatrix = ImagePathMatrix.cpu().detach().numpy()
    if TextPathMatrix is not None:
        TextPathMatrix = TextPathMatrix.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(ImagePathMatrix, interpolation='nearest')
    axes[0].set_title(f"Image NW Path (Epoch {epoch}, Batch {batch_idx})")
    axes[0].axis("off")

    if TextPathMatrix is not None:
        axes[1].imshow(TextPathMatrix, interpolation='nearest')
        axes[1].set_title(f"Text NW Path (Epoch {epoch}, Batch {batch_idx})")
        axes[1].axis("off")

    plt.show()



def visualize_heatmap(pathMatrix, title,image_path):
    pathMatrix = pathMatrix.cpu().detach().numpy()
    # Plot heatmap
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(pathMatrix, cmap='jet', linewidths=0.1, linecolor='black')
    plt.title(title)
    # Show axis labels only every N elements
    N = 5  # Change this value for different spacing
    x_indices = np.arange(pathMatrix.shape[1])
    y_indices = np.arange(pathMatrix.shape[0])
    x_tick_locs = np.arange(0, pathMatrix.shape[1], N)
    y_tick_locs = np.arange(0, pathMatrix.shape[0], N)
    ax.set_xticks(x_tick_locs + 0.5)
    ax.set_yticks(y_tick_locs + 0.5)
    ax.set_xticklabels(x_indices[x_tick_locs], rotation=0, fontsize=8)
    ax.set_yticklabels(y_indices[y_tick_locs], rotation=0, fontsize=8)

    # Save heatmap
    plt.savefig(image_path, dpi=300, bbox_inches='tight')



def VisualizeWithImageAxis(ImageScoreMatrix, TextScoreMatrix, image_path, title1, title2, patches_y=None, patches_x=None,
                        zoom_factor=0.2, y_axis_x_offset=-30, x_axis_y_offset=25):
    # Process matrices to get paths for heatmap display
    ImageScoreMatrix_np = ImageScoreMatrix.cpu().numpy()
    Textscorematrix_np = TextScoreMatrix.cpu().numpy()

    # Plot heatmaps side by side
    # Reduced figsize from (160, 80) to a more manageable size.
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    matrices_to_plot = [ImageScoreMatrix_np, Textscorematrix_np]
    titles = [title1, title2]

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



def VisualizeWithText(matrix_data, x_labels, y_labels, image_path, title):
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



def VisualizeDualCharHeatmaps(ScoreMatrix, PathMatrix, x_labels, y_labels, image_path, title1, title2):
    if torch.is_tensor(ScoreMatrix):
        ScoreMatrix = ScoreMatrix.cpu().numpy()
    if torch.is_tensor(PathMatrix):
        PathMatrix = PathMatrix.cpu().numpy()

    # Dynamically adjust figsize based on label count; ensure minimum size
    # Consider width for two plots
    fig_width = max(24, len(x_labels) * 0.35 * 2) # Double width for two plots
    fig_height = max(10, len(y_labels) * 0.35)
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    matrices_to_plot = [ScoreMatrix, PathMatrix]
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
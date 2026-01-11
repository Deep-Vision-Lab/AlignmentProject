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


def saveImageTensorAsPNG(tensor, path):
    """
    Save a torch image tensor (C,H,W or H,W or 1,H,W) to a PNG file at `path`.
    Handles tensors on GPU and common float ranges (0-1 or 0-255).
    """
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu()
    arr = np.array(tensor)
    # Handle channel-first (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    # Handle single-channel (H,W) or (1,H,W) -> (H,W)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    # Normalize floats in range [0,1] to [0,255] for uint8, otherwise clip
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        else:
            arr = arr.clip(0, 255)
        arr = arr.astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use matplotlib to save image
    plt.imsave(path, arr, cmap='gray' if arr.ndim == 2 else None)


def saveSlidingWindowsWithOverlap(image, output_path, window_size, title="Sliding Windows (Half Overlap)", cols=8):
    """
    Extract sliding windows from an image with half overlap, flip them, and save as a grid.
    
    Args:
        image: Input image tensor of shape (C, H, W) or (1, C, H, W) or (H, W)
        output_path: Path to save the grid image (individual windows saved in same directory)
        window_size: Size of the sliding window (int or tuple)
        title: Title for the grid image
        cols: Number of columns in the grid
    """
    if torch.is_tensor(image):
        image = image.detach().cpu()
    
    # Ensure image is 4D (B, C, H, W)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif image.ndim == 3:
        image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    # Calculate stride as half the window size (50% overlap)
    if isinstance(window_size, int):
        stride = window_size // 2
    else:
        stride = (window_size[0] // 2, window_size[1] // 2)
    
    # Extract sliding windows
    windows = sliding_window(image, window_size, stride).squeeze(0)  # (num_windows, C, H, W)
    
    # Flip the windows (vertical flip)
    windows = torch.flip(windows, dims=[0])
    
    num_windows = windows.shape[0]
    rows = (num_windows + cols - 1) // cols  # Calculate rows needed
    
    # Create grid figure
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).flatten() if rows > 1 or cols > 1 else [axes]
    
    for i in range(num_windows):
        ax = axes[i]
        window = windows[i]
        
        # Handle different tensor shapes
        if window.ndim == 3 and window.shape[0] in (1, 3):
            # (C, H, W) -> (H, W, C)
            img = window.permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img.squeeze(-1)
        elif window.ndim == 2:
            img = window.numpy()
        else:
            img = window.numpy()
        
        # Normalize if needed
        if np.issubdtype(img.dtype, np.floating):
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(f"W{i}", fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(num_windows, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"{title}\n({num_windows} windows, stride={stride})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)



def saveWindowsAsGrid(windows, output_path, title="Windows Grid", cols=8):
    """
    Save all windows from a tensor as a grid image and also save each window individually.
    
    Args:
        windows: Tensor of shape (num_windows, C, H, W) or (num_windows, H, W)
        output_path: Path to save the grid image (individual windows saved in same directory)
        title: Title for the grid image
        cols: Number of columns in the grid
    """
    if torch.is_tensor(windows):
        windows = windows.detach().cpu()
    
    num_windows = windows.shape[0]
    rows = (num_windows + cols - 1) // cols  # Calculate rows needed
    
    # Create output directory for individual windows
    output_dir = os.path.dirname(output_path)
    windows_dir = os.path.join(output_dir, "individual_windows")
    os.makedirs(windows_dir, exist_ok=True)
    
    # Create grid figure
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).flatten() if rows > 1 or cols > 1 else [axes]
    
    for i in range(num_windows):
        ax = axes[i]
        window = windows[i]
        
        # Handle different tensor shapes
        if window.ndim == 3 and window.shape[0] in (1, 3):
            # (C, H, W) -> (H, W, C)
            img = window.permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img.squeeze(-1)
        elif window.ndim == 2:
            img = window.numpy()
        else:
            img = window.numpy()
        
        # Normalize if needed
        if np.issubdtype(img.dtype, np.floating):
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(f"W{i}", fontsize=8)
        ax.axis('off')
        
        # Save individual window
        individual_path = os.path.join(windows_dir, f"window_{i:04d}.png")
        plt.imsave(individual_path, img, cmap='gray' if img.ndim == 2 else None)
        print(f"Saved window {i} to {individual_path}")
    
    # Hide unused subplots
    for j in range(num_windows, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved windows grid to {output_path}")


def save_debug_visualizations(model, text1, text2, image1, image2, tokens_a, tokens_b, 
                            NWTextTensor, diffNWimage,
                            original_diffNWImageSimilar, interpolated_diffNWImageSimilar,
                            original_NWTextSimilar, interpolated_NWTextSimilar,
                            epoch, batch_idx):
    # Prepare directories for saving visualizations
    loss_dir = f'TrainResults/{loss_type}'
    os.makedirs(loss_dir, exist_ok=True)
    
    # saving image1 and image2.
    lines_dir = f'{loss_dir}/InputImages/{model.model_arch}/Epoch_{epoch}'
    os.makedirs(lines_dir, exist_ok=True)
    for i in range(image1.size(0)):
        lines_dir_per_item = f'{lines_dir}/{i}'
        os.makedirs(lines_dir_per_item, exist_ok=True)
        saveImageTensorAsPNG(image1[i], f'{lines_dir_per_item}/Image1.png')
        saveImageTensorAsPNG(image2[i], f'{lines_dir_per_item}/Image2.png')
        
        # Save sliding windows for image1 and image2
        saveSlidingWindowsWithOverlap(
            image=image1[i],
            output_path=f'{lines_dir_per_item}/Image1_SlidingWindows.png',
            window_size=model.window_size,
            title="Image1 Sliding Windows (Half Overlap)",
            cols=8
        )
        saveSlidingWindowsWithOverlap(
            image=image2[i],
            output_path=f'{lines_dir_per_item}/Image2_SlidingWindows.png',
            window_size=model.window_size,
            title="Image2 Sliding Windows (Half Overlap)",
            cols=8
        )

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
        model, text1,text2, debug_image1, debug_image2, vectors_epoch_dir, epoch, batch_idx,
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



def saveHeatmapPlots(model, text1, text2, image1, image2, similarity_epoch_dir, epoch, batch_idx, 
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
            f'{image_similarity}/OriginalImageSimilarityMatrix.png'
        )
        # Save Image similarity matrix heatmap
        visualize_heatmap(
            debug_interpolated_diffNWImageSimilar[i],
            f"Interpolated Image Similarity Matrix Heatmap",
            f'{image_similarity}/InterpolatedImageSimilarityMatrix.png'
        )

        text_similarity = f'{similarity_epoch_dir}/{i}/TextDomain'
        os.makedirs(text_similarity, exist_ok=True)
        # Save Original Text similarity matrix heatmap
        visualize_heatmap(
            debug_original_NWTextSimilar[i],
            f"Original Text Similarity Matrix Heatmap",
            f'{text_similarity}/OriginalTextSimilarityMatrix.png'
        )
        # Save Interpolated Text similarity matrix heatmap
        visualize_heatmap(
            debug_interpolated_NWTextSimilar[i],
            f"Interpolated Text Similarity Matrix Heatmap",
            f'{text_similarity}/InterpolatedTextSimilarityMatrix.png'
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

        y_text = text1[i]
        x_text = text2[i]

        # Extract paths using diff_NW_Path for visualization
        textPath, _ = diff_NW_Path(
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

        score_matrix_dir_per_batch = f'{matrices_epoch_dir}/{i}/score_matrix'
        os.makedirs(score_matrix_dir_per_batch, exist_ok=True)
        # Visualize scoreMatrices instead of raw matrices
        VisualizeWithImageAxis(
            ImageMatrix=debug_diffNWimage[i],
            TextMatrix=debug_NWTextMatrix[i],
            image_path=f"{score_matrix_dir_per_batch}/ScoreMatrix.png",
            title1="Image NW score matrix Heatmap",
            title2="Text NW score matrix Heatmap",
            patches_right_y=y_text,
            patches_right_x=x_text
        )
        
        # Visualize the distance between score matrices
        score_matrix_distance = torch.abs(debug_diffNWimage[i].squeeze() - debug_NWTextMatrix[i].squeeze())
        visualize_heatmap(
            score_matrix_distance,
            "Distance Between Image & Text Score Matrices",
            f"{score_matrix_dir_per_batch}/ScoreMatrixDistance.png"
        )
        # Save sum of score_matrix_distance to txt in score_matrix directory
        try:
            score_sum = score_matrix_distance.sum().item()
            with open(f"{score_matrix_dir_per_batch}/ScoreMatrixDistanceSum.txt", "w") as f:
                f.write(f"{score_sum}\n")
        except Exception as e:
            print(f"Failed to write ScoreMatrixDistanceSum.txt: {e}")
        
        del score_matrix_distance


        path_dir_per_batch = f'{matrices_epoch_dir}/{i}/path'
        os.makedirs(path_dir_per_batch, exist_ok=True)
        # Visualize scoreMatrices instead of raw matrices
        VisualizeWithImageAxis(
            ImageMatrix=imagePath[0],
            TextMatrix=textPath[0],
            image_path=f"{path_dir_per_batch}/Paths.png",
            title1="Image NW Path Heatmap",
            title2="Text NW Path Heatmap",
            patches_right_y=y_text,
            patches_right_x=x_text
        )
        
        # Visualize the distance between paths
        path_distance = torch.abs(imagePath[0] - textPath[0])
        visualize_heatmap(
            path_distance,
            "Distance Between Image & Text Paths",
            f"{path_dir_per_batch}/PathDistance.png"
        )
        # Save sum of path_distance to txt in path directory
        try:
            path_sum = path_distance.sum().item()
            with open(f"{path_dir_per_batch}/PathDistanceSum.txt", "w") as f:
                f.write(f"{path_sum}\n")
        except Exception as e:
            print(f"Failed to write PathDistanceSum.txt: {e}")
        del path_distance

        # Visualize vertical vectors using HeightDiff_loss helper
        # Note: This is optional and can be commented out if not needed
        if loss_type == 'HeightDiff':
            from LossFunctionWithHelpers import HeightDiff_loss
            
            Vertical_HN_Vector, Vertical_HDIFF_Vector, Horizontal_HN_Vector, Horizontal_HDIFF_Vector = HeightDiff_loss(
                debug_diffNWimage[i].squeeze()*imagePath[0],
                debug_NWTextMatrix[i].squeeze()*textPath[0],
                lamda=1.0, loss_calc=False
            )

            height_vectors_dir_per_batch = f'{matrices_epoch_dir}/{i}/HeightVectors'
            os.makedirs(height_vectors_dir_per_batch, exist_ok=True)
            
            # Save vertical vectors visualization (displayed horizontally, stacked vertically)
            VisualizeHorizontalVectorHeatmaps(
                vector1=Vertical_HDIFF_Vector,
                vector2=Vertical_HN_Vector,
                image_path=f"{height_vectors_dir_per_batch}/VerticalVectors.png",
                title1="Image Vertical Vector Heatmap",
                title2="Text Vertical Vector Heatmap",
                show_values=True,
                value_fmt=".2f"
            )
            
            # Calculate and visualize the distance between the two vertical vectors
            distance_vector = torch.abs(Vertical_HDIFF_Vector - Vertical_HN_Vector)
            VisualizeSingleHorizontalVectorHeatmap(
                vector=distance_vector,
                image_path=f"{height_vectors_dir_per_batch}/VerticalVectorsDistance.png",
                title="Distance Between Image & Text Vertical Vectors",
                show_values=True,
                value_fmt=".2f"
            )
            # Save sum of vertical vectors distance to txt in HeightVectors directory
            try:
                vectors_sum = distance_vector.sum().item()
                with open(f"{height_vectors_dir_per_batch}/VerticalVectorsDistanceSum.txt", "w") as f:
                    f.write(f"{vectors_sum}\n")
            except Exception as e:
                print(f"Failed to write VerticalVectorsDistanceSum.txt: {e}")
            del distance_vector
            del Vertical_HN_Vector, Vertical_HDIFF_Vector
            del Horizontal_HN_Vector, Horizontal_HDIFF_Vector

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
    ax.set_xticklabels(x_indices[x_tick_locs], rotation=90, fontsize=5)
    ax.set_yticklabels(y_indices[y_tick_locs], rotation=90, fontsize=5)

    # Save heatmap
    plt.savefig(image_path, dpi=300, bbox_inches='tight')



def VisualizeWithImageAxis(ImageMatrix, TextMatrix, image_path, title1, title2, 
                           patches_left_y=[], patches_left_x=[],
                            patches_right_y=[], patches_right_x=[], zoom_factor=0.2, 
                            y_axis_x_offset=-30, x_axis_y_offset=25):
    # Convert tensors to numpy
    ImageMatrix_np = ImageMatrix.detach().cpu().numpy() if torch.is_tensor(ImageMatrix) else np.array(ImageMatrix)
    TextMatrix_np = TextMatrix.detach().cpu().numpy() if torch.is_tensor(TextMatrix) else np.array(TextMatrix)

    matrices_to_plot = [ImageMatrix_np, TextMatrix_np]
    titles = [title1, title2]
    use_img_labels = patches_left_y is not None and patches_left_x is not None

    # Save a single PNG file with the heatmaps of ImageMatrix and TextMatrix side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for idx, ax in enumerate(axes):
        matrix_data = matrices_to_plot[idx]
        if idx == 1:
            label_x = patches_right_x
            label_y = patches_right_y
        else:
            label_x = list(range(matrix_data.shape[1]))
            label_y = list(range(matrix_data.shape[0]))
            
        sns.heatmap(matrix_data, cmap='jet', linewidths=0.1, 
                    linecolor='black', ax=ax, annot=False, fmt=".3f",
                    xticklabels=label_x, yticklabels=label_y)
        
        # Adjust tick labels: rotation and much smaller font size
        ax.tick_params(axis='x', rotation=90, labelsize=5)
        ax.tick_params(axis='y', rotation=90, labelsize=5)
        
        ax.text(0.5, -0.05, titles[idx], transform=ax.transAxes,
                ha="center", va="top", fontsize=10)
        
        if use_img_labels:
            if len(patches_left_y) == matrix_data.shape[0]:
                for k in range(matrix_data.shape[0]):
                    patch_tensor = patches_left_y[k]
                    img = patch_tensor.cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1) if img.max() > 1.0 and img.dtype == np.float32 else img
                    oi = OffsetImage(img, zoom=zoom_factor)
                    ab = AnnotationBbox(oi, (0, k + 0.5),
                                        xybox=(y_axis_x_offset, 0), frameon=False,
                                        xycoords='data', boxcoords="offset points", pad=0.1)
                    ax.add_artist(ab)
            if len(patches_left_x) == matrix_data.shape[1]:
                for k in range(matrix_data.shape[1]):
                    patch_tensor = patches_left_x[k]
                    img = patch_tensor.cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1) if img.max() > 1.0 and img.dtype == np.float32 else img
                    oi = OffsetImage(img, zoom=zoom_factor)
                    ab = AnnotationBbox(oi, (k + 0.5, 0),
                                        xybox=(0, x_axis_y_offset), frameon=False,
                                        xycoords='data', boxcoords="offset points", pad=0.1)
                    ax.add_artist(ab)
    if use_img_labels:
        fig.subplots_adjust(left=0.4, bottom=0.35)
    fig.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



def VisualizeWithText(matrix_data, x_labels, y_labels, image_path, title):
    if torch.is_tensor(matrix_data):
        matrix_data = matrix_data.detach().cpu().numpy()

    # Dynamically adjust figsize based on label count; ensure minimum size
    fig_width = max(12, len(x_labels) * 0.35)
    fig_height = max(10, len(y_labels) * 0.35)
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(matrix_data, annot=True, fmt=".0f", cmap='viridis',
                     xticklabels=x_labels, yticklabels=y_labels,
                     linewidths=.5, cbar=True)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='x', labelrotation=90, labelsize=5) # Rotate x-labels, adjust size
    ax.tick_params(axis='y', labelrotation=90, labelsize=5)  # Adjust y-label size
    plt.tight_layout()
    plt.savefig(image_path, dpi=150) # DPI can be adjusted based on needs
    plt.close()



def VisualizeDualCharHeatmaps(ScoreMatrix, PathMatrix, x_labels, y_labels, image_path, title1, title2):
    if torch.is_tensor(ScoreMatrix):
        ScoreMatrix = ScoreMatrix.detach().cpu().numpy()
    if torch.is_tensor(PathMatrix):
        PathMatrix = PathMatrix.detach().cpu().numpy()

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
        ax.tick_params(axis='y', labelrotation=90, labelsize=8)

    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()


def VisualizeHorizontalVectorHeatmaps(vector1, vector2, image_path, title1, title2, 
                                       show_values=True, value_fmt=".2f", cmap='jet'):
    """
    Create two horizontal heatmaps stacked vertically (one under the other).
    Each vector is displayed as a single row with values annotated and indices on the x-axis.
    
    Args:
        vector1: First vector (1D tensor or array) - displayed as top heatmap
        vector2: Second vector (1D tensor or array) - displayed as bottom heatmap
        image_path: Path to save the output image
        title1: Title for the first (top) heatmap
        title2: Title for the second (bottom) heatmap
        show_values: Whether to annotate cells with their values
        value_fmt: Format string for value annotations (e.g., ".2f", ".0f")
        cmap: Colormap to use for the heatmaps
    """
    # Convert tensors to numpy and ensure 1D
    if torch.is_tensor(vector1):
        vector1 = vector1.detach().cpu().numpy()
    if torch.is_tensor(vector2):
        vector2 = vector2.detach().cpu().numpy()
    
    vector1 = np.array(vector1).flatten()
    vector2 = np.array(vector2).flatten()
    
    # Reshape to (1, N) for horizontal display
    matrix1 = vector1.reshape(1, -1)
    matrix2 = vector2.reshape(1, -1)
    
    # Calculate figure size based on vector length
    vec_len = max(len(vector1), len(vector2))
    fig_width = max(12, vec_len * 0.5)
    fig_height = 6  # Fixed height for two single-row heatmaps
    
    # Create two subplots stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    matrices = [matrix1, matrix2]
    titles = [title1, title2]
    
    for idx, ax in enumerate(axes):
        matrix_data = matrices[idx]
        x_labels = list(range(matrix_data.shape[1]))
        
        # Create heatmap with values and x-axis indices
        sns.heatmap(
            matrix_data, 
            cmap=cmap, 
            linewidths=0.5, 
            linecolor='black',
            ax=ax, 
            annot=show_values, 
            fmt=value_fmt,
            annot_kws={'fontsize': 8},
            xticklabels=x_labels,
            yticklabels=False,  # No y-axis labels for single row
            cbar=True,
            cbar_kws={'shrink': 0.5}
        )
        
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('Index', fontsize=10)
        ax.tick_params(axis='x', labelrotation=0, labelsize=8)
    
    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def VisualizeSingleHorizontalVectorHeatmap(vector, image_path, title, 
                                            show_values=True, value_fmt=".2f", cmap='jet'):
    """
    Create a single horizontal heatmap for a vector.
    The vector is displayed as a single row with values annotated and indices on the x-axis.
    
    Args:
        vector: Vector (1D tensor or array) to display
        image_path: Path to save the output image
        title: Title for the heatmap
        show_values: Whether to annotate cells with their values
        value_fmt: Format string for value annotations (e.g., ".2f", ".0f")
        cmap: Colormap to use for the heatmap
    """
    # Convert tensor to numpy and ensure 1D
    if torch.is_tensor(vector):
        vector = vector.detach().cpu().numpy()
    
    vector = np.array(vector).flatten()
    
    # Reshape to (1, N) for horizontal display
    matrix = vector.reshape(1, -1)
    
    # Calculate figure size based on vector length
    vec_len = len(vector)
    fig_width = max(12, vec_len * 0.5)
    fig_height = 3  # Fixed height for single-row heatmap
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    x_labels = list(range(matrix.shape[1]))
    
    # Create heatmap with values and x-axis indices
    sns.heatmap(
        matrix, 
        cmap=cmap, 
        linewidths=0.5, 
        linecolor='black',
        ax=ax, 
        annot=show_values, 
        fmt=value_fmt,
        annot_kws={'fontsize': 8},
        xticklabels=x_labels,
        yticklabels=False,  # No y-axis labels for single row
        cbar=True,
        cbar_kws={'shrink': 0.5}
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Index', fontsize=10)
    ax.tick_params(axis='x', labelrotation=0, labelsize=8)
    
    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
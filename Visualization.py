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


def save_debug_visualizations(model, text1, text2, image1, image2,
                            TextSimilarGT, TextSimilarPred,
                            Similar_TxtImg1, Similar_TxtImg2,
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

    similar_IMGTXT_epoch_dir = f'{loss_dir}/SimilarityMatricesPerEpoch/{model.model_arch}/Epoch_{epoch}'
    os.makedirs(similar_IMGTXT_epoch_dir, exist_ok=True)

    # Clone tensors for visualization to avoid affecting gradients
    debug_image1 = image1.detach().cpu()
    debug_image2 = image2.detach().cpu()
    debug_NWTextGT = TextSimilarGT.detach().cpu()
    debug_diffNWPred = TextSimilarPred.detach().cpu()
    debug_Similar_TxtImg1 = Similar_TxtImg1.detach().cpu()
    debug_Similar_TxtImg2 = Similar_TxtImg2.detach().cpu()

    # Save heatmap visualizations
    saveHeatmapPlots(
        text1, text2,
        debug_image1, debug_image2,
        similar_IMGTXT_epoch_dir,
        debug_diffNWPred, debug_NWTextGT,
        debug_Similar_TxtImg1, debug_Similar_TxtImg2,
    )

    del debug_image1, debug_image2
    del debug_diffNWPred, debug_NWTextGT
    del similar_IMGTXT_epoch_dir
    del debug_Similar_TxtImg1, debug_Similar_TxtImg2




def saveHeatmapPlots(text1, text2, image1, image2, 
                    similarity_epoch_dir,
                    debug_similarTextPred, debug_similarTextGT, 
                    debug_Similar_TxtImg1, debug_Similar_TxtImg2):
    
    for i in range(image1.size(0)): # Iterate through items in the current batch
        similarity_dir_per_batch = f'{similarity_epoch_dir}/{i}'
        os.makedirs(similarity_dir_per_batch, exist_ok=True)

        # Generate patches for visualization
        image1_i = image1[i].unsqueeze(0)
        image2_i = image2[i].unsqueeze(0)
        
        # Calculate stride using the same logic as in train.py to match model output dimensions
        # Use stride_ratio if available from Parameters.py, otherwise default to 1.0 (no overlap)
        ratio = stride_ratio if 'stride_ratio' in globals() else 1.0
        model_stride = max(1, int(window_size * ratio))
        model_window_size = window_size
        
        # print(f"DEBUG: Visualization using window={model_window_size}, stride={model_stride}")
        
        windows_img1 = sliding_window(image1_i, model_window_size, model_stride).squeeze(0)
        windows_img2 = sliding_window(image2_i, model_window_size, model_stride).squeeze(0)

        y_image = torch.flip(windows_img1, dims=[0]) # From image1, for Y-axis
        # Flip x-axis patches vertically (dim 2) in addition to sequence reversal (dim 0)
        # matches user request to "make a flip for the image patches on the x-axis"
        x_image = torch.flip(windows_img2, dims=[0]) # From image2, for X-axis

        # Prepare text labels (Full and Stripped versions)
        y_text_full = list(text1[i])
        x_text_full = list(text2[i])
        y_text_stripped = list(text1[i].replace(" ", ""))
        x_text_stripped = list(text2[i].replace(" ", ""))

        # Helper to select correct labels based on matrix dimension
        def get_labels(dim_len, full_labels, stripped_labels):
            if dim_len == len(full_labels):
                return full_labels
            elif dim_len == len(stripped_labels):
                return stripped_labels
            return list(range(dim_len))

        # Select labels for Image-Text Similarity matrices
        # debug_Similar_TxtImg1 is [Text, Image]
        h_sim1, _ = debug_Similar_TxtImg1[i].shape
        y_text_sim1 = get_labels(h_sim1, y_text_full, y_text_stripped)
        
        h_sim2, _ = debug_Similar_TxtImg2[i].shape
        y_text_sim2 = get_labels(h_sim2, y_text_full, y_text_stripped) # Using y_text_full/stripped because logic below used y_text?
        # WAIT: In original code:
        # VisualizeMatrixWithAxis(... cur_patches_y=y_text, cur_patches_x=y_image) for Image1-Text1
        # VisualizeMatrixWithAxis(... cur_patches_y=y_text, cur_patches_x=x_image) for Image2-Text2
        # Use x_text for Image2-Text2 (Text2 maps to Image2)
        x_text_sim2 = get_labels(h_sim2, x_text_full, x_text_stripped)

        # Visualize similarity matrix instead of raw matrices
        VisualizeMatrixWithAxis(
            matrix_data=debug_Similar_TxtImg1[i],
            image_path=f"{similarity_dir_per_batch}/IMG_TXT_Similarity_Image1_Text1.png",
            title="Image1-Text1 Similarity Heatmap",
            cur_patches_y=y_text_sim1,
            cur_patches_x=y_image
        )
        VisualizeMatrixWithAxis(
            matrix_data=debug_Similar_TxtImg2[i],
            image_path=f"{similarity_dir_per_batch}/IMG_TXT_Similarity_Image2_Text2.png",
            title="Image2-Text2 Similarity Heatmap",
            cur_patches_y=x_text_sim2, # Corrected to Text2
            cur_patches_x=x_image
        )

        # Select labels for Text-Text Similarity matrices
        # debug_similarTextPred is [Text1, Text2]
        pred_h, pred_w = debug_similarTextPred[i].shape
        y_text_pred = get_labels(pred_h, y_text_full, y_text_stripped)
        x_text_pred = get_labels(pred_w, x_text_full, x_text_stripped)

        gt_h, gt_w = debug_similarTextGT[i].shape
        y_text_gt = get_labels(gt_h, y_text_full, y_text_stripped)
        x_text_gt = get_labels(gt_w, x_text_full, x_text_stripped)

        VisualizeMatrixWithAxis(
            matrix_data=debug_similarTextPred[i],
            image_path=f"{similarity_dir_per_batch}/Txt1_Txt2_Similarity_Predicted.png",
            title="Predicted Similarity Heatmap",
            cur_patches_y=y_text_pred,
            cur_patches_x=x_text_pred
        )
        VisualizeMatrixWithAxis(
            matrix_data=debug_similarTextGT[i],
            image_path=f"{similarity_dir_per_batch}/Txt1_Txt2_Similarity_GT.png",
            title="Ground Truth Similarity Heatmap",
            cur_patches_y=y_text_gt,
            cur_patches_x=x_text_gt
        )


        del windows_img1, windows_img2, y_image, x_image
        del image1_i, image2_i



def visualize_paths(ImgTxt1Similar, ImgTxt2Similar=None, epoch=0, batch_idx=0):
    ImgTxt1Similar = ImgTxt1Similar.cpu().detach().numpy()
    if ImgTxt2Similar is not None:
        ImgTxt2Similar = ImgTxt2Similar.cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(ImgTxt1Similar, interpolation='nearest')
    axes[0].set_title(f"Image NW Path (Epoch {epoch}, Batch {batch_idx})")
    axes[0].axis("off")

    if ImgTxt2Similar is not None:
        axes[1].imshow(ImgTxt2Similar, interpolation='nearest')
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



def VisualizeMatrixWithAxis(matrix_data, image_path, title, 
                            cur_patches_y, cur_patches_x, zoom_factor=0.3, 
                            y_axis_x_offset=-40, x_axis_y_offset=40):
    # Convert inputs to correct types
    matrix_data = matrix_data.detach().cpu().numpy() if torch.is_tensor(matrix_data) else np.array(matrix_data)

    # Dynamic figsize: Make it bigger so image patches on axes are visible
    # But ensure we don't exceed matplotlib's pixel limit (2^16 = 65536)
    # Safe limit at 300 DPI is ~200 inches.
    h, w = matrix_data.shape
    fig_w = max(25, min(w * 0.6, 180)) # 0.6 inches per column, max 180 inches
    fig_h = max(20, min(h * 0.6, 180)) # 0.6 inches per row, max 180 inches

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Determine labels and if needs image drawing for Y-axis
    draw_y_imgs = False
    if cur_patches_y is not None and len(cur_patches_y) == matrix_data.shape[0]:
        # check if first element is tensor
        if len(cur_patches_y) > 0 and (torch.is_tensor(cur_patches_y[0]) or isinstance(cur_patches_y[0], np.ndarray)):
            label_y = list(range(matrix_data.shape[0]))
            draw_y_imgs = True
        else:
            label_y = cur_patches_y
    else:
        label_y = list(range(matrix_data.shape[0]))

    # Determine labels and if needs image drawing for X-axis
    draw_x_imgs = False
    if cur_patches_x is not None and len(cur_patches_x) == matrix_data.shape[1]:
        # check if first element is tensor
        if len(cur_patches_x) > 0 and (torch.is_tensor(cur_patches_x[0]) or isinstance(cur_patches_x[0], np.ndarray)):
            label_x = False # Drop x-axis indexes
            draw_x_imgs = True
        else:
            label_x = cur_patches_x
    else:
        label_x = False # Drop x-axis indexes

    sns.heatmap(matrix_data, cmap='jet', linewidths=0.1, 
                linecolor='black', ax=ax, annot=True, fmt=".2f",
                xticklabels=label_x, yticklabels=label_y,
                cbar=True, annot_kws={"size": 10}) # Increased annotation font size slightly, assuming cells are big
    
    # Adjust tick labels: rotation and much smaller font size
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    ax.set_title(title, fontsize=16)
    
    if draw_y_imgs:
        for k in range(matrix_data.shape[0]):
            patch_tensor = cur_patches_y[k]
            img = patch_tensor.cpu().permute(1, 2, 0).numpy() if torch.is_tensor(patch_tensor) else patch_tensor
            # If numpy array is (C,H,W), permute. If (H,W,C), keep.
            if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[2] not in [1, 3]: 
                    img = np.transpose(img, (1, 2, 0))
            
            img = np.clip(img, 0, 1) if img.max() > 1.0 and img.dtype == np.float32 else img
            oi = OffsetImage(img, zoom=zoom_factor)
            ab = AnnotationBbox(oi, (0, k + 0.5),
                                xybox=(y_axis_x_offset, 0), frameon=False,
                                xycoords='data', boxcoords="offset points", pad=0.1)
            ax.add_artist(ab)
    
    if draw_x_imgs:
        for k in range(matrix_data.shape[1]):
            patch_tensor = cur_patches_x[k]
            img = patch_tensor.cpu().permute(1, 2, 0).numpy() if torch.is_tensor(patch_tensor) else patch_tensor
            if img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[2] not in [1, 3]:
                    img = np.transpose(img, (1, 2, 0))

            img = np.clip(img, 0, 1) if img.max() > 1.0 and img.dtype == np.float32 else img
            oi = OffsetImage(img, zoom=zoom_factor)
            
            # Place images at the bottom of the heatmap
            ab = AnnotationBbox(oi, (k + 0.5, matrix_data.shape[0]),
                                xybox=(0, -x_axis_y_offset), frameon=False,
                                xycoords='data', boxcoords="offset points", pad=0.1)
            ax.add_artist(ab)

    # Use tight_layout but leave space for custom artists (images)
    # Since we use bbox_inches='tight' in savefig, simple layout adjust is often enough
    fig.tight_layout()
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
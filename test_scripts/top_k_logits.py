import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from Parameters import *
from embeddingModel import EmbeddingModel, sliding_window
# from DiffNWAlgo import DiffNWAlgo
from newDataLoader import transform
from pathExtractor import *

def add_patch_axis_labels(ax, patches, axis='x', zoom=0.3):
    patches = patches.cpu()
    for i, patch in enumerate(patches):
        # Format patch for display
        if patch.shape[0] == 3: # C, H, W
             disp_patch = patch.permute(1, 2, 0).numpy()
        else: # 1, H, W or H, W
             disp_patch = patch.squeeze().numpy()
        
        # Normalize
        disp_patch = (disp_patch - disp_patch.min()) / (disp_patch.max() - disp_patch.min() + 1e-6)
        
        # Create imagebox, rotating for y-axis to fit better and flow vertically
        if axis == 'y':
            # Rotate 90 degrees to align with Y axis direction
            # User requested flip/adjustment.
            # Combined with Mirror Flip above:
            # - Mirror (L->R)
            # - Rot90 (k=1, CCW) -> Text flows Up? or Down?
            #   [A B] -> Mirror [B A] -> Rot [A] (A at bottom)
            #                                [B]
            disp_patch = np.rot90(disp_patch, k=1)
            
        imagebox = OffsetImage(disp_patch, zoom=zoom, cmap='gray')
        
        if axis == 'x':
            ab = AnnotationBbox(imagebox, (i, 0),
                                xybox=(0, -25),
                                xycoords=('data', 'axes fraction'),
                                boxcoords="offset points",
                                box_alignment=(0.5, 1.0),
                                frameon=False)
        else: # y axis
            ab = AnnotationBbox(imagebox, (0, i),
                                xybox=(-25, 0),
                                xycoords=('axes fraction', 'data'),
                                boxcoords="offset points",
                                box_alignment=(1.0, 0.5), # Right align to the axis
                                frameon=False)
            
        ax.add_artist(ab)

def compute_accumulated_matrix(similarity_matrix, gap_penalty):
    """
    Computes the accumulated score matrix (forward pass of NW)
    sim_matrix: [H, W]
    """
    H, W = similarity_matrix.shape
    acc_matrix = torch.zeros((H, W), device=similarity_matrix.device)
    
    # Initialization
    acc_matrix[0, 0] = similarity_matrix[0, 0]
    
    # Initialize first row and column
    for i in range(1, H):
        acc_matrix[i, 0] = acc_matrix[i-1, 0] + gap_penalty
    for j in range(1, W):
        acc_matrix[0, j] = acc_matrix[0, j-1] + gap_penalty
        
    # DP
    for i in range(1, H):
        for j in range(1, W):
            match = acc_matrix[i-1, j-1] + similarity_matrix[i, j]
            delete = acc_matrix[i-1, j] + gap_penalty
            insert = acc_matrix[i, j-1] + gap_penalty
            acc_matrix[i, j] = max(match, delete, insert)
            
    return acc_matrix

def load_image(image_path, device):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)
        
    # Apply transforms
    # transform expects PIL image
    if transform:
        img_tensor = transform(img)
    else:
        import torchvision.transforms as T
        t = T.Compose([
            T.Resize((128, 1024)),
            T.ToTensor()
        ])
        img_tensor = t(img)
    
    if not isinstance(img_tensor, torch.Tensor):
        # In case transform returns something else or ToTensorWithGrad behavior
        img_tensor = torch.tensor(img_tensor)

    return img_tensor.unsqueeze(0).to(device) # [1, C, H, W]

def main():
    parser = argparse.ArgumentParser(description="Test Alignment Model")
    parser.add_argument("weights", type=str, help="Path to model weights")
    parser.add_argument("image1", type=str, help="Path to first image")
    parser.add_argument("image2", type=str, help="Path to second image")
    parser.add_argument("--output", type=str, default="test_output.png", help="Path to save visualization")
    
    args = parser.parse_args()
    
    print(f"Model Architecture: {model_arch}")
    print(f"Device: {device}")

    # Initialize model
    print("Initializing model...")
    model = EmbeddingModel(
        window_size=window_size,
        stride=window_size, # Using window_size as stride as in train.py
        vector_size=vector_size,
        model_arch=model_arch,
        device=device
    ).to(device)
    
    # Load weights
    print(f"Loading weights from {args.weights}")
    if os.path.exists(args.weights):
        try:
            checkpoint = torch.load(args.weights, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)
    else:
        print(f"Weights file not found: {args.weights}")
        sys.exit(1)

    model.eval()

    # Load images
    print("Loading images...")
    image1 = load_image(args.image1, device)
    image2 = load_image(args.image2, device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        # Get tokens
        tokens_a, tokens_b = model(image1, image2)

        # Flip tokens for Arabic (RTL)
        tokens_a = torch.flip(tokens_a, dims=[-2])
        tokens_b = torch.flip(tokens_b, dims=[-2])
        
        # Normalize
        # In train.py: normalized_tokens_a = F.normalize(flip_tokens_a, p=2, dim=-1)
        # Note: train.py flips tokens. If we want to align images in reading direction,
        # and images are same direction, we probably don't need to flip.
        # But if the model learned features assuming flipped input, we might need to.
        # Arabic is RTL. Images might be stored LTR or RTL?
        # Assuming standard image coordinates (left 0 to right W).
        # Anyhow, we compute similarity.
        
        norm_tokens_a = F.normalize(tokens_a, p=2, dim=-1)
        norm_tokens_b = F.normalize(tokens_b, p=2, dim=-1)
        
        # Compute Similarity Matrix: [B, SeqA, SeqB]
        # sim = A @ B.T
        similarity_matrix = torch.bmm(norm_tokens_a, norm_tokens_b.transpose(1, 2))
        
        # Create alignment path by taking top-k logits per row
        B, H, W = similarity_matrix.shape
        top_k = min(5, W)  # Top 5 or fewer if W is small
        
        # Get indices of top-k values in each row (along columns)
        topk_values, topk_indices = torch.topk(similarity_matrix, k=top_k, dim=2)  # [B, H, k]
        
        # Create a colored alignment path (values 1, 2, 3, ... for top-1, top-2, top-3, ...)
        alignment_path = torch.zeros_like(similarity_matrix)
        
        for b in range(B):
            for h in range(H):
                for k_idx in range(top_k):
                    w_idx = topk_indices[b, h, k_idx]
                    # Assign value based on rank (1 = best, 2 = second, etc.)
                    alignment_path[b, h, w_idx] = top_k - k_idx  # top-1 gets highest value

        # Extract patches for visualization
        # Sliding window returns [B, num_windows, C, H, W]
        patches_a = sliding_window(image1, window_size, window_size).squeeze(0)
        patches_b = sliding_window(image2, window_size, window_size).squeeze(0)

        # Flip patches to match token order (RTL) - REMOVED to keep L->R visualization
        patches_a = torch.flip(patches_a, dims=[0])
        # patches_b = torch.flip(patches_b, dims=[0])
        
    
    # Visualization
    print("Visualizing result...")
    
    # Convert to numpy
    sim_mat = similarity_matrix.squeeze(0).cpu().numpy()
    align_mat = alignment_path.squeeze(0).cpu().numpy()
    top_k_val = top_k  # Save for use in visualization

    # Flip matrices to match L->R patch order (since tokens were RTL)
    # Patches B (X-axis) is L->R. Tokens B is R->L. So we Flip X (axis 1).
    # Patches A (Y-axis) is flipped to R->L. Tokens A is R->L. So we DO NOT Flip Y (axis 0).
    sim_mat = np.flip(sim_mat, axis=1)
    align_mat = np.flip(align_mat, axis=1)
    
    # Setup plot (make it taller for x-axis labels and wider for y-axis labels)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot Alignment Path with colors for top-k
    # Create a custom colormap: 0=white, 1=light ... 5=dark (top-1)
    from matplotlib.colors import ListedColormap
    # Extended colors for top-5: White -> Very Light -> Light -> Medium -> Dark -> Very Dark
    colors = ['white', '#E1F5FE', '#B3E5FC', '#4FC3F7', '#0288D1', '#01579B']
    cmap_topk = ListedColormap(colors[:top_k_val + 1])  # +1 for 0 (no match)
    
    im2 = ax.imshow(align_mat, cmap=cmap_topk, aspect='auto', vmin=0, vmax=top_k_val)
    ax.set_title(f"Alignment Path (Top-{top_k_val} per row)")
    
    # Hide default ticks/labels
    ax.set_xticks(range(len(patches_b)))
    ax.set_xticklabels([])
    ax.set_yticks(range(len(patches_a)))
    ax.set_yticklabels([])

    # Add grid lines separating every row and column
    # Horizontal lines
    y_positions = np.arange(len(patches_a) + 1) - 0.5
    ax.hlines(y_positions, -0.5, len(patches_b) - 0.5, colors='black', linewidths=1)
    
    # Vertical lines
    x_positions = np.arange(len(patches_b) + 1) - 0.5
    ax.vlines(x_positions, -0.5, len(patches_a) - 0.5, colors='black', linewidths=1)

    # Add patch images to axes
    add_patch_axis_labels(ax, patches_b, axis='x', zoom=0.3)
    add_patch_axis_labels(ax, patches_a, axis='y', zoom=0.3)
    
    # Legend and colorbar removed as requested

    plt.tight_layout() # Adjust layout to make room for labels
    plt.savefig(args.output, bbox_inches='tight') # Ensure labels are saved
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()

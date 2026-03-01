"""
Script to visualize a heatmap showing the connection between image patches and Arabic letters.
Y-axis: Image patches (displayed as actual images)
X-axis: Arabic letters + space
Values: How much each patch is connected to each letter
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import torch
import torch.nn.functional as F
from Parameters import *
from embeddingModel import EmbeddingModel
from textEmbedding import TextEmbedding


"""
Run example:
    python compute_letter_per_img.py --output ../Results/patch_letter_heatmap.png
"""

# Define all Arabic letters (28 basic letters) plus space
ARABIC_LETTERS = [
    'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز',
    'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
    'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'إ', 'ؤ',
    'ئ', 'ة', 'ى', ' '
]


def extract_patches(image_path, patch_width=window_size, stride=window_size):
    """
    Extract patches from an image using a sliding window.
    
    Args:
        image_path: Path to the image file
        patch_width: Width of each patch
        stride: Stride for sliding window
    
    Returns:
        List of patch images (numpy arrays)
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    h, w = img_array.shape
    patches = []
    
    for x in range(0, w - patch_width + 1, stride):
        patch = img_array[:, x:x + patch_width]
        patches.append(patch)
    
    return patches


def compute_patch_letter_similarity(patches, letters=ARABIC_LETTERS, image_path=None, model_cnn=None, model_text=None):
    """
    Compute similarity between patches and letters based on cosine similarity
    between their embeddings.
    
    Args:
        patches: List of image patches (numpy arrays)
        letters: List of letters to compute similarity for
        image_path: Path to the image file (to get tokens from CNN model)
        model_cnn: The CNN embedding model
        model_text: The text embedding model
    
    Returns:
        numpy.ndarray: Similarity matrix (num_patches x num_letters)
    """
    if image_path is None or model_cnn is None or model_text is None:
        return np.zeros((len(patches), len(letters)))
    
    model_cnn.eval()
    model_text.eval()
    
    with torch.no_grad():
        # Load and preprocess image for the model
        img = Image.open(image_path).convert('RGB')
        # Use simple transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get tokens from CNN (image_a, image_b are needed due to model interface)
        # We'll use the same image twice
        tokens, _ = model_cnn(img_tensor, img_tensor)
        tokens = tokens[0]  # [num_patches, vector_size]
        
        # Flip tokens along the patch dimension as done in train.py (Arabic RTL support)
        tokens = torch.flip(tokens, dims=[0])
        
        # Re-sample tokens if max_patches was used in visualize_patch_letter_heatmap
        if tokens.shape[0] != len(patches):
            indices = np.linspace(0, tokens.shape[0] - 1, len(patches), dtype=int)
            tokens = tokens[indices]
            
        # Get embeddings for letters
        letter_embeddings = []
        for char in letters:
            emb = model_text(char) # [1, vector_size]
            letter_embeddings.append(emb)
        letter_embeddings = torch.cat(letter_embeddings, dim=0) # [num_letters, vector_size]
        
        # Compute dot product between tokens and letter embeddings
        # tokens: [num_patches, vector_size], letter_embeddings: [num_letters, vector_size]
        similarity = torch.mm(tokens, letter_embeddings.transpose(0, 1)) # [num_patches, num_letters]
        
        result = similarity
        # # Find closest letter (highest dot product)
        # max_indices = similarity.argmax(dim=1)
        
        # # Create a result matrix that highlights the closest letter
        # result = torch.zeros_like(similarity)
        # for i, idx in enumerate(max_indices):
        #     result[i, idx] = similarity[i, idx]
            
        return result.cpu().numpy()


def visualize_patch_letter_heatmap(image_path, text, output_path=None, 
                                    patch_width=window_size, stride=window_size, max_patches=30,
                                    patch_rotation=90, flip_horizontal=False, flip_vertical=False,
                                    reverse_patches=False, model_cnn=None, model_text=None):
    """
    Create a heatmap with image patches on X-axis and Arabic letters on Y-axis.
    
    Args:
        image_path: Path to the image file
        text: The corresponding text
        output_path: Path to save the figure (optional)
        patch_width: Width of each patch
        stride: Stride for sliding window
        max_patches: Maximum number of patches to display
        patch_rotation: Rotation angle for image patches
        flip_horizontal: Whether to flip patches horizontally
        flip_vertical: Whether to flip patches vertically
        reverse_patches: Whether to reverse the order of extracted patches
        model_cnn: CNN model for patch embeddings
        model_text: Text model for letter embeddings
    """
    # Extract patches
    patches = extract_patches(image_path, patch_width, stride)
    
    if len(patches) > max_patches:
        # Sample evenly
        indices = np.linspace(0, len(patches) - 1, len(patches), dtype=int)
        patches = [patches[i] for i in indices]
    
    if reverse_patches:
        patches = patches[::-1]
    
    num_patches = len(patches)
    
    # Compute similarity matrix (num_patches x num_letters)
    # We want patches on Y-axis (rows) and letters on X-axis (cols)
    similarity = compute_patch_letter_similarity(patches, image_path=image_path, model_cnn=model_cnn, model_text=model_text)
    
    # Create figure with a single heatmap
    # Width driven by letters, Height driven by patches
    fig, ax_heatmap = plt.subplots(figsize=(12, max(10, num_patches * 0.4)))
    
    # Plot heatmap
    im = ax_heatmap.imshow(similarity, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')
    
    # Set x-axis labels (Arabic letters)
    ax_heatmap.set_xticks(np.arange(len(ARABIC_LETTERS)))
    ax_heatmap.set_xticklabels(ARABIC_LETTERS, fontsize=11)
    
    # Set y-axis (patch indices) - hide labels to show images instead
    ax_heatmap.set_yticks(np.arange(num_patches))
    ax_heatmap.set_yticklabels(['' for _ in range(num_patches)])
    
    # Add grid lines between cells
    ax_heatmap.set_xticks(np.arange(len(ARABIC_LETTERS) + 1) - 0.5, minor=True)
    ax_heatmap.set_yticks(np.arange(num_patches + 1) - 0.5, minor=True)
    ax_heatmap.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)
    
    # Add patches as y-axis labels on the left
    for i, patch in enumerate(patches):
        # Scale and rotate patch for display
        patch_img = Image.fromarray(patch.astype(np.uint8))
        
        # Apply flips
        if flip_horizontal:
            patch_img = patch_img.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            patch_img = patch_img.transpose(Image.FLIP_TOP_BOTTOM)
            
        if patch_rotation != 0:
            patch_img = patch_img.rotate(patch_rotation, expand=True)
        
        # Add patch image to the axis
        imagebox = OffsetImage(np.array(patch_img), zoom=0.2, cmap='gray')
        # Place left of the axis: xybox=(-20, 0) moves it left
        ab = AnnotationBbox(imagebox, (0, i),
                           xybox=(-20, 0),
                           xycoords=('axes fraction', 'data'),
                           boxcoords="offset points",
                           box_alignment=(1.0, 0.5),
                           frameon=False)
        ax_heatmap.add_artist(ab)
    
    ax_heatmap.set_xlabel('Arabic Letters', fontsize=14, fontweight='bold')
    ax_heatmap.set_ylabel('Image Patches', fontsize=14, fontweight='bold')
    ax_heatmap.set_title('Patch-Letter Connection Heatmap', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label('Connection Strength', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    # Default configuration parameters
    image_path = 'DataSet/Synthetic_Arabic/images/img1_1.png'
    text_path = 'DataSet/Synthetic_Arabic/texts/text1_1.txt'
    output_path = 'test_scripts/Results/patch_letter_heatmap.png'
    
    # Heatmap settings
    max_patches = 50
    patch_rotation = 90
    flip_horizontal = False
    flip_vertical = False
    reverse_patches = True
    
    # Ensure we can find the files if run from either project root or test_scripts
    if not os.path.exists(image_path):
        image_path = os.path.join('..', image_path)
    if not os.path.exists(text_path):
        text_path = os.path.join('..', text_path)
        
    print(f"Image: {image_path}")
    
    # Read text
    if not os.path.exists(text_path):
        print(f"Error: Text file not found at {text_path}")
        return
        
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
    
    # Initialize Models
    print("Loading models...")
    model_cnn = EmbeddingModel(
        window_size=window_size,
        stride=window_size,
        vector_size=vector_size,
        device=device
    ).to(device)
    
    # Load weights if available
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'Weights', '14586433', 'model_epoch_130.pth')
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model_cnn.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: Weights not found at {weights_path}. Using random initialization.")
        
    model_text = TextEmbedding(embedding_dim=vector_size).to(device)
    
    visualize_patch_letter_heatmap(
        image_path=image_path,
        text=text,
        output_path=output_path,
        patch_width=window_size,
        stride=window_size,
        max_patches=max_patches,
        patch_rotation=patch_rotation,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        reverse_patches=reverse_patches,
        model_cnn=model_cnn,
        model_text=model_text
    )
    
    print("Done!")


if __name__ == "__main__":
    main()

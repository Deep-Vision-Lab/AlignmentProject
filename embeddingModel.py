from Parameters import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
import math
import time

import gc


# ============================================================================
# GPU TIMER FOR IMAGE EMBEDDING PROFILING
# ============================================================================
class ImageEmbedTimer:
    """
    GPU-aware timer for profiling image embedding phases.
    Uses CUDA events for accurate GPU timing.
    """
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.use_cuda = torch.cuda.is_available()
        self.timings = {}
        self.starts = {}
        
    def start(self, name):
        if not self.enabled:
            return
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.starts[name] = start_event
        else:
            self.starts[name] = time.time()
    
    def stop(self, name):
        if not self.enabled or name not in self.starts:
            return
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            duration_ms = self.starts[name].elapsed_time(end_event)
        else:
            duration_ms = (time.time() - self.starts[name]) * 1000
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_ms)
        del self.starts[name]
    
    def get_summary(self):
        """Return timing summary as dict"""
        summary = {}
        for name, durations in self.timings.items():
            summary[name] = {
                'avg_ms': sum(durations) / len(durations) if durations else 0,
                'total_ms': sum(durations),
                'count': len(durations)
            }
        return summary
    
    def reset(self):
        self.timings = {}
        self.starts = {}


# Global timer for image embedding (set enabled=True to profile)
img_embed_timer = ImageEmbedTimer(enabled=False)


# ============================================================================
# OPTIMIZATION 1: Learnable Positional Encoding for Image Patches
# ============================================================================
class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding that injects position information into patch embeddings.
    Critical for transformers to understand that Patch 1 comes before Patch 2.
    
    Formula: ImageVectors = CNN(Image) + E_pos
    where E_pos = PositionalEncoding(Range(0, num_patches))
    """
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, embed_dim] - sequence of patch embeddings
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (standard in NLP).
    Alternative to learnable encoding - more generalizable to different sequence lengths.
    """
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2]) if embed_dim % 2 == 1 else torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, embed_dim]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# OPTIMIZATION 3: BiLSTM for Local Sequence Context
# ============================================================================
class BiLSTMContextEncoder(nn.Module):
    """
    Bi-Directional LSTM that captures local sequence context between neighboring patches.
    
    Why: CNN extracts features from patches in isolation, but patches often contain 
    partial letters. BiLSTM allows information to flow between neighboring patches.
    
    Effect: The vector for "Patch 5" will now contain context from "Patch 4" and "Patch 6",
    helping the model understand it is looking at the middle of a word.
    """
    def __init__(self, input_dim, hidden_dim=None, num_layers=2, dropout=0.1, bidirectional=True):
        super(BiLSTMContextEncoder, self).__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2 if bidirectional else input_dim
        
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # BiLSTM layer - disable cuDNN dropout to avoid training mode issues
        # Set dropout=0 in LSTM and handle dropout separately
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # Disable internal dropout to avoid cuDNN training mode issues
            bidirectional=bidirectional
        )
        
        # Output projection to match original dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Linear(lstm_output_dim, input_dim)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Dropout (applied separately to avoid cuDNN issues)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, embed_dim] - sequence of patch embeddings from CNN
        Returns:
            contextualized: [B, seq_len, embed_dim] - patches with neighboring context
        """
        # Store residual for skip connection
        residual = x
        
        # Flatten parameters for cuDNN optimization (helps with training mode consistency)
        self.lstm.flatten_parameters()
        
        # Apply BiLSTM
        lstm_out, _ = self.lstm(x)  # [B, seq_len, hidden_dim * 2]
        
        # Project back to original dimension
        output = self.output_proj(lstm_out)  # [B, seq_len, embed_dim]
        
        # Apply dropout
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


# Sliding window function to divide image into patches (subwindows)
def sliding_window(image, window_size, stride, debug_mode=False, save_dir=False):
    patches = []
    # Unfolding the image into patches of size window_size
    for i  in range(image.shape[0]):
        image_windows = []
        for j in range(0, image.shape[3] - window_size + 1, stride):
            image_i = image[i, :, :, j:j + window_size]
            image_windows.append(image_i)
            if debug_mode:
                torchvision.utils.save_image(image_i, f"{save_dir}/patch_b{i}_w{j}.png")
        patches.append(torch.stack(image_windows, dim=0))
        del image_windows
    result = torch.stack(patches, dim=0)
    del patches
    return result

# CNN model for extracting features from sliding window patches
def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1



class EmbeddingModel(nn.Module):
    def __init__(self, window_size=128, stride=64, vector_size=512,
                  device='cuda', use_flip=True ,use_bilstm=True, use_positional_encoding=True, 
                  positional_encoding_type='learnable', bilstm_layers=2, dropout=0.1,
                  use_space_gate=True, space_threshold=0.05,
                  space_vector=None):
        """
        Image embedding model that extracts patch-level features using CNN + BiLSTM.
        
        Args:
            window_size: Size of sliding window for patch extraction
            stride: Stride for sliding window
            vector_size: Dimension of output feature vectors
            device: Device to run on ('cuda' or 'cpu')
            use_bilstm: Enable BiLSTM for sequence context
            use_positional_encoding: Enable positional encoding
            positional_encoding_type: 'learnable' or 'sinusoidal'
            bilstm_layers: Number of BiLSTM layers
            dropout: Dropout rate
            use_space_gate: Enable black patch detection gate (recommended)
            space_threshold: Pixel intensity threshold for detecting black patches (0-1)
            text_embedding: Reference to TextEmbedding model to get space character embedding
        """
        super(EmbeddingModel, self).__init__()
        
        self.device = device
        self.use_bilstm = use_bilstm
        
        # ============================================================================
        # SPACE GATE: Detect black patches and inject space embedding from text model
        # ============================================================================
        self.use_flip = use_flip
        self.use_space_gate = use_space_gate
        self.space_threshold = space_threshold  # Patches with mean pixel < threshold are "space"
        self.space_vector = space_vector  # Embedding vector for space character from text model

        # CNN encoder (ResNet34) for patch feature extraction
        self.cnn_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        cnn_feature_dim = self.cnn_encoder.fc.in_features
        self.cnn_encoder.fc = nn.Linear(cnn_feature_dim, vector_size)
        self.cnn_encoder = self.cnn_encoder.to(device)
        
        # ============================================================================
        # OPTIMIZATION 3: BiLSTM for Local Sequence Context (between CNN and Normalize)
        # ============================================================================
        if use_bilstm:
            self.bilstm_context = BiLSTMContextEncoder(
                input_dim=vector_size,
                hidden_dim=vector_size // 2,  # Will output vector_size after concatenation
                num_layers=bilstm_layers,
                dropout=dropout,
                bidirectional=True
            ).to(device)
        else:
            self.bilstm_context = None
        
        # ============================================================================
        # OPTIMIZATION 1: Positional Encoding (after BiLSTM, before Transformer)
        # ============================================================================
        self.use_positional_encoding = use_positional_encoding
        if positional_encoding_type == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(
                embed_dim=vector_size,
                max_len=512,
                dropout=dropout
            ).to(device)
        else:  # 'sinusoidal'
            self.positional_encoding = SinusoidalPositionalEncoding(
                embed_dim=vector_size,
                max_len=512,
                dropout=dropout
            ).to(device)
            
        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
    
    def detect_black_patches(self, patches):
        """
        Detect which patches are "black" (empty/space) based on pixel intensity.
        
        Uses a soft threshold (sigmoid) for differentiability. Patches with very 
        low pixel values will get values close to 1.0, while normal patches get 
        values close to 0.0.
        
        Args:
            patches: [B, num_patches, C, H, W] - extracted image patches
        
        Returns:
            space_weight: [B, num_patches] - soft weight in [0, 1], higher = more likely space
        """
        if not self.use_space_gate:
            return None
        
        # Compute mean pixel intensity per patch
        # patches: [B, num_patches, C, H, W]
        # We want mean across channels, height, width -> [B, num_patches]
        patch_means = patches.abs().mean(dim=(2, 3, 4))  # Use abs() for normalized images
        
        # Soft threshold using sigmoid: smooth transition around threshold
        # Lower patch_means -> higher space_weight (more likely to be space)
        # temperature controls sharpness: lower = sharper transition
        temperature = 10e-6  # Controls how sharp the transition is
        space_weight = torch.sigmoid((self.space_threshold - patch_means) / temperature)
        
        return space_weight
    
    def apply_space_gate(self, features, space_mask):
        """
        Blend CNN features with space embedding from text model based on soft space weights.
        
        This is fully differentiable - gradients flow through both the CNN path
        and the space embedding path, weighted by how "black" each patch is.
        
        Args:
            features: [B, num_patches, vector_size] - CNN output features
            space_vector: [vector_size] - space embedding vector from text model
            space_mask: [B, num_patches] - soft space weights
        
        Returns:
            blended_features: [B, num_patches, vector_size] - soft blend of features and space
        """
        # Expand space_mask to broadcast with features: [B, num_patches] -> [B, num_patches, 1]
        space_mask = space_mask.unsqueeze(-1)  # [B, num_patches, 1]
        
        # Expand space_vector to match features shape: [vector_size] -> [1, 1, vector_size] -> broadcast to [B, num_patches, vector_size]
        space_vec = self.space_vector.unsqueeze(0)  # [1, 1, vector_size]
        space_vec = space_vec.expand(features.shape[0], features.shape[1], -1)  # [B, num_patches, vector_size]
        
        return (1 - space_mask) * features + space_mask * space_vec
        
    def _process_cnn_branch(self, tokens_a, tokens_b, show_dims=False):
        """Process CNN branch"""
        batches_num, windows_num, Channels, H, W = tokens_a.shape
        
        # Reshape patches
        reshaped_tokens_a = tokens_a.reshape(batches_num * windows_num, Channels, H, W)
        reshaped_tokens_b = tokens_b.reshape(batches_num * windows_num, Channels, H, W)
        if show_dims: 
            print(f"Patches after reshaping: {reshaped_tokens_a.shape}, {reshaped_tokens_b.shape}")
        
        del tokens_a, tokens_b
    
        # Standard forward pass
        encoded_tokens_a = self.cnn_encoder(reshaped_tokens_a)
        encoded_tokens_b = self.cnn_encoder(reshaped_tokens_b)

        if show_dims: 
            print(f"Tokens after CNN: {encoded_tokens_a.shape}, {encoded_tokens_b.shape}")
        
        del reshaped_tokens_a, reshaped_tokens_b
        
        return encoded_tokens_a, encoded_tokens_b, batches_num, windows_num

    def forward(self, image_a, image_b, show_dims=False, debug=False, timer=None):
        # Use provided timer or global timer
        t = timer if timer is not None else img_embed_timer
        
        # ==================== PHASE: Sliding Window (Patch Extraction) ====================
        t.start('img_1a_sliding_window')
        tokens_a = sliding_window(image_a, self.window_size, self.stride, debug_mode=debug).to(device)
        tokens_b = sliding_window(image_b, self.window_size, self.stride, debug_mode=debug).to(device)
        t.stop('img_1a_sliding_window')
        
        if show_dims: 
            print(f"Patches: {tokens_a.shape}, {tokens_b.shape}")
        
        batches_num, windows_num = tokens_a.shape[:2]
        
        if self.use_flip:
            # Flip tokens to have shape [B, num_patches, C, H, W] for CNN processing
            tokens_a = torch.flip(tokens_a, dims=[1])
            tokens_b = torch.flip(tokens_b, dims=[1])

        # ==================== PHASE: Space Gate Detection ====================
        if self.use_space_gate:
            t.start('img_1b_space_detection')
            self.is_black_a = self.detect_black_patches(tokens_a) if self.use_space_gate else None
            self.is_black_b = self.detect_black_patches(tokens_b) if self.use_space_gate else None
            t.stop('img_1b_space_detection')
        
            if show_dims:
                num_black_a = self.is_black_a.sum().item()
                num_black_b = self.is_black_b.sum().item()
                print(f"Black patches detected: A={num_black_a}, B={num_black_b}")
        
        # ==================== PHASE: CNN Encoding ====================
        t.start('img_1c_cnn_encoding')
        encoded_tokens_a, encoded_tokens_b, batches_num, windows_num = self._process_cnn_branch(
            tokens_a, tokens_b, show_dims
        )
        t.stop('img_1c_cnn_encoding')
        
        # Reshape to final output
        features_vector_a = encoded_tokens_a.view(batches_num, windows_num, self.vector_size)
        features_vector_b = encoded_tokens_b.view(batches_num, windows_num, self.vector_size)
        del encoded_tokens_a, encoded_tokens_b
        
        # ==================== PHASE: Space Gate Application ====================
        t.start('img_1d_space_gate_apply')
        if self.use_space_gate and self.space_vector is not None:
            features_vector_a = self.apply_space_gate(features_vector_a, self.is_black_a)
            features_vector_b = self.apply_space_gate(features_vector_b, self.is_black_b)
        t.stop('img_1d_space_gate_apply')
        
        # ==================== PHASE: BiLSTM Context ====================
        t.start('img_1e_bilstm')
        if self.use_bilstm:
            features_vector_a = self.bilstm_context(features_vector_a)
            features_vector_b = self.bilstm_context(features_vector_b)
        t.stop('img_1e_bilstm')
        
        # ==================== PHASE: Positional Encoding ====================
        t.start('img_1f_positional_encoding')
        if self.use_positional_encoding:
            features_vector_a = self.positional_encoding(features_vector_a)
            features_vector_b = self.positional_encoding(features_vector_b)
        t.stop('img_1f_positional_encoding')
        
        return features_vector_a, features_vector_b


def visualize_patches_black_detection(image_path, window_size=128, stride=64, 
                                       space_threshold=0.05, save_dir="patch_visualization"):
    """
    Visualize patches extracted from an image and show which ones are detected as black/space.
    
    Args:
        image_path: Path to the input image
        window_size: Size of sliding window for patch extraction
        stride: Stride for sliding window
        space_threshold: Pixel intensity threshold for detecting black patches (0-1)
        save_dir: Directory to save the visualization results
    """
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Extract patches using sliding window
    patches = sliding_window(image_tensor, window_size, stride)  # [1, num_patches, C, H, W]
    patches = patches.squeeze(0)  # [num_patches, C, H, W]
    num_patches = patches.shape[0]
    print(f"Number of patches extracted: {num_patches}")
    
    # Calculate patch means (same logic as detect_black_patches)
    patch_means = patches.abs().mean(dim=(1, 2, 3))  # [num_patches]
    
    # Detect black patches using soft threshold (same as in the model)
    temperature = 10e-6
    space_weights = torch.sigmoid((space_threshold - patch_means) / temperature)
    
    # Determine if patch is "black" (space_weight > 0.5)
    is_black = space_weights > 0.5
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total patches: {num_patches}")
    print(f"Black patches detected: {is_black.sum().item()}")
    print(f"Non-black patches: {(~is_black).sum().item()}")
    print(f"Space threshold: {space_threshold}")
    print(f"{'='*60}\n")
    
    # Create figure for all patches overview
    cols = min(10, num_patches)
    rows = (num_patches + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Patch Visualization - Image: {os.path.basename(image_path)}\n"
                 f"Window: {window_size}, Stride: {stride}, Threshold: {space_threshold}", 
                 fontsize=14)
    
    for idx in range(num_patches):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # Convert patch to numpy for visualization
        patch_np = patches[idx].permute(1, 2, 0).numpy()  # [H, W, C]
        
        # Show patch
        ax.imshow(patch_np)
        
        # Color code based on black detection
        border_color = 'red' if is_black[idx] else 'green'
        status = "BLACK" if is_black[idx] else "OK"
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        
        # Title with patch info
        ax.set_title(f"P{idx}: {status}\nmean={patch_means[idx]:.4f}\nweight={space_weights[idx]:.2f}", 
                     fontsize=8, color=border_color)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_patches, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    overview_path = os.path.join(save_dir, "patches_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overview to: {overview_path}")
    
    # Save individual patches with their status
    individual_dir = os.path.join(save_dir, "individual_patches")
    os.makedirs(individual_dir, exist_ok=True)
    
    for idx in range(num_patches):
        fig, ax = plt.subplots(figsize=(4, 4))
        
        patch_np = patches[idx].permute(1, 2, 0).numpy()
        ax.imshow(patch_np)
        
        status = "BLACK" if is_black[idx] else "NORMAL"
        border_color = 'red' if is_black[idx] else 'green'
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
        
        ax.set_title(f"Patch {idx} - {status}\n"
                     f"Mean Intensity: {patch_means[idx]:.6f}\n"
                     f"Space Weight: {space_weights[idx]:.4f}", 
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        
        patch_path = os.path.join(individual_dir, f"patch_{idx:03d}_{status.lower()}.png")
        plt.savefig(patch_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_patches} individual patches to: {individual_dir}")
    
    # Create a bar plot of patch means
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, num_patches * 0.5), 8))
    
    # Bar plot of means
    colors = ['red' if b else 'green' for b in is_black]
    bars = ax1.bar(range(num_patches), patch_means.numpy(), color=colors, alpha=0.7)
    ax1.axhline(y=space_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'Threshold ({space_threshold})')
    ax1.set_xlabel('Patch Index', fontsize=12)
    ax1.set_ylabel('Mean Pixel Intensity', fontsize=12)
    ax1.set_title('Patch Mean Intensities (Red = Black/Space, Green = Normal)', fontsize=14)
    ax1.legend()
    ax1.set_xlim(-0.5, num_patches - 0.5)
    
    # Bar plot of space weights
    ax2.bar(range(num_patches), space_weights.numpy(), color=colors, alpha=0.7)
    ax2.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label='Decision boundary (0.5)')
    ax2.set_xlabel('Patch Index', fontsize=12)
    ax2.set_ylabel('Space Weight (Sigmoid)', fontsize=12)
    ax2.set_title('Space Detection Weights (Higher = More Likely Black/Space)', fontsize=14)
    ax2.legend()
    ax2.set_xlim(-0.5, num_patches - 0.5)
    
    plt.tight_layout()
    stats_path = os.path.join(save_dir, "patch_statistics.png")
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics plot to: {stats_path}")
    
    # Save detailed CSV with patch info
    csv_path = os.path.join(save_dir, "patch_data.csv")
    with open(csv_path, 'w') as f:
        f.write("patch_index,mean_intensity,space_weight,is_black\n")
        for idx in range(num_patches):
            f.write(f"{idx},{patch_means[idx]:.6f},{space_weights[idx]:.6f},{is_black[idx].item()}\n")
    print(f"Saved patch data CSV to: {csv_path}")
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION COMPLETE!")
    print(f"All results saved to: {save_dir}")
    print(f"{'='*60}")
    
    return patches, patch_means, space_weights, is_black


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize image patches and black patch detection')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--window_size', type=int, default=128, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=64, help='Sliding window stride')
    parser.add_argument('--threshold', type=float, default=0.05, help='Black patch threshold')
    parser.add_argument('--save_dir', type=str, default='patch_visualization', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic image')
    
    args = parser.parse_args()
    
    if args.demo or args.image is None:
        # Create a demo image with some black regions
        print("Creating demo image with black regions...")
        demo_dir = "patch_visualization_demo"
        os.makedirs(demo_dir, exist_ok=True)
        
        # Create synthetic image: mostly content with some black (space) regions
        import numpy as np
        from PIL import Image
        
        # Create image with patterns and black spaces
        img_width, img_height = 1024, 128
        img_array = np.ones((img_height, img_width, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Add some "text-like" dark regions
        for i in range(0, img_width, 150):
            if (i // 150) % 3 != 0:  # Leave every 3rd region as "space"
                # Add dark pattern (simulating text)
                x_start = i + 20
                x_end = min(i + 130, img_width)
                img_array[20:108, x_start:x_end] = np.random.randint(50, 150, 
                    (88, x_end - x_start, 3), dtype=np.uint8)
            else:
                # Keep as black/space region
                x_start = i
                x_end = min(i + 150, img_width)
                img_array[:, x_start:x_end] = 0  # Black region
        
        demo_image_path = os.path.join(demo_dir, "demo_image.png")
        Image.fromarray(img_array).save(demo_image_path)
        print(f"Demo image saved to: {demo_image_path}")
        
        # Run visualization on demo image
        visualize_patches_black_detection(
            image_path=demo_image_path,
            window_size=args.window_size,
            stride=args.stride,
            space_threshold=args.threshold,
            save_dir=demo_dir
        )
    else:
        # Run on user-provided image
        import os
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            exit(1)
        
        visualize_patches_black_detection(
            image_path=args.image,
            window_size=args.window_size,
            stride=args.stride,
            space_threshold=args.threshold,
            save_dir=args.save_dir
        )
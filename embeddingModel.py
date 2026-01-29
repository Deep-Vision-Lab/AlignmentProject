from Parameters import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
import math

import gc


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



# Transformer-based model for processing token sequences
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    
    def forward(self, tokens):
        tokens = self.pos_encoder(tokens)
        return self.transformer_encoder(tokens)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.to(device)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  
        return x


class EmbeddingModel(nn.Module):
    def __init__(self, window_size=128, stride=64, vector_size=512, model_arch='CNN-Transformer',
                  device='cuda', use_bilstm=True, use_positional_encoding=True, 
                  positional_encoding_type='learnable', bilstm_layers=2, dropout=0.1,
                  use_space_gate=True, space_threshold=0.05):
        """
        Image embedding model that extracts patch-level features.
        
        Args:
            window_size: Size of sliding window for patch extraction
            stride: Stride for sliding window
            vector_size: Dimension of output feature vectors
            model_arch: Architecture type ('CNN', 'CNN-Transformer', 'dinov2', 'Transformer')
            device: Device to run on ('cuda' or 'cpu')
            use_bilstm: Enable BiLSTM for sequence context
            use_positional_encoding: Enable positional encoding
            positional_encoding_type: 'learnable' or 'sinusoidal'
            bilstm_layers: Number of BiLSTM layers
            dropout: Dropout rate
            use_space_gate: Enable black patch detection gate (recommended)
            space_threshold: Pixel intensity threshold for detecting black patches (0-1)
        """
        super(EmbeddingModel, self).__init__()
        
        self.model_arch = model_arch
        self.device = device
        self.use_bilstm = use_bilstm
        self.use_positional_encoding = use_positional_encoding
        
        # ============================================================================
        # SPACE GATE: Detect black patches and inject space embedding
        # ============================================================================
        self.use_space_gate = use_space_gate
        self.space_threshold = space_threshold  # Patches with mean pixel < threshold are "space"
        
        # Learnable space embedding - will be matched against <SPACE> token in text
        # This is injected when a black patch is detected, bypassing the CNN
        if use_space_gate:
            self.space_embedding = nn.Parameter(torch.zeros(vector_size))
            # Initialize to small random values so it can be learned
            nn.init.normal_(self.space_embedding, mean=0.0, std=0.02)
        
        if model_arch == 'dinov2':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.dinov2 = self.dinov2.to(device)
            # dinov2_vits14 outputs 384 features
            self.feature_proj = nn.Linear(384, vector_size).to(device)
            self.dinov2_batch_size = 32  # Process patches in mini-batches to save memory
            # CNN to merge batch and windows dimensions instead of reshape
            self.patch_conv = nn.Conv2d(3, 1,kernel_size=3, padding=1).to(device)

        if model_arch == 'CNN-Transformer' or model_arch == 'CNN':
            self.cnn_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            cnn_feature_dim = self.cnn_encoder.fc.in_features
            self.cnn_encoder.fc = nn.Linear(cnn_feature_dim, vector_size)
            self.cnn_encoder = self.cnn_encoder.to(device)

        if model_arch == 'CNN-Transformer' or model_arch == 'Transformer':
            self.transformer_encoder = TransformerEncoder(d_model=vector_size, nhead=8, num_layers=6).to(device)
        
        if model_arch == 'Transformer':
            self.channel_reducer = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False).to(device)
            # Fix the initialization issue - we'll handle this in forward()
            self.convert_to_vectors = None
        
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
        if use_positional_encoding:
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
        else:
            self.positional_encoding = None
            
        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
    
    def detect_black_patches(self, patches):
        """
        Detect which patches are "black" (empty/space) based on pixel intensity.
        
        Black patches have very low pixel values (near 0) and should map to the
        <SPACE> token in text alignment. This creates a hard gate that bypasses
        the CNN for empty regions.
        
        Args:
            patches: [B, num_patches, C, H, W] - extracted image patches
        
        Returns:
            is_black: [B, num_patches] - boolean mask, True where patch is black/space
        """
        if not self.use_space_gate:
            return None
        
        # Compute mean pixel intensity per patch
        # patches: [B, num_patches, C, H, W]
        # We want mean across channels, height, width -> [B, num_patches]
        patch_means = patches.abs().mean(dim=(2, 3, 4))  # Use abs() for normalized images
        
        # Black patches have very low mean intensity
        is_black = patch_means < self.space_threshold
        
        return is_black
    
    def apply_space_gate(self, features, is_black_mask):
        """
        Replace features of black patches with the learned space embedding.
        
        This is the "gating" mechanism that forces black patches to output
        a consistent space vector, regardless of what the CNN produces.
        
        Args:
            features: [B, num_patches, vector_size] - CNN output features
            is_black_mask: [B, num_patches] - boolean mask for black patches
        
        Returns:
            gated_features: [B, num_patches, vector_size] - features with space injection
        """
        if is_black_mask is None or not self.use_space_gate:
            return features
        
        # Clone to avoid modifying original
        gated_features = features.clone()
        
        # Expand space embedding for broadcasting: [vector_size] -> [1, 1, vector_size]
        # Ensure it's on the same device as features
        space_vec = self.space_embedding.to(features.device).view(1, 1, -1)
        
        # Create mask for broadcasting: [B, num_patches] -> [B, num_patches, 1]
        mask_expanded = is_black_mask.unsqueeze(-1).float().to(features.device)
        
        # Blend: where mask is 1 (black), use space_embedding; otherwise use CNN features
        # This is differentiable thanks to soft masking
        gated_features = (1 - mask_expanded) * features + mask_expanded * space_vec
        
        return gated_features
        
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

    def _process_transformer_branch(self, tokens_a, tokens_b, show_dims=False):
        """Process Transformer-only branch"""
        batches_num, windows_num, Channels, H, W = tokens_a.shape
        
        # Initialize convert_to_vectors if needed
        if self.convert_to_vectors is None:
            self.convert_to_vectors = nn.Sequential(
                nn.Linear(in_features=H, out_features=self.vector_size)
            ).to(tokens_a.device)
        
        # Channel reduction
        reduced_tokens_a = self.channel_reducer(tokens_a)
        reduced_tokens_b = self.channel_reducer(tokens_b)
        if show_dims: 
            print(f"After channel reduction: {reduced_tokens_a.shape}, {reduced_tokens_b.shape}")
        
        del tokens_a, tokens_b
        
        # Mean operations
        mean_tokens_a = reduced_tokens_a.mean(dim=2)
        mean_tokens_b = reduced_tokens_b.mean(dim=2)
        del reduced_tokens_a, reduced_tokens_b
        
        mean_a = mean_tokens_a.mean(dim=-1)
        mean_b = mean_tokens_b.mean(dim=-1)
        del mean_tokens_a, mean_tokens_b
        
        # Reshape for processing
        reshaped_means_a = mean_a.reshape(-1, mean_a.shape[-1])
        reshaped_means_b = mean_b.reshape(-1, mean_b.shape[-1])
        del mean_a, mean_b
        
        # Convert to vectors
        encoded_tokens_a = self.convert_to_vectors(reshaped_means_a)
        encoded_tokens_b = self.convert_to_vectors(reshaped_means_b)
            
        del reshaped_means_a, reshaped_means_b
        
        return encoded_tokens_a, encoded_tokens_b, batches_num, windows_num

    def _process_transformer_encoder(self, encoded_tokens_a, encoded_tokens_b, show_dims=False):
        """Process transformer encoder"""
        # Standard forward pass
        featured_tokens_a = self.transformer_encoder(encoded_tokens_a)
        featured_tokens_b = self.transformer_encoder(encoded_tokens_b)
            
        if show_dims: 
            print(f"After transformer: {featured_tokens_a.shape}, {featured_tokens_b.shape}")
        
        del encoded_tokens_a, encoded_tokens_b
        
        return featured_tokens_a, featured_tokens_b

    def _process_dinov2_branch(self, tokens_a, tokens_b, show_dims=False):
        """Process DINOv2 branch"""
        batches_num, windows_num, Channels, H, W = tokens_a.shape
        
        # Use CNN to process patches - flatten batch and windows, apply conv, then process
        # First flatten to [B*W, C, H, W] using view (preserves gradients)
        tokens_a = tokens_a.reshape(batches_num * windows_num, Channels, H, W)
        tokens_b = tokens_b.reshape(batches_num * windows_num, Channels, H, W)
        if show_dims:
            print(f"Patches after reshaping: {tokens_a.shape}, {tokens_b.shape}")
        # # Apply learnable CNN projection
        # tokens_a = self.patch_conv(tokens_a).squeeze()
        # tokens_b = self.patch_conv(tokens_b).squeeze()
        # if show_dims: 
        #     print(f"Patches after patch conv: {tokens_a.shape}, {tokens_b.shape}")

        # # Reshape back to [B, W, H, W] for further processing
        # tokens_a = tokens_a.reshape(batches_num, windows_num, H, W)
        # tokens_b = tokens_b.reshape(batches_num, windows_num, H, W)

        # DINOv2 requires input dimensions to be multiples of 14 (patch size)
        # Resize patches to nearest multiple of 14 that's >= current size
        target_size = max(14, ((max(H, W) + 13) // 14) * 14)
        tokens_a = F.interpolate(tokens_a, size=(target_size, target_size), mode='bilinear', align_corners=False)
        tokens_b = F.interpolate(tokens_b, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        if show_dims:
            print(f"Patches after resizing for DINOv2: {tokens_a.shape}, {tokens_b.shape}")
        
        # Run DINOv2
        feat_a = self.dinov2(tokens_a)
        feat_b = self.dinov2(tokens_b)
        
        # Project to desired vector size
        featured_tokens_a = self.feature_proj(feat_a)
        featured_tokens_b = self.feature_proj(feat_b)
        
        if show_dims:
            print(f"After DINOv2: {featured_tokens_a.shape}, {featured_tokens_b.shape}")
        
        # Reshape to [batch, windows, vector_size]
        features_vector_a = featured_tokens_a.reshape(batches_num, windows_num, self.vector_size)
        features_vector_b = featured_tokens_b.reshape(batches_num, windows_num, self.vector_size)
        
        del featured_tokens_a, featured_tokens_b
        
        return features_vector_a, features_vector_b

    def forward(self, image_a, image_b, show_dims=False, debug=False):
        # Extract patches
        tokens_a = sliding_window(image_a, self.window_size, self.stride, debug_mode=debug).to(device)
        tokens_b = sliding_window(image_b, self.window_size, self.stride, debug_mode=debug).to(device)
        if show_dims: 
            print(f"Patches: {tokens_a.shape}, {tokens_b.shape}")
        
        batches_num, windows_num = tokens_a.shape[:2]
        
        # ============================================================
        # SPACE GATE: Detect black/empty patches before processing
        # These will be replaced with space embeddings after CNN
        # ============================================================
        is_black_a = self.detect_black_patches(tokens_a) if self.use_space_gate else None
        is_black_b = self.detect_black_patches(tokens_b) if self.use_space_gate else None
        
        if show_dims and is_black_a is not None:
            num_black_a = is_black_a.sum().item()
            num_black_b = is_black_b.sum().item()
            print(f"Black patches detected: A={num_black_a}, B={num_black_b}")
        
        # Process based on architecture
        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'CNN':
            encoded_tokens_a, encoded_tokens_b, batches_num, windows_num = self._process_cnn_branch(
                tokens_a, tokens_b, show_dims
            )
            
            if self.model_arch == 'CNN':
                # CNN-only: reshape to final output
                features_vector_a = encoded_tokens_a.view(batches_num, windows_num, self.vector_size)
                features_vector_b = encoded_tokens_b.view(batches_num, windows_num, self.vector_size)
                del encoded_tokens_a, encoded_tokens_b
                
                # ============================================================
                # SPACE GATE: Replace black patch features with space embedding
                # This happens BEFORE BiLSTM so context can flow around spaces
                # ============================================================
                if self.use_space_gate:
                    features_vector_a = self.apply_space_gate(features_vector_a, is_black_a)
                    features_vector_b = self.apply_space_gate(features_vector_b, is_black_b)
                
                # ============================================================
                # OPTIMIZATION 3: Apply BiLSTM for local sequence context
                # This allows information to flow between neighboring patches
                # ============================================================
                if self.bilstm_context is not None:
                    features_vector_a = self.bilstm_context(features_vector_a)
                    features_vector_b = self.bilstm_context(features_vector_b)
                
                # ============================================================
                # OPTIMIZATION 1: Add positional encoding
                # Assigns order information (1st, 2nd, 3rd patch...)
                # ============================================================
                if self.positional_encoding is not None:
                    features_vector_a = self.positional_encoding(features_vector_a)
                    features_vector_b = self.positional_encoding(features_vector_b)
                
                return features_vector_a, features_vector_b
                
        elif self.model_arch == 'Transformer':
            encoded_tokens_a, encoded_tokens_b, batches_num, windows_num = self._process_transformer_branch(
                tokens_a, tokens_b, show_dims
            )
        elif self.model_arch == 'dinov2':
            # DINOv2 model processing
            features_vector_a, features_vector_b = self._process_dinov2_branch(
                tokens_a, tokens_b, show_dims
            )
        # CNN-Transformer or Transformer: process through transformer encoder
        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'Transformer':
            featured_tokens_a, featured_tokens_b = self._process_transformer_encoder(
                encoded_tokens_a, encoded_tokens_b, show_dims
            )
        return features_vector_a, features_vector_b


# Example usage
if __name__ == "__main__":
    # Simulate grayscale image inputs
    image_a = torch.randn(4, 3, 128, 1024).to('cuda') # Batch of 32 grayscale images
    image_b = torch.randn(4, 3, 128, 1024).to('cuda')
    
    # Instantiate the alignment model
    model = EmbeddingModel(
        window_size=16,
        stride=8, 
        vector_size=64,
        model_arch='dinov2'
    ).to('cuda') # ['CNN-Transformer','CNN', 'dinov2', 'Transformer']

    # Forward pass: get token sequences for both images
    tokens_a, tokens_b = model(image_a, image_b,  show_dims=True)
    
    # Output token shapes
    print(f"Tokens A shape: {tokens_a.shape}")
    print(f"Tokens B shape: {tokens_b.shape}")
    del image_a, image_b 
    del tokens_a, tokens_b, 
    del model
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass
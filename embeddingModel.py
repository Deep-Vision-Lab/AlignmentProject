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
                  positional_encoding_type='learnable', bilstm_layers=2, dropout=0.1):
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
        
        # ==================== PHASE: Positional Encoding (BEFORE BiLSTM) ====================
        # Inject position info so BiLSTM can distinguish "1st alif" from "3rd alif"
        t.start('img_1e_positional_encoding')
        if self.use_positional_encoding:
            features_vector_a = self.positional_encoding(features_vector_a)
            features_vector_b = self.positional_encoding(features_vector_b)
        t.stop('img_1e_positional_encoding')
        
        # ==================== PHASE: BiLSTM Context ====================
        t.start('img_1f_bilstm')
        if self.use_bilstm:
            features_vector_a = self.bilstm_context(features_vector_a)
            features_vector_b = self.bilstm_context(features_vector_b)
        t.stop('img_1f_bilstm')
        
        return features_vector_a, features_vector_b

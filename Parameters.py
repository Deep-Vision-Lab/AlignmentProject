import torch

# Model parameters
loss_type = 'ContrastiveSoftDTW'  # ['MSE', 'ContrastiveSoftDTW']
model_arch = 'CNN' # ['CNN-Transformer', 'CNN', 'dinov2', 'Transformer']
normalize_type = 'average' # ['min_max', 'mean_std', 'average']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 4
epochs = 300
learning_rate = 1e-4

# Data parameters
window_size = 16
vector_size = 128
lang = 'Arabic' # ['English', 'Arabic']

# ============================================================================
# Sliding Window with Overlap
# ============================================================================
# Using overlap creates redundant patches that make letter features more robust.
# stride_ratio = 0.5 means 50% overlap (stride = window_size // 2)
# stride_ratio = 0.25 means 75% overlap (stride = window_size // 4)
stride_ratio = 0.5  # Recommended: 0.5 (50% overlap) or 0.25 (75% overlap)

# ============================================================================
# ARCHITECTURE FLAGS
# ============================================================================
# Positional Encoding (Critical for Transformers)
use_positional_encoding = True
positional_encoding_type = 'learnable'  # ['learnable', 'sinusoidal']

# BiLSTM for local sequence context
use_bilstm = True
bilstm_layers = 2

# ============================================================================
# Space/Black Patch Handling
# ============================================================================
# When enabled, black patches (spaces between words) are detected and mapped
# to a special <SPACE> embedding instead of being processed by CNN.
use_space_gate = True           # Enable black patch detection gate
space_threshold = 0.05          # Patches with mean pixel intensity < threshold are "space"
include_spaces = True           # Include space characters in text embeddings

# ============================================================================
# Contrastive Soft-DTW Parameters (CUDA-accelerated)
# ============================================================================
# Used when loss_type = 'ContrastiveSoftDTW'
# Combines Soft-DTW with InfoNCE-style contrastive learning
# Uses CUDA-accelerated Soft-DTW from soft-dtw-cuda.py
# All contrastive logic is inside the loss function - no special training loop needed
contrastive_soft_dtw_gamma = 0.1            # Soft-DTW smoothing (gamma -> 0: hard DTW, gamma -> inf: average)
contrastive_soft_dtw_temperature = 0.1      # InfoNCE temperature (lower = sharper distinction between pairs)
contrastive_soft_dtw_mse_weight = 1.0       # Weight for MSE reconstruction loss
contrastive_soft_dtw_contrastive_weight = 0.5  # Weight for contrastive DTW loss
# Note: Normalization is disabled because image and text have different sequence lengths

# Dropout for regularization
model_dropout = 0.0

# Debugging and visualization parameters
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = True # Set to True to log training to Weights & Biases
show_gradients = False # Set to True to print gradients for debugging
preprocess = 'none' # ['none', 'Normalize', 'ExtractPatches']
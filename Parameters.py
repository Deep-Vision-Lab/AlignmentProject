import torch

# Model parameters
normalize_type = 'l2' # ['min_max', 'mean_std', 'average', 'l2']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 64
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
# Positional Encoding (Disabled - BiLSTM provides relative position context)
# Keeping this ON would break staircase for repeated letters since
# static text embeddings don't have position info
use_positional_encoding = False
positional_encoding_type = 'sinusoidal'  # ['learnable', 'sinusoidal']

# BiLSTM for local sequence context
use_bilstm = True
bilstm_layers = 2

# ============================================================================
# Contrastive Soft-DTW Parameters (CUDA-accelerated)
# ============================================================================
# Used when loss_type = 'ContrastiveSoftDTW'
# Combines Soft-DTW with InfoNCE-style contrastive learning
# Uses CUDA-accelerated Soft-DTW from soft-dtw-cuda.py
# All contrastive logic is inside the loss function - no special training loop needed
contrastive_soft_dtw_gamma = 0.1          # Soft-DTW smoothing (gamma -> 0: hard DTW, gamma -> inf: average)
contrastive_soft_dtw_gamma_min = 0.001    # Minimum gamma after annealing
contrastive_soft_dtw_gamma_decay = 0.95   # Multiplicative decay per epoch

# Sakoe-Chiba Band: restricts DTW warping path to a diagonal band
# Prevents distant identical letters from being incorrectly aligned
# Set as a fraction of sequence length (e.g., 0.2 = 20% of image width)
sakoe_chiba_bandwidth_ratio = 0.0        # Bandwidth as fraction of sequence length (0 = no band constraint)
contrastive_margin = 10.0                 # Margin for triplet loss: forces pos_cost to beat neg_cost by this amount


# Dropout for regularization
model_dropout = 0.0

# Negative sampling
num_negatives = 20  # Number of negative samples per positive pair (in-batch negatives)

# Debugging and visualization parameters
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = True # Set to True to log training to Weights & Biases
show_gradients = False # Set to True to print gradients for debugging
preprocess = 'none' # ['none', 'Normalize', 'ExtractPatches']
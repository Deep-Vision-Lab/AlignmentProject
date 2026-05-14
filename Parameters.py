import torch
import os

# Model parameters
normalize_type = 'l2' # ['min_max', 'mean_std', 'average', 'l2']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 32
epochs = 100
learning_rate = 1e-4

# Data parameters
window_size = 16
vector_size = 128
lang = 'Arabic' # ['English', 'Arabic']
num_samples = 100000  # [10000, 50000, 100000]

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
# BiLSTM for local sequence context (CRNN architecture)
use_bilstm = False
bilstm_layers = 1

# ============================================================================
# Contrastive Soft-DTW Parameters (CUDA-accelerated)
# ============================================================================
# Used when loss_type = 'ContrastiveSoftDTW'
# Combines Soft-DTW with InfoNCE-style contrastive learning
# Uses CUDA-accelerated Soft-DTW from soft-dtw-cuda.py
# All contrastive logic is inside the loss function - no special training loop needed
contrastive_soft_dtw_gamma = 1e-4          # Soft-DTW smoothing (gamma -> 0: hard DTW, gamma -> inf: average)

# Sakoe-Chiba Band: restricts DTW warping path to a diagonal band
# Prevents distant identical letters from being incorrectly aligned
# Set as a fraction of sequence length (e.g., 0.2 = 20% of image width)
sakoe_chiba_bandwidth_ratio = 0.0        # Bandwidth as fraction of sequence length (0 = no band constraint)
contrastive_margin = 100.0                 # Margin for triplet loss: forces pos_cost to beat neg_cost by this amount


# Dropout for regularization
model_dropout = 0.3

# Negative sampling
num_negatives = 10  # Number of negative samples per positive pair (in-batch negatives)

# Needleman-Wunsch / alignment scoring parameters
matchScore    =  10    # reward for a matching character/patch
mismatchScore = -27    # penalty for a mismatch
gapScore      = -10    # penalty per gap step

# ============================================================================
# MULTI-SCALE ALIGNMENT (Multi-Resolution Loss)
# ============================================================================
# Computes Contrastive Soft-DTW at two window sizes simultaneously:
#   - Large window (macro): learns global structure (word spacing, ascenders)
#   - Small window (micro): learns fine-grained details (dots, diacritics)
# Loss_total = Loss_norm(macro) + alpha * Loss_norm(micro)
multi_scale_enabled = os.environ.get('MULTI_SCALE_ENABLED', 'False').lower() in ('true', '1', 't')
multi_scale_window_sizes = [16, 8]   # [macro_window, micro_window]
multi_scale_alpha = 0.5              # Weight for micro-scale loss (start at 0.5, tune later)

# Debugging and visualization parameters
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = True # Set to True to log training to Weights & Biases
show_gradients = False # Set to True to print gradients for debugging
preprocess = 'none' # ['none', 'Normalize', 'ExtractPatches']
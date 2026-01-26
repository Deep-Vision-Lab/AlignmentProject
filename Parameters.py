import torch

# Model parameters
loss_type = 'ContrastiveMSE'  # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 
                               # 'Dice', 'Wasserstein', 'SoftCrossEntropy', 
                               # 'DiagonalAlignment', 'CombinedAlignment', 'MSEWithDiagonalReg',
                               # 'ContrastiveMSE']  # <-- NEW: Contrastive loss for letter classification
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
# OPTIMIZATION 4: Sliding Window with Overlap
# ============================================================================
# Using overlap creates redundant patches that make letter features more robust.
# stride_ratio = 0.5 means 50% overlap (stride = window_size // 2)
# stride_ratio = 0.25 means 75% overlap (stride = window_size // 4)
# Effect: Creates smoother sequences, letters won't be cut in half
stride_ratio = 0.5  # Recommended: 0.5 (50% overlap) or 0.25 (75% overlap)

# ============================================================================
# ARCHITECTURE OPTIMIZATION FLAGS
# ============================================================================
# OPTIMIZATION 1: Positional Encoding (Critical for Transformers)
use_positional_encoding = True
positional_encoding_type = 'learnable'  # ['learnable', 'sinusoidal']

# OPTIMIZATION 3: BiLSTM for local sequence context
use_bilstm = True
bilstm_layers = 2

# OPTIMIZATION 2: Diagonal/Monotonic constraint weights (used in CombinedAlignment loss)
diagonal_constraint_weight = 0.3
monotonic_constraint_weight = 0.2
mse_weight = 1.0

# ============================================================================
# OPTIMIZATION 5: Space/Black Patch Handling
# ============================================================================
# When enabled, black patches (spaces between words) are detected and mapped
# to a special <SPACE> embedding instead of being processed by CNN.
# This guarantees 100% accuracy for space detection.
use_space_gate = True           # Enable black patch detection gate
space_threshold = 0.05          # Patches with mean pixel intensity < threshold are "space"
include_spaces = True           # Include space characters in text embeddings

# ============================================================================
# OPTIMIZATION 6: Contrastive Learning Parameters
# ============================================================================
# Used when loss_type = 'ContrastiveMSE'
# Teaches model to distinguish correct letter from wrong letters in batch
contrastive_margin = 0.3        # Margin for contrastive loss (how far apart negatives should be)
contrastive_weight = 0.5        # Weight of contrastive loss relative to MSE

# Dropout for regularization
model_dropout = 0.0

# Debugging and visualization parameters
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = True # Set to True to log training to Weights & Biases
show_gradients = False # Set to True to print gradients for debugging
preprocess = 'none' # Set to True to normalize before loss computation
                    # ['none', 'Normalize', 'ExtractPatches']
Regular_ScoreMatrix_Load = False  # Set to True to load regular score matrices, False to load diff NW matrices

# Needleman-Wunsch parameters
matchScore = 1
mismatchScore = -3
gapScore = -1

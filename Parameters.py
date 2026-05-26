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
num_samples = 50000  # [10000, 50000, 100000]

# ============================================================================
# Text Embedding Selection
# ============================================================================
# 'char'     -> learned per-character table (random init, frozen by default).
# 'fasttext' -> facebook/fasttext-ar-vectors pretrained vectors (frozen).
#               Adds a small Linear adapter if fastText's 300-d != vector_size.
text_embedder_type = os.environ.get('TEXT_EMBEDDER', 'fasttext').lower()
# Optional local path to a fasttext model.bin. If None, the file is
# auto-downloaded from HuggingFace (facebook/fasttext-ar-vectors).
text_embedder_model_path = os.environ.get('TEXT_EMBEDDER_MODEL_PATH') or None
# Where HuggingFace caches the downloaded model (None -> ~/.cache/huggingface).
text_embedder_cache_dir = os.environ.get('TEXT_EMBEDDER_CACHE_DIR') or None
# The fastText vectors are 300-d. If vector_size != 300, an adapter Linear is
# inserted after the lookup. The fastText weights themselves stay frozen;
# this flag only controls whether that small adapter is trainable.
fasttext_projection_trainable = True

# ============================================================================
# Fine-tuning parameters
# ============================================================================
# Used when train.py is launched with --finetune. The dataset must follow the
# same DataSet/Synthetic_{lang}_{num_samples}/ layout (images/, texts/, ...).
# Override these on the fly by passing --data_dir to train.py.
finetune_lang = 'English'                # Language of the finetune dataset
finetune_num_samples = 10000             # Size of the finetune dataset
finetune_data_dir = f'DataSet/Synthetic_{finetune_lang}_{finetune_num_samples}'
finetune_learning_rate = 1e-5            # Typically 1/10 of pretraining LR
finetune_epochs = 30                     # Usually fewer epochs than pretraining

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
use_bilstm = True
bilstm_layers = 1

# ============================================================================
# Contrastive Soft-DTW Parameters (CUDA-accelerated)
# ============================================================================
# Used when loss_type = 'ContrastiveSoftDTW'
# Combines Soft-DTW with InfoNCE-style contrastive learning
# Uses CUDA-accelerated Soft-DTW from soft-dtw-cuda.py
# All contrastive logic is inside the loss function - no special training loop needed
contrastive_soft_dtw_gamma = 0.1          # Soft-DTW smoothing (gamma -> 0: hard DTW, gamma -> inf: average)

# Sakoe-Chiba Band: the CUDA kernel implements `abs(i - j) <= bw`, which is
# only valid for square matrices. Our similarity matrices are rectangular
# (N text rows ≈ 30, M image cols ≈ 100-200), so any non-zero kernel band
# makes the natural diagonal path's endpoint unreachable → DTW returns inf
# → loss becomes NaN. We keep this param at 0 (disabled) and use the SOFT
# diagonal-band penalty below instead.
sakoe_chiba_bandwidth_ratio = 0.0
contrastive_margin = 1.0                  # Margin for triplet loss on length-normalized DTW costs
# Softmax temperature for converting cosine similarity (range [-1, 1]) into a
# peaked distribution over image patches. Without this, softmax is nearly
# uniform over ~100+ patches and the NLL distance saturates at log(N),
# erasing the alignment gradient signal.
contrastive_temperature = 0.07

# ---------- Diagonal prior (encourages the staircase) ----------
# Adds an auxiliary term that maximises <sim_pos, diagonal_gauss_mask>, pulling
# similarity UP on the expected diagonal stripe and DOWN off-diagonal. The
# contrastive margin alone is too weak — it allows fuzzy non-staircase solutions.
diagonal_prior_weight = 0.1              # Strength of the diagonal-prior term (0 disables it)
diagonal_prior_sigma_ratio = 0.15        # Gaussian width as fraction of max(N, M)

# Soft diagonal band added to the DTW distance matrix. Quadratic in normalised
# index distance from the expected diagonal j ≈ i·(M-1)/(N-1). Replaces the
# kernel-level Sakoe-Chiba band (which can't handle rectangular matrices).
# Set to 0 to disable. Reasonable range: 0.5 – 5.0.
dtw_band_penalty_weight = 2.0


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
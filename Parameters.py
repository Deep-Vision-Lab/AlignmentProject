import torch
import os

# Model parameters
normalize_type = 'l2' # ['min_max', 'mean_std', 'average', 'l2']
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 32
epochs = 100
learning_rate = 1e-4


def _env_value(name, default):
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    return value


def _env_float(name, default):
    return float(_env_value(name, default))


def _env_bool(name, default):
    value = str(_env_value(name, default)).lower()
    return value in ('true', '1', 't', 'yes')

# ============================================================================
# Alignment Loss Selection
# ============================================================================
alignment_loss_type = str(_env_value('ALIGNMENT_LOSS_TYPE', 'd3tw')).lower()

# CTC / hybrid weights and settings
ctc_weight = _env_float('CTC_WEIGHT', 1.0)
d3tw_weight = _env_float('D3TW_WEIGHT', 1.0)
ctc_blank_token = _env_value('CTC_BLANK_TOKEN', '<ctc_blank>')
ctc_zero_infinity = _env_bool('CTC_ZERO_INFINITY', True)
ctc_reduction = str(_env_value('CTC_REDUCTION', 'mean')).lower()
contrastive_ctc_loss_type = str(_env_value('CONTRASTIVE_CTC_LOSS_TYPE', 'infonce')).lower()
contrastive_ctc_tau = _env_float('CONTRASTIVE_CTC_TAU', 0.1)
contrastive_ctc_margin = _env_float('CONTRASTIVE_CTC_MARGIN', 0.2)
save_ctc_vocab = _env_bool('SAVE_CTC_VOCAB', True)

# Data parameters
# For clean synthetic data, window_size=16 works well.
# For Arabic manuscripts, consider window_size=32 or multi_scale_window_sizes=[32, 16]
# because connected strokes and dots may need larger visual context.
window_size = 16
vector_size = 128
lang = 'Arabic' # ['English', 'Arabic']
num_samples = 10000  # [10000, 50000, 100000]

# ============================================================================
# Text Embedding Selection
# ============================================================================
# 'char'               -> learned per-character table (random init, frozen).
# 'fasttext'           -> facebook/fasttext-ar-vectors pretrained vectors (frozen).
#                         Linguistic vectors; may not align with visual features.
# 'orthogonal_char'    -> deterministic unit-sphere random vectors (frozen).
#                         Recommended visual-alignment baseline: no linguistic
#                         bias, each character gets a unique stable direction.
# 'random_frozen_char' -> raw Gaussian random frozen vectors (ablation).
# 'shape_group_char'   -> shape-aware Arabic embeddings; visually-confusable
#                         letters (ب/ت/ث/ن/ي) blended toward a group centroid.
text_embedder_type = os.environ.get('TEXT_EMBEDDER', 'fasttext').lower()
# Optional local path to a fasttext model.bin. If None, the file is
# auto-downloaded from HuggingFace (facebook/fasttext-ar-vectors).
text_embedder_model_path = os.environ.get('TEXT_EMBEDDER_MODEL_PATH') or None
# Where HuggingFace caches the downloaded model (None -> ~/.cache/huggingface).
text_embedder_cache_dir = os.environ.get('TEXT_EMBEDDER_CACHE_DIR') or None
# The fastText vectors are 300-d. If vector_size != 300, an adapter Linear is
# inserted after the lookup. The fastText weights themselves stay frozen;
# this flag only controls whether that small adapter is trainable.
fasttext_projection_trainable = False

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
window_overlap_mode = str(_env_value('WINDOW_OVERLAP_MODE', 'custom')).lower()

# ============================================================================
# D3TW-guided Character Pooling
# ============================================================================
use_d3tw_char_pooling = _env_bool('USE_D3TW_CHAR_POOLING', False)
char_pool_weight = _env_float('CHAR_POOL_WEIGHT', 0.5)
char_pool_tau = _env_float('CHAR_POOL_TAU', 0.07)
char_pool_warmup_epochs = int(_env_value('CHAR_POOL_WARMUP_EPOCHS', 5))
char_pool_ramp_epochs = int(_env_value('CHAR_POOL_RAMP_EPOCHS', 10))
char_pool_method = str(_env_value('CHAR_POOL_METHOD', 'hard_mean')).lower()
char_pool_detach_alignment = _env_bool('CHAR_POOL_DETACH_ALIGNMENT', True)
char_pool_skip_spaces = _env_bool('CHAR_POOL_SKIP_SPACES', False)
char_pool_min_windows_per_char = int(_env_value('CHAR_POOL_MIN_WINDOWS_PER_CHAR', 1))
char_pool_use_char_bank = _env_bool('CHAR_POOL_USE_CHAR_BANK', True)

# ============================================================================
# ARCHITECTURE FLAGS
# ============================================================================
# Sequence context encoder after the CNN patch features.
# Options: "bilstm" (default/current), "transformer", "none".
sequence_encoder_type = str(_env_value('SEQUENCE_ENCODER_TYPE', 'bilstm')).lower()
# Backward-compatible flag for older scripts/checkpoints.
use_bilstm = sequence_encoder_type == "bilstm"
bilstm_layers = 2
# Hidden state size per direction in the BiLSTM.  The bidirectional output is
# bilstm_hidden_dim * 2, which is projected back to vector_size.
# Raising this from the old default (vector_size // 2 = 64) to vector_size (128)
# doubles the LSTM memory capacity.
bilstm_hidden_dim = vector_size  # 128

# Transformer sequence encoder options. Positional index 0 follows the model
# sequence order; for Arabic this already means the rightmost window after flip.
transformer_num_layers = int(_env_value('TRANSFORMER_NUM_LAYERS', 2))
transformer_num_heads = int(_env_value('TRANSFORMER_NUM_HEADS', 4))
transformer_ff_dim = int(_env_value('TRANSFORMER_FF_DIM', 512))
transformer_dropout = _env_float('TRANSFORMER_DROPOUT', 0.1)
transformer_activation = str(_env_value('TRANSFORMER_ACTIVATION', 'gelu')).lower()
transformer_norm_first = _env_bool('TRANSFORMER_NORM_FIRST', True)
transformer_positional_encoding = str(_env_value(
    'TRANSFORMER_POSITIONAL_ENCODING', 'sinusoidal'
)).lower()
transformer_max_len = int(_env_value('TRANSFORMER_MAX_LEN', 4096))
return_attention_weights = _env_bool('RETURN_ATTENTION_WEIGHTS', False)

# Optional auxiliary cross-attention similarity. Disabled by default and never
# required for image-only inference/evaluation.
use_cross_attention = _env_bool('USE_CROSS_ATTENTION', False)
cross_attention_type = str(_env_value('CROSS_ATTENTION_TYPE', 'text_to_image')).lower()
cross_attention_num_heads = int(_env_value('CROSS_ATTENTION_NUM_HEADS', 4))
cross_attention_dropout = _env_float('CROSS_ATTENTION_DROPOUT', 0.1)
cross_attention_weight = _env_float('CROSS_ATTENTION_WEIGHT', 0.0)

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
contrastive_margin = 10.0                  # Margin for triplet loss on length-normalized DTW costs
# Softmax temperature for converting cosine similarity (range [-1, 1]) into a
# peaked distribution over image patches. Without this, softmax is nearly
# uniform over ~100+ patches and the NLL distance saturates at log(N),
# erasing the alignment gradient signal.
contrastive_temperature = 0.07

# ---------- Diagonal prior (encourages the staircase) ----------
# Adds an auxiliary term that maximises <sim_pos, diagonal_gauss_mask>, pulling
# similarity UP on the expected diagonal stripe and DOWN off-diagonal. The
# contrastive margin alone is too weak — it allows fuzzy non-staircase solutions.
diagonal_prior_weight = 0.01              # Strength of the diagonal-prior term (0 disables it)
diagonal_prior_sigma_ratio = 0.15        # Gaussian width as fraction of max(N, M)

# Soft diagonal band added to the DTW distance matrix. Quadratic in normalised
# index distance from the expected diagonal j ≈ i·(M-1)/(N-1). Replaces the
# kernel-level Sakoe-Chiba band (which can't handle rectangular matrices).
# Set to 0 to disable. Reasonable range: 0.5 – 5.0.
dtw_band_penalty_weight = 2.0


# Dropout for regularization
model_dropout = 0.3

# ============================================================================
# Negative Sampling Mode
# ============================================================================
# 'mixed'              -> current default: crop/drop/shuffle + random in-batch.
# 'length_controlled'  -> word-shuffle + same-length alternatives. Prevents
#                         length bias where shorter negatives get unfairly lower
#                         DTW cost. Use for fig05/fig08 evaluation.
# 'dot_confusion'      -> substitutes visually-confusable Arabic letters
#                         (ب↔ت↔ث, ج↔ح↔خ, etc.). Hardest negatives for Arabic.
# 'same_length_random' -> random Arabic chars preserving exact char count.
# 'shuffle_only'       -> word shuffle only (no crop/drop).
negative_mode = os.environ.get('NEGATIVE_MODE', 'mixed').lower()

# Number of negative samples per positive pair (in-batch negatives)
num_negatives = 10

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
multi_scale_window_sizes = [32, 16]   # [macro_window, micro_window]  — see window_size comment above
multi_scale_alpha = 0.5              # Weight for micro-scale loss (start at 0.5, tune later)

# ============================================================================
# Blank / Transition Token Support
# ============================================================================
# use_blank_token=True prepends/appends a boundary space so alignment can
# assign background windows to a non-character anchor (like CTC blank).
# blank_insert_mode controls where blanks are inserted beyond boundaries:
#   'boundaries'    -> only at start and end of transcript (default, safe).
#   'between_words' -> also between every word.
#   'none'          -> disabled (fall back to existing pad_text behaviour).
use_blank_token    = False
blank_insert_mode  = "boundaries"  # boundaries | between_words | none

# ============================================================================
# Loss Type
# ============================================================================
# 'margin'   -> current: triplet margin loss on normalised DTW costs.
# 'infonce'  -> InfoNCE cross-entropy over [pos, neg_1…neg_K] costs.
# 'hybrid'   -> margin + lambda_infonce * infonce (recommended).
contrastive_loss_type   = os.environ.get('LOSS_TYPE', 'margin').lower()
contrastive_infonce_tau = 0.1   # temperature for InfoNCE softmax
lambda_infonce          = 1.0   # weight for InfoNCE term in hybrid mode

# ============================================================================
# DTW Cost Normalization
# ============================================================================
# Normalising by the longer sequence (max_len) is the standard approach.
# Using 'path_len' normalises by the actual aligned path length instead —
# fairer when positive and negative transcripts have very different lengths.
# Options: 'max_len' (current) | 'path_len' | 'text_len' | 'image_len'
dtw_cost_normalization = os.environ.get('DTW_NORM', 'max_len').lower()

# Debugging and visualization parameters
debug = True # Set to True to save patches and heatmaps for debugging
debug_wandb = True # Set to True to log training to Weights & Biases
show_gradients = False # Set to True to print gradients for debugging
preprocess = 'none' # ['none', 'Normalize', 'ExtractPatches']

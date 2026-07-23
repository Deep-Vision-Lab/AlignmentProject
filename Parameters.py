import os

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# Training
batch_size = int(os.environ.get("BATCH_SIZE", 32))
epochs = int(os.environ.get("EPOCHS", 35))
learning_rate = float(os.environ.get("LEARNING_RATE", 1e-4))
valid_every_n_epochs = int(os.environ.get("VALID_EVERY_N_EPOCHS", 2))
valid_max_batches = int(os.environ.get("VALID_MAX_BATCHES", 20))
log_memory_every_n_batches = int(os.environ.get("LOG_MEMORY_EVERY_N_BATCHES", 25))

# Data
window_size = int(os.environ.get("WINDOW_SIZE", 32))
stride_ratio = float(os.environ.get("STRIDE_RATIO", 0.5))
window_overlap_mode = os.environ.get("WINDOW_OVERLAP_MODE", "custom").lower()
vector_size = int(os.environ.get("VECTOR_SIZE", 128))
lang = os.environ.get("LANGUAGE", "Arabic")
num_samples = int(os.environ.get("NUM_SAMPLES", 8000))
text_encoder_type = os.environ.get("TEXT_ENCODER_TYPE", "arabic_span").lower()
arabic_text_model_name = os.environ.get(
    "ARABIC_TEXT_MODEL_NAME",
    "aubmindlab/bert-base-arabertv02",
)

# Span semantics. A core span is limited to two visible characters. One following
# character may be retained separately as overlap context, but only for a
# one-character core. This prevents a two-character window from being trained or
# displayed as an unrelated three-character core.
max_text_token_chars = int(os.environ.get("MAX_TEXT_TOKEN_CHARS", 2))
max_text_span_chars = int(os.environ.get("MAX_TEXT_SPAN_CHARS", 2))
max_windows_per_span = int(os.environ.get("MAX_WINDOWS_PER_SPAN", 3))
span_boundary_context_chars = int(os.environ.get("SPAN_BOUNDARY_CONTEXT_CHARS", 1))
span_boundary_context_max_core_chars = int(
    os.environ.get("SPAN_BOUNDARY_CONTEXT_MAX_CORE_CHARS", 1)
)
span_include_space_context = os.environ.get("SPAN_INCLUDE_SPACE_CONTEXT", "0").lower() in {
    "1", "true", "yes", "on"
}
span_allow_character_space_surfaces = os.environ.get(
    "SPAN_ALLOW_CHARACTER_SPACE_SURFACES", "0"
).lower() in {"1", "true", "yes", "on"}
span_space_token = os.environ.get("SPAN_SPACE_TOKEN", "<SPACE>")
strip_span_text_edges = os.environ.get("STRIP_SPAN_TEXT_EDGES", "1").lower() in {
    "1", "true", "yes", "on"
}

# Frozen span feature cache.
span_feature_cache_size = int(os.environ.get("SPAN_FEATURE_CACHE_SIZE", 2048))
span_feature_cache_dtype = os.environ.get("SPAN_FEATURE_CACHE_DTYPE", "float16").lower()
clear_span_cache_each_epoch = os.environ.get("CLEAR_SPAN_CACHE_EACH_EPOCH", "1").lower() in {
    "1", "true", "yes", "on"
}

# Fine-tuning
finetune_lang = "English"
finetune_num_samples = 10000
finetune_data_dir = f"DataSet/Synthetic_{finetune_lang}"
finetune_learning_rate = 1e-5
finetune_epochs = 30

# Image sequence encoder
use_bilstm = os.environ.get("USE_BILSTM", "1").lower() in {"1", "true", "yes", "on"}
bilstm_layers = int(os.environ.get("BILSTM_LAYERS", 2))
bilstm_hidden_dim = vector_size
# Three-window fusion is gated per window and feature before the BiLSTM.
use_local_window_grouping = os.environ.get("USE_LOCAL_WINDOW_GROUPING", "1").lower() in {
    "1", "true", "yes", "on"
}
local_group_size = int(os.environ.get("LOCAL_GROUP_SIZE", 3))

# Contrastive Soft-DTW
contrastive_soft_dtw_gamma = float(os.environ.get("CONTRASTIVE_SOFT_DTW_GAMMA", 0.1))
contrastive_margin = float(os.environ.get("CONTRASTIVE_MARGIN", 10.0))
contrastive_temperature = float(os.environ.get("CONTRASTIVE_TEMPERATURE", 0.07))
span_window_count_penalty = float(os.environ.get("SPAN_WINDOW_COUNT_PENALTY", 0.05))
span_space_max_windows = int(os.environ.get("SPAN_SPACE_MAX_WINDOWS", 2))
span_extra_windows_per_core = int(os.environ.get("SPAN_EXTRA_WINDOWS_PER_CORE", 1))
span_dtw_backend = os.environ.get("SPAN_DTW_BACKEND", "jax").lower()
span_dtw_bucket_text_lengths = os.environ.get("SPAN_DTW_BUCKET_TEXT_LENGTHS", "1").lower() in {
    "1", "true", "yes", "on"
}
span_dtw_text_bucket_size = int(os.environ.get("SPAN_DTW_TEXT_BUCKET_SIZE", 16))
span_dtw_max_text_bucket = int(os.environ.get("SPAN_DTW_MAX_TEXT_BUCKET", 256))
span_dtw_active_negatives_per_sample = int(os.environ.get("SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE", 4))

# Local hard negatives.
use_local_hard_negatives = os.environ.get("USE_LOCAL_HARD_NEGATIVES", "1").lower() in {
    "1", "true", "yes", "on"
}
local_hard_negative_weight = float(os.environ.get("LOCAL_HARD_NEGATIVE_WEIGHT", 0.25))
local_hard_negative_margin = float(os.environ.get("LOCAL_HARD_NEGATIVE_MARGIN", 0.35))
local_hard_negative_top_k = int(os.environ.get("LOCAL_HARD_NEGATIVE_TOP_K", 12))
local_hard_negative_exclude_radius = int(os.environ.get("LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS", 3))
# Keep pixel contrast at 0.15, but admit sparse dot/stroke windows at one percent ink.
local_hard_negative_min_ink = float(os.environ.get("LOCAL_HARD_NEGATIVE_MIN_INK", 0.01))
local_hard_negative_every_n_batches = int(os.environ.get("LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES", 2))
local_hard_negative_max_samples_per_batch = int(os.environ.get("LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH", 8))

# Image-image pair contrastive loss.
use_image_pair_contrastive = os.environ.get("USE_IMAGE_PAIR_CONTRASTIVE", "1").lower() in {
    "1", "true", "yes", "on"
}
image_pair_loss_weight = float(os.environ.get("IMAGE_PAIR_LOSS_WEIGHT", 0.40))
image_pair_margin = float(os.environ.get("IMAGE_PAIR_MARGIN", 0.40))
image_pair_top_k = int(os.environ.get("IMAGE_PAIR_TOP_K", 8))
image_text_loss_on_both_lines = os.environ.get("IMAGE_TEXT_LOSS_ON_BOTH_LINES", "1").lower() in {
    "1", "true", "yes", "on"
}
image_pair_every_n_batches = int(os.environ.get("IMAGE_PAIR_EVERY_N_BATCHES", 1))
image_pair_max_samples_per_batch = int(os.environ.get("IMAGE_PAIR_MAX_SAMPLES_PER_BATCH", 8))
sequence_consistency_loss_weight = float(os.environ.get("SEQUENCE_CONSISTENCY_LOSS_WEIGHT", 0.05))

# Embedding variance regularization.
image_variance_loss_weight = float(os.environ.get("IMAGE_VARIANCE_LOSS_WEIGHT", 0.01))
image_variance_target_std = float(os.environ.get("IMAGE_VARIANCE_TARGET_STD", 0.05))

# Negatives
negative_mode = os.environ.get("NEGATIVE_MODE", "mixed").lower()
num_negatives = int(os.environ.get("NUM_NEGATIVES", 4))
span_negative_grad_mode = os.environ.get("SPAN_NEGATIVE_GRAD_MODE", "hardest").lower()

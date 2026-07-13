import os

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# Training
batch_size = int(os.environ.get("BATCH_SIZE", 32))
epochs = int(os.environ.get("EPOCHS", 100))
learning_rate = float(os.environ.get("LEARNING_RATE", 1e-4))
valid_every_n_epochs = int(os.environ.get("VALID_EVERY_N_EPOCHS", 1))
valid_max_batches = int(os.environ.get("VALID_MAX_BATCHES", 0))
log_memory_every_n_batches = int(os.environ.get("LOG_MEMORY_EVERY_N_BATCHES", 25))

# Data
window_size = int(os.environ.get("WINDOW_SIZE", 16))
stride_ratio = float(os.environ.get("STRIDE_RATIO", 0.5))
window_overlap_mode = os.environ.get("WINDOW_OVERLAP_MODE", "custom").lower()
vector_size = int(os.environ.get("VECTOR_SIZE", 128))
lang = os.environ.get("LANGUAGE", "Arabic")
num_samples = int(os.environ.get("NUM_SAMPLES", 10000))
text_encoder_type = os.environ.get("TEXT_ENCODER_TYPE", "char").lower()
arabic_text_model_name = os.environ.get(
    "ARABIC_TEXT_MODEL_NAME",
    "aubmindlab/bert-base-arabertv02",
)
max_text_token_chars = int(os.environ.get("MAX_TEXT_TOKEN_CHARS", 2))
max_text_span_chars = int(os.environ.get("MAX_TEXT_SPAN_CHARS", 2))
max_windows_per_span = int(os.environ.get("MAX_WINDOWS_PER_SPAN", 4))
strip_span_text_edges = os.environ.get("STRIP_SPAN_TEXT_EDGES", "1").lower() in {
    "1", "true", "yes", "on"
}

# Frozen span feature cache. The AraBERT backbone is frozen, so caching pooled
# backbone features is safe for eval/inference, but generated training negatives
# are often unique. Keep the cache bounded and store it on CPU in reduced
# precision by default.
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

# Contrastive Soft-DTW
contrastive_soft_dtw_gamma = float(os.environ.get("CONTRASTIVE_SOFT_DTW_GAMMA", 0.1))
contrastive_margin = float(os.environ.get("CONTRASTIVE_MARGIN", 10.0))
contrastive_temperature = float(os.environ.get("CONTRASTIVE_TEMPERATURE", 0.07))
span_dtw_backend = os.environ.get("SPAN_DTW_BACKEND", "torch").lower()
span_dtw_bucket_text_lengths = os.environ.get("SPAN_DTW_BUCKET_TEXT_LENGTHS", "1").lower() in {
    "1", "true", "yes", "on"
}
span_dtw_text_bucket_size = int(os.environ.get("SPAN_DTW_TEXT_BUCKET_SIZE", 16))
span_dtw_max_text_bucket = int(os.environ.get("SPAN_DTW_MAX_TEXT_BUCKET", 256))

# Local hard negatives.
# Stronger defaults because the heatmaps showed many scattered false-positive
# windows above 0.8 cosine. This trains the local pre-BiLSTM CNN windows to be
# more discriminative.
use_local_hard_negatives = os.environ.get("USE_LOCAL_HARD_NEGATIVES", "1").lower() in {
    "1", "true", "yes", "on"
}
local_hard_negative_weight = float(os.environ.get("LOCAL_HARD_NEGATIVE_WEIGHT", 0.4))
local_hard_negative_margin = float(os.environ.get("LOCAL_HARD_NEGATIVE_MARGIN", 0.35))
local_hard_negative_top_k = int(os.environ.get("LOCAL_HARD_NEGATIVE_TOP_K", 12))
local_hard_negative_exclude_radius = int(os.environ.get("LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS", 3))
local_hard_negative_min_ink = float(os.environ.get("LOCAL_HARD_NEGATIVE_MIN_INK", 0.05))
# Run local hard-negative path mining every N batches. 2 is a good speed/quality
# tradeoff because it halves Python hard-path mining while keeping the signal.
local_hard_negative_every_n_batches = int(os.environ.get("LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES", 2))
# When the local hard-negative loss runs, only mine hard DTW paths for a rotating
# subset of the batch. The 2026-07-14 logs showed this path miner is the main
# remaining bottleneck: local batches were ~50s vs ~23s without local mining.
local_hard_negative_max_samples_per_batch = int(os.environ.get("LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH", 8))

# Image-image pair contrastive loss.
# Uses img1/text1 and img2/text2 from the same sample. DTW gives pseudo span-window
# regions in both images. Regions with matching span text are positives; other
# regions are hard negatives. This directly targets line2-part-in-line1 retrieval.
use_image_pair_contrastive = os.environ.get("USE_IMAGE_PAIR_CONTRASTIVE", "1").lower() in {
    "1", "true", "yes", "on"
}
image_pair_loss_weight = float(os.environ.get("IMAGE_PAIR_LOSS_WEIGHT", 0.2))
image_pair_margin = float(os.environ.get("IMAGE_PAIR_MARGIN", 0.35))
image_pair_top_k = int(os.environ.get("IMAGE_PAIR_TOP_K", 8))
# Fast image-pair controls used by train_fast_image_pair.py.
image_text_loss_on_both_lines = os.environ.get("IMAGE_TEXT_LOSS_ON_BOTH_LINES", "0").lower() in {
    "1", "true", "yes", "on"
}
image_pair_every_n_batches = int(os.environ.get("IMAGE_PAIR_EVERY_N_BATCHES", 1))
image_pair_max_samples_per_batch = int(os.environ.get("IMAGE_PAIR_MAX_SAMPLES_PER_BATCH", 8))
# Keep sequence consistency optional. It can be large and is less important than
# image-image contrastive for speeding up the current part-search issue.
sequence_consistency_loss_weight = float(os.environ.get("SEQUENCE_CONSISTENCY_LOSS_WEIGHT", 0.0))

# Embedding variance regularization.
# Keeps image embeddings from collapsing into a narrow cone where unrelated
# windows all have high cosine similarity.
image_variance_loss_weight = float(os.environ.get("IMAGE_VARIANCE_LOSS_WEIGHT", 0.01))
image_variance_target_std = float(os.environ.get("IMAGE_VARIANCE_TARGET_STD", 0.05))

# Negatives
negative_mode = os.environ.get("NEGATIVE_MODE", "mixed").lower()
# Keep four transcript negatives by default for training quality. The speed
# optimizations should come from validation limits/local top-k/optional warmup,
# not from weakening the contrastive task to one negative.
num_negatives = int(os.environ.get("NUM_NEGATIVES", 4))
span_negative_grad_mode = os.environ.get("SPAN_NEGATIVE_GRAD_MODE", "hardest").lower()

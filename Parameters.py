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

# Negatives
negative_mode = os.environ.get("NEGATIVE_MODE", "mixed").lower()
num_negatives = int(os.environ.get("NUM_NEGATIVES", 10))
span_negative_grad_mode = os.environ.get("SPAN_NEGATIVE_GRAD_MODE", "hardest").lower()

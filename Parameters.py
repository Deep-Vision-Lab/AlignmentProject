import os

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# Training
batch_size = int(os.environ.get("BATCH_SIZE", 32))
epochs = int(os.environ.get("EPOCHS", 100))
learning_rate = float(os.environ.get("LEARNING_RATE", 1e-4))

# Data
window_size = int(os.environ.get("WINDOW_SIZE", 16))
stride_ratio = float(os.environ.get("STRIDE_RATIO", 0.5))
window_overlap_mode = os.environ.get("WINDOW_OVERLAP_MODE", "custom").lower()
vector_size = int(os.environ.get("VECTOR_SIZE", 128))
lang = os.environ.get("LANGUAGE", "Arabic")
num_samples = int(os.environ.get("NUM_SAMPLES", 10000))

# Fine-tuning
finetune_lang = "English"
finetune_num_samples = 10000
finetune_data_dir = f"DataSet/Synthetic_{finetune_lang}_{finetune_num_samples}"
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

# Negatives
negative_mode = os.environ.get("NEGATIVE_MODE", "mixed").lower()
num_negatives = int(os.environ.get("NUM_NEGATIVES", 10))

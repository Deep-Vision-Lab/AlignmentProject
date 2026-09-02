"""Single source of truth for AlignmentProject experiments."""
from __future__ import annotations

import os
import torch


def _flag(value: bool) -> str:
    return "1" if bool(value) else "0"


# 1. EXPERIMENT / RUNTIME
device = "cuda" if torch.cuda.is_available() else "cpu"
experiment_name = "vit_baseline"
dataset_type = "auto"  # auto | real | synthetic
train_seed = 42
dataset_split_seed = 42
use_amp = True
use_wandb = True
wandb_project = "alignment-vit"
log_memory_every_n_batches = 25
profile_training = False
profile_max_batches = 0

# 2. TRAINING / OPTIMIZATION
batch_size = 16  # per GPU micro-batch
gradient_accumulation_steps = 4
epochs = 20
learning_rate = 1e-4
finetune_epochs = 30
finetune_learning_rate = 2e-5
valid_every_n_epochs = 1
valid_max_batches = 20
full_checkpoint_every_n_epochs = 5
model_weights_every_n_epochs = 2
use_fused_adam = True
allow_tf32 = True
cudnn_benchmark = True
float32_matmul_precision = "high"
ddp_static_graph = True
use_channels_last = True
torch_compile_visual = False
torch_compile_mode = "reduce-overhead"

# 3. DATASET / SYNTHETIC TRAINING
num_samples = 10000
synthetic_train_samples = 9000
synthetic_augment = True
synthetic_augment_probability = 0.30
synthetic_aug_rotate_deg = 1.0
synthetic_aug_translate_x = 0.015
synthetic_aug_translate_y = 0.03
synthetic_aug_scale_min = 0.95
synthetic_aug_scale_max = 1.00
synthetic_aug_brightness = 0.10
synthetic_aug_contrast = 0.12

# 3B. REAL-DATA AUGMENTATION
real_manifest_name = "dataset_manifest.jsonl"
real_train_samples_per_epoch = 6000
real_augment = True
real_dataset_labels = "high_match,medium_match"
real_text_key = "text_original_path"
real_min_text_score = 0.0
real_split_by_pair_id = True
real_validate_paths = False
real_filter_infeasible_span_dtw = True
real_max_alignment_windows = 63
real_aug_stitch_prob = 0.0
real_aug_stitch_pool_size = 32
real_aug_stitch_max_text_chars = 120
real_aug_stitch_prefer_adjacent = True
real_aug_stitch_gap_min = 0.08
real_aug_stitch_gap_max = 0.18
real_aug_appearance_prob = 0.85
real_aug_rotate_deg = 1.25
real_aug_translate_x = 0.012
real_aug_translate_y = 0.035
real_aug_brightness = 0.12
real_aug_contrast = 0.18
real_aug_blur_prob = 0.15
real_aug_blur_radius = 0.8
real_aug_noise_prob = 0.18
real_aug_noise_std = 5.0
real_aug_morph_prob = 0.25
real_aug_speckle_prob = 0.12
real_aug_speckle_fraction = 0.0006
real_binarize = True
real_binarize_method = "otsu"
real_binarize_threshold = 180
real_binarize_autocontrast = True
real_binarize_auto_invert = True

# 4. IMAGE GEOMETRY / WINDOWING
line_height = 128
line_width = 1024
window_size = 32
stride_ratio = 0.5
window_overlap_mode = "custom"
vector_size = 128
lang = "Arabic"
target_ink_height_ratio = 0.72
ink_contrast_threshold = 0.15

# 5. VISUAL ENCODER (ViT BASELINE)
use_bilstm = False
bilstm_layers = 2
bilstm_hidden_dim = vector_size
use_local_window_grouping = False
local_group_size = 3
vit_input_height = 128
vit_layers = 1
vit_heads = 4
vit_mlp_dim = 512
vit_dropout = 0.10
vit_max_tokens = 256
vit_position_base_tokens = 63
vit_binarize_input = True
vit_binarize_contrast_threshold = 0.15

# 6. TEXT ENCODER / SPAN SEMANTICS
text_encoder_type = "arabic_span"
arabic_text_model_name = "aubmindlab/bert-base-arabertv02"
max_text_token_chars = 2
max_text_span_chars = 2
max_windows_per_span = 3
span_boundary_context_chars = 1
span_boundary_context_max_core_chars = 1
span_include_space_context = False
span_allow_character_space_surfaces = False
span_space_token = "<SPACE>"
strip_span_text_edges = True
span_use_blank_transitions = True
span_blank_penalty = 0.35
span_space_max_windows = 2
span_extra_windows_per_core = 1
span_feature_cache_size = 2048
span_feature_cache_dtype = "float16"
span_backbone_batch_size = 512
clear_span_cache_each_epoch = True

# 7. SPAN-DTW / GLOBAL IMAGE-TEXT ALIGNMENT
contrastive_soft_dtw_gamma = 0.1
contrastive_margin = 10.0
contrastive_temperature = 0.07
span_window_count_penalty = 0.05
span_dtw_backend = "jax"
span_dtw_bucket_text_lengths = True
span_dtw_text_bucket_size = 64
span_dtw_max_text_bucket = 256
span_dtw_batch_bucket_size = 32
span_dtw_batch_bucket_mode = "power2"
span_dtw_active_negatives_per_sample = 4

# 8. NEGATIVE TRANSCRIPTS
negative_mode = "mixed"
num_negatives = 10
span_negative_grad_mode = "hardest"

# 9. PRE-TRANSFORMER LOCAL HARD NEGATIVES
use_local_hard_negatives = True
local_hard_negative_weight = 0.25
local_hard_negative_margin = 0.35
local_hard_negative_top_k = 12
local_hard_negative_exclude_radius = 3
local_hard_negative_min_ink = 0.01
local_hard_negative_every_n_batches = 2
local_hard_negative_max_samples_per_batch = 8

# 10. IMAGE-IMAGE PAIR LOSS
use_image_pair_contrastive = True
image_pair_loss_weight = 0.40
image_pair_margin = 0.40
image_pair_top_k = 8
image_text_loss_on_both_lines = True
image_pair_every_n_batches = 1
image_pair_max_samples_per_batch = 8
sequence_consistency_loss_weight = 0.05
pair_composition_max_regions = 2
pair_composition_max_chars = 3
order_temperature = 0.07
order_monotonic_margin = 0.02
order_position_component_weight = 1.0
order_monotonic_component_weight = 1.0

# 11. ANTI-COLLAPSE / REGULARIZATION
# Applied to L2-normalized local and contextual window directions.
image_variance_loss_weight = 0.10
image_variance_target_std = 0.05

# 12. DOMAIN / ZERO-SHOT PREPROCESSING
zero_shot_profile = False
zero_shot_preprocess = True
zero_shot_preserve_aspect = True
zero_shot_foreground_crop = True
zero_shot_source_geometry = True

# 13. JAX / HUGGINGFACE / DATALOADER RUNTIME
hf_hub_offline = True
transformers_offline = True
tokenizers_parallelism = False
jax_compilation_cache_dir = ".jax_cache/span_dtw"
jax_persistent_cache_min_compile_time_secs = 0
jax_persistent_cache_min_entry_size_bytes = -1
xla_python_client_preallocate = False
dist_timeout_seconds = 7200
dataloader_mp_context = "spawn"

# 14. EVALUATION DEFAULTS
evaluation_feature = "contextual"
evaluation_score_mode = "auto"
evaluation_score_clip = 4.0
evaluation_threshold = 0.0
evaluation_gap = -0.30
evaluation_n_samples = 100
evaluation_real_split = "test"

# Backward-compatible names retained for helper modules.
finetune_lang = "Arabic"
finetune_num_samples = num_samples
finetune_data_dir = ""


def export_environment() -> None:
    values = {
        "BATCH_SIZE": batch_size,
        "EPOCHS": epochs,
        "LEARNING_RATE": learning_rate,
        "VALID_EVERY_N_EPOCHS": valid_every_n_epochs,
        "VALID_MAX_BATCHES": valid_max_batches,
        "LOG_MEMORY_EVERY_N_BATCHES": log_memory_every_n_batches,
        "GRADIENT_ACCUMULATION_STEPS": gradient_accumulation_steps,
        "TRAIN_SEED": train_seed,
        "DATASET_SPLIT_SEED": dataset_split_seed,
        "USE_AMP": _flag(use_amp),
        "USE_WANDB": _flag(use_wandb),
        "WANDB_PROJECT": wandb_project,
        "PROFILE_TRAINING": _flag(profile_training),
        "PROFILE_MAX_BATCHES": profile_max_batches,
        "FULL_CHECKPOINT_EVERY_N_EPOCHS": full_checkpoint_every_n_epochs,
        "MODEL_WEIGHTS_EVERY_N_EPOCHS": model_weights_every_n_epochs,
        "USE_FUSED_ADAM": _flag(use_fused_adam),
        "ALLOW_TF32": _flag(allow_tf32),
        "CUDNN_BENCHMARK": _flag(cudnn_benchmark),
        "FLOAT32_MATMUL_PRECISION": float32_matmul_precision,
        "DDP_STATIC_GRAPH": _flag(ddp_static_graph),
        "USE_CHANNELS_LAST": _flag(use_channels_last),
        "TORCH_COMPILE_VISUAL": _flag(torch_compile_visual),
        "TORCH_COMPILE_MODE": torch_compile_mode,
        "NUM_SAMPLES": num_samples,
        "SYNTHETIC_TRAIN_SAMPLES": synthetic_train_samples,
        "SYNTHETIC_AUGMENT": _flag(synthetic_augment),
        "SYNTHETIC_AUGMENT_PROBABILITY": synthetic_augment_probability,
        "SYNTHETIC_AUG_ROTATE_DEG": synthetic_aug_rotate_deg,
        "SYNTHETIC_AUG_TRANSLATE_X": synthetic_aug_translate_x,
        "SYNTHETIC_AUG_TRANSLATE_Y": synthetic_aug_translate_y,
        "SYNTHETIC_AUG_SCALE_MIN": synthetic_aug_scale_min,
        "SYNTHETIC_AUG_SCALE_MAX": synthetic_aug_scale_max,
        "SYNTHETIC_AUG_BRIGHTNESS": synthetic_aug_brightness,
        "SYNTHETIC_AUG_CONTRAST": synthetic_aug_contrast,
        "REAL_MANIFEST_NAME": real_manifest_name,
        "REAL_TRAIN_SAMPLES_PER_EPOCH": real_train_samples_per_epoch,
        "REAL_AUGMENT": _flag(real_augment),
        "REAL_DATASET_LABELS": real_dataset_labels,
        "REAL_TEXT_KEY": real_text_key,
        "REAL_MIN_TEXT_SCORE": real_min_text_score,
        "REAL_SPLIT_BY_PAIR_ID": _flag(real_split_by_pair_id),
        "REAL_VALIDATE_PATHS": _flag(real_validate_paths),
        "REAL_FILTER_INFEASIBLE_SPAN_DTW": _flag(real_filter_infeasible_span_dtw),
        "REAL_MAX_ALIGNMENT_WINDOWS": real_max_alignment_windows,
        "REAL_AUG_STITCH_PROB": real_aug_stitch_prob,
        "REAL_AUG_STITCH_POOL_SIZE": real_aug_stitch_pool_size,
        "REAL_AUG_STITCH_MAX_TEXT_CHARS": real_aug_stitch_max_text_chars,
        "REAL_AUG_STITCH_PREFER_ADJACENT": _flag(real_aug_stitch_prefer_adjacent),
        "REAL_AUG_STITCH_GAP_MIN": real_aug_stitch_gap_min,
        "REAL_AUG_STITCH_GAP_MAX": real_aug_stitch_gap_max,
        "REAL_AUG_APPEARANCE_PROB": real_aug_appearance_prob,
        "REAL_AUG_ROTATE_DEG": real_aug_rotate_deg,
        "REAL_AUG_TRANSLATE_X": real_aug_translate_x,
        "REAL_AUG_TRANSLATE_Y": real_aug_translate_y,
        "REAL_AUG_BRIGHTNESS": real_aug_brightness,
        "REAL_AUG_CONTRAST": real_aug_contrast,
        "REAL_AUG_BLUR_PROB": real_aug_blur_prob,
        "REAL_AUG_BLUR_RADIUS": real_aug_blur_radius,
        "REAL_AUG_NOISE_PROB": real_aug_noise_prob,
        "REAL_AUG_NOISE_STD": real_aug_noise_std,
        "REAL_AUG_MORPH_PROB": real_aug_morph_prob,
        "REAL_AUG_SPECKLE_PROB": real_aug_speckle_prob,
        "REAL_AUG_SPECKLE_FRACTION": real_aug_speckle_fraction,
        "REAL_BINARIZE": _flag(real_binarize),
        "REAL_BINARIZE_METHOD": real_binarize_method,
        "REAL_BINARIZE_THRESHOLD": real_binarize_threshold,
        "REAL_BINARIZE_AUTOCONTRAST": _flag(real_binarize_autocontrast),
        "REAL_BINARIZE_AUTO_INVERT": _flag(real_binarize_auto_invert),
        "LINE_HEIGHT": line_height,
        "LINE_WIDTH": line_width,
        "WINDOW_SIZE": window_size,
        "STRIDE_RATIO": stride_ratio,
        "WINDOW_OVERLAP_MODE": window_overlap_mode,
        "VECTOR_SIZE": vector_size,
        "LANGUAGE": lang,
        "TARGET_INK_HEIGHT_RATIO": target_ink_height_ratio,
        "INK_CONTRAST_THRESHOLD": ink_contrast_threshold,
        "USE_BILSTM": _flag(use_bilstm),
        "BILSTM_LAYERS": bilstm_layers,
        "USE_LOCAL_WINDOW_GROUPING": _flag(use_local_window_grouping),
        "LOCAL_GROUP_SIZE": local_group_size,
        "VIT_INPUT_HEIGHT": vit_input_height,
        "VIT_LAYERS": vit_layers,
        "VIT_HEADS": vit_heads,
        "VIT_MLP_DIM": vit_mlp_dim,
        "VIT_DROPOUT": vit_dropout,
        "VIT_MAX_TOKENS": vit_max_tokens,
        "VIT_POSITION_BASE_TOKENS": vit_position_base_tokens,
        "VIT_BINARIZE_INPUT": _flag(vit_binarize_input),
        "VIT_BINARIZE_CONTRAST_THRESHOLD": vit_binarize_contrast_threshold,
        "TEXT_ENCODER_TYPE": text_encoder_type,
        "ARABIC_TEXT_MODEL_NAME": arabic_text_model_name,
        "MAX_TEXT_TOKEN_CHARS": max_text_token_chars,
        "MAX_TEXT_SPAN_CHARS": max_text_span_chars,
        "MAX_WINDOWS_PER_SPAN": max_windows_per_span,
        "SPAN_BOUNDARY_CONTEXT_CHARS": span_boundary_context_chars,
        "SPAN_BOUNDARY_CONTEXT_MAX_CORE_CHARS": span_boundary_context_max_core_chars,
        "SPAN_INCLUDE_SPACE_CONTEXT": _flag(span_include_space_context),
        "SPAN_ALLOW_CHARACTER_SPACE_SURFACES": _flag(span_allow_character_space_surfaces),
        "SPAN_SPACE_TOKEN": span_space_token,
        "STRIP_SPAN_TEXT_EDGES": _flag(strip_span_text_edges),
        "SPAN_USE_BLANK_TRANSITIONS": _flag(span_use_blank_transitions),
        "SPAN_BLANK_PENALTY": span_blank_penalty,
        "SPAN_SPACE_MAX_WINDOWS": span_space_max_windows,
        "SPAN_EXTRA_WINDOWS_PER_CORE": span_extra_windows_per_core,
        "SPAN_FEATURE_CACHE_SIZE": span_feature_cache_size,
        "SPAN_FEATURE_CACHE_DTYPE": span_feature_cache_dtype,
        "SPAN_BACKBONE_BATCH_SIZE": span_backbone_batch_size,
        "CLEAR_SPAN_CACHE_EACH_EPOCH": _flag(clear_span_cache_each_epoch),
        "CONTRASTIVE_SOFT_DTW_GAMMA": contrastive_soft_dtw_gamma,
        "CONTRASTIVE_MARGIN": contrastive_margin,
        "CONTRASTIVE_TEMPERATURE": contrastive_temperature,
        "SPAN_WINDOW_COUNT_PENALTY": span_window_count_penalty,
        "SPAN_DTW_BACKEND": span_dtw_backend,
        "SPAN_DTW_BUCKET_TEXT_LENGTHS": _flag(span_dtw_bucket_text_lengths),
        "SPAN_DTW_TEXT_BUCKET_SIZE": span_dtw_text_bucket_size,
        "SPAN_DTW_MAX_TEXT_BUCKET": span_dtw_max_text_bucket,
        "SPAN_DTW_BATCH_BUCKET_SIZE": span_dtw_batch_bucket_size,
        "SPAN_DTW_BATCH_BUCKET_MODE": span_dtw_batch_bucket_mode,
        "SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE": span_dtw_active_negatives_per_sample,
        "NEGATIVE_MODE": negative_mode,
        "NUM_NEGATIVES": num_negatives,
        "SPAN_NEGATIVE_GRAD_MODE": span_negative_grad_mode,
        "USE_LOCAL_HARD_NEGATIVES": _flag(use_local_hard_negatives),
        "LOCAL_HARD_NEGATIVE_WEIGHT": local_hard_negative_weight,
        "LOCAL_HARD_NEGATIVE_MARGIN": local_hard_negative_margin,
        "LOCAL_HARD_NEGATIVE_TOP_K": local_hard_negative_top_k,
        "LOCAL_HARD_NEGATIVE_EXCLUDE_RADIUS": local_hard_negative_exclude_radius,
        "LOCAL_HARD_NEGATIVE_MIN_INK": local_hard_negative_min_ink,
        "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES": local_hard_negative_every_n_batches,
        "LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH": local_hard_negative_max_samples_per_batch,
        "USE_IMAGE_PAIR_CONTRASTIVE": _flag(use_image_pair_contrastive),
        "IMAGE_PAIR_LOSS_WEIGHT": image_pair_loss_weight,
        "IMAGE_PAIR_MARGIN": image_pair_margin,
        "IMAGE_PAIR_TOP_K": image_pair_top_k,
        "IMAGE_TEXT_LOSS_ON_BOTH_LINES": _flag(image_text_loss_on_both_lines),
        "IMAGE_PAIR_EVERY_N_BATCHES": image_pair_every_n_batches,
        "IMAGE_PAIR_MAX_SAMPLES_PER_BATCH": image_pair_max_samples_per_batch,
        "SEQUENCE_CONSISTENCY_LOSS_WEIGHT": sequence_consistency_loss_weight,
        "PAIR_COMPOSITION_MAX_REGIONS": pair_composition_max_regions,
        "PAIR_COMPOSITION_MAX_CHARS": pair_composition_max_chars,
        "ORDER_TEMPERATURE": order_temperature,
        "ORDER_MONOTONIC_MARGIN": order_monotonic_margin,
        "ORDER_POSITION_COMPONENT_WEIGHT": order_position_component_weight,
        "ORDER_MONOTONIC_COMPONENT_WEIGHT": order_monotonic_component_weight,
        "IMAGE_VARIANCE_LOSS_WEIGHT": image_variance_loss_weight,
        "IMAGE_VARIANCE_TARGET_STD": image_variance_target_std,
        "ZERO_SHOT_PROFILE": _flag(zero_shot_profile),
        "ZERO_SHOT_PREPROCESS": _flag(zero_shot_preprocess),
        "ZERO_SHOT_PRESERVE_ASPECT": _flag(zero_shot_preserve_aspect),
        "ZERO_SHOT_FOREGROUND_CROP": _flag(zero_shot_foreground_crop),
        "ZERO_SHOT_SOURCE_GEOMETRY": _flag(zero_shot_source_geometry),
        "HF_HUB_OFFLINE": _flag(hf_hub_offline),
        "TRANSFORMERS_OFFLINE": _flag(transformers_offline),
        "TOKENIZERS_PARALLELISM": str(tokenizers_parallelism).lower(),
        "JAX_COMPILATION_CACHE_DIR": jax_compilation_cache_dir,
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": jax_persistent_cache_min_compile_time_secs,
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": jax_persistent_cache_min_entry_size_bytes,
        "XLA_PYTHON_CLIENT_PREALLOCATE": str(xla_python_client_preallocate).lower(),
        "DIST_TIMEOUT_SECONDS": dist_timeout_seconds,
        "DATALOADER_MP_CONTEXT": dataloader_mp_context,
    }
    for key, value in values.items():
        os.environ[key] = str(value)

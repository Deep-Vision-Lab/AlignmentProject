import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from saveDATA import *
from Parameters import *
from Evaluation import *
from DiffNWAlgo import *
from wandb_config import *
from newDataLoader import *
from newDataLoader import build_dataloaders
from pathExtractor import *
from Visualization import *
from textEmbedding import *
from embeddingModel import *
from embeddingModel import img_embed_timer
from NormalizeFuncs import *
from LossFunctionWithHelpers import *
from ctc_utils import (
    CTCVocabulary,
    compute_ctc_loss,
    compute_contrastive_ctc_loss,
)
from alignment_pooling import (
    CharacterBank,
    build_char_bank,
    collect_unique_chars,
    compute_char_pool_contrastive_loss,
    groups_from_assignment,
    get_char_pool_weight,
    hard_d3tw_path_from_similarity,
    path_to_assignment,
    pool_visual_by_assignment,
)
from token_bank import (
    BigramFusionMLP,
    TokenBank,
    build_adjacent_pair_visuals,
    build_token_bank,
    collect_bigram_tokens,
    compute_bigram_token_contrastive_loss,
    get_aux_weight,
    save_token_bank_json,
)
from ngram_tokenizer import (
    NGramTokenizer,
    collect_ngram_tokens,
    save_ngram_vocab_json,
)
from token_embedding_bank import (
    build_token_embedding_bank,
    compute_char_aux_loss_from_token_pool,
    compute_token_pool_contrastive_loss,
    encode_text_units,
)
# SimilarityTransformer removed - using direct cosine similarity between CNN+BiLSTM and text embeddings

import os
import gc
import time
import wandb
import warnings
import argparse
import json


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def compute_stride(window_size, stride_ratio, window_overlap_mode):
    """Resolve the sliding-window stride for a named overlap experiment."""
    if window_overlap_mode == "no_overlap":
        return window_size
    if window_overlap_mode == "light_overlap":
        return max(1, window_size // 2)
    if window_overlap_mode == "dense_overlap":
        return max(1, window_size // 4)
    if window_overlap_mode == "custom":
        return max(1, int(window_size * stride_ratio))
    raise ValueError(
        "window_overlap_mode must be one of no_overlap, light_overlap, "
        f"dense_overlap, custom; got {window_overlap_mode!r}"
    )

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train the alignment model')
parser.add_argument('--job_id', type=str, required=True, help='Job ID for saving results')
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to dataset root directory (overrides Parameters.py defaults). '
                         'Must contain images/ and texts/ subdirectories with the standard naming.')
parser.add_argument('--pretrained_weights', type=str, default=None,
                    help='Path to a .pth weights file to load before training (for finetuning). '
                         'Loads model weights only; optimizer/epoch state is fresh.')
parser.add_argument('--finetune', action='store_true',
                    help='Switch to the finetune dataset configured in Parameters.py '
                         '(finetune_data_dir, finetune_learning_rate, finetune_epochs). '
                         '--data_dir, if also set, takes precedence over finetune_data_dir.')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to a checkpoint .pth (saved by this script) to resume from. '
                         'Restores model, optimizer, scheduler, scaler, and epoch counter.')
# Architecture overrides (supplement Parameters.py without editing it)
parser.add_argument('--window_size',    type=int,   default=None,
                    help='Override Parameters.window_size (e.g. 32 for larger windows).')
parser.add_argument('--stride_ratio',   type=float, default=None,
                    help='Override Parameters.stride_ratio (e.g. 0.25 for 75%% overlap).')
parser.add_argument('--text_embedder', '--text_embedder_type', dest='text_embedder',
                    type=str,   default=None,
                    help='Override Parameters.text_embedder_type '
                         '(char|fasttext|orthogonal_char|random_frozen_char|shape_group_char).')
parser.add_argument('--negative_mode',  type=str,   default=None,
                    help='Override Parameters.negative_mode '
                         '(mixed|length_controlled|dot_confusion|same_length_random|shuffle_only).')
parser.add_argument('--window_overlap_mode', type=str, default=None,
                    choices=['no_overlap', 'light_overlap', 'dense_overlap', 'custom'])
parser.add_argument('--multi_scale',    action='store_true', default=False,
                    help='Enable multi-scale alignment (overrides Parameters.multi_scale_enabled).')
parser.add_argument('--loss_type',      type=str,   default=None,
                    help='Override Parameters.contrastive_loss_type (margin|infonce|hybrid).')
parser.add_argument('--epochs', type=int, default=None,
                    help='Override Parameters.epochs for this run.')
parser.add_argument('--learning_rate', type=float, default=None,
                    help='Override Parameters.learning_rate for this run.')
parser.add_argument('--num_negatives', type=int, default=None,
                    help='Override Parameters.num_negatives for this run.')
parser.add_argument('--alignment_loss_type', type=str, default=None,
                    choices=['d3tw', 'ctc', 'contrastive_ctc', 'ctc_d3tw',
                             'contrastive_ctc_d3tw', 'd3tw_char_pool',
                             'contrastive_d3tw_char_pool'],
                    help='Alignment objective to train.')
parser.add_argument('--use_d3tw_char_pooling', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--char_pool_weight', type=float, default=None)
parser.add_argument('--char_pool_tau', type=float, default=None)
parser.add_argument('--char_pool_warmup_epochs', type=int, default=None)
parser.add_argument('--char_pool_ramp_epochs', type=int, default=None)
parser.add_argument('--char_pool_method', choices=['hard_mean', 'soft_weighted'], default=None)
parser.add_argument('--char_pool_detach_alignment', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--char_pool_skip_spaces', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--char_pool_min_windows_per_char', type=int, default=None)
parser.add_argument('--text_unit_type', choices=['char', 'ngram'], default=None)
parser.add_argument('--ngram_min_n', type=int, default=None)
parser.add_argument('--ngram_max_n', type=int, default=None)
parser.add_argument('--ngram_min_freq', type=int, default=None)
parser.add_argument('--ngram_max_vocab_size', type=int, default=None)
parser.add_argument('--ngram_tokenizer_mode', choices=['greedy_longest'], default=None)
parser.add_argument('--ngram_skip_spaces', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--ngram_include_ligatures', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--token_pool_weight', type=float, default=None)
parser.add_argument('--token_pool_tau', type=float, default=None)
parser.add_argument('--token_pool_warmup_epochs', type=int, default=None)
parser.add_argument('--token_pool_ramp_epochs', type=int, default=None)
parser.add_argument('--token_pool_detach_alignment', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--token_pool_min_windows_per_token', type=int, default=None)
parser.add_argument('--use_char_aux_loss', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--char_aux_weight', type=float, default=None)
parser.add_argument('--char_aux_tau', type=float, default=None)
parser.add_argument('--char_aux_warmup_epochs', type=int, default=None)
parser.add_argument('--char_aux_ramp_epochs', type=int, default=None)
parser.add_argument('--use_bigram_token_loss', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--bigram_token_weight', type=float, default=None)
parser.add_argument('--bigram_token_tau', type=float, default=None)
parser.add_argument('--bigram_token_warmup_epochs', type=int, default=None)
parser.add_argument('--bigram_token_ramp_epochs', type=int, default=None)
parser.add_argument('--bigram_token_skip_spaces', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--bigram_token_min_freq', type=int, default=None)
parser.add_argument('--bigram_token_max_vocab_size', type=int, default=None)
parser.add_argument('--bigram_token_fusion', choices=['mean', 'mlp'], default=None)
parser.add_argument('--bigram_token_include_ligatures', type=_str2bool, nargs='?', const=True,
                    default=None)
parser.add_argument('--ctc_weight', type=float, default=None,
                    help='Weight for CTC loss in hybrid modes.')
parser.add_argument('--d3tw_weight', type=float, default=None,
                    help='Weight for D3TW loss in hybrid modes.')
parser.add_argument('--contrastive_ctc_loss_type', type=str, default=None,
                    choices=['infonce', 'margin'],
                    help='Contrastive CTC ranking objective.')
parser.add_argument('--contrastive_ctc_tau', type=float, default=None,
                    help='Temperature for InfoNCE over CTC costs.')
parser.add_argument('--contrastive_ctc_margin', type=float, default=None,
                    help='Margin for contrastive CTC loss.')
parser.add_argument('--sequence_encoder_type', type=str, default=None,
                    choices=['bilstm', 'transformer', 'none'],
                    help='Sequence encoder after CNN features.')
parser.add_argument('--transformer_num_layers', type=int, default=None)
parser.add_argument('--transformer_num_heads', type=int, default=None)
parser.add_argument('--transformer_ff_dim', type=int, default=None)
parser.add_argument('--transformer_dropout', type=float, default=None)
parser.add_argument('--transformer_activation', type=str, default=None,
                    choices=['relu', 'gelu'])
parser.add_argument('--transformer_norm_first', action=argparse.BooleanOptionalAction,
                    default=None)
parser.add_argument('--transformer_positional_encoding', type=str, default=None,
                    choices=['sinusoidal', 'learnable'])
parser.add_argument('--transformer_max_len', type=int, default=None)
parser.add_argument('--return_attention_weights', action=argparse.BooleanOptionalAction,
                    default=None)
parser.add_argument('--use_cross_attention', action='store_true', default=None,
                    help='Enable optional auxiliary cross-attention similarity.')
parser.add_argument('--cross_attention_type', type=str, default=None,
                    choices=['text_to_image', 'image_to_text', 'bidirectional'])
parser.add_argument('--cross_attention_num_heads', type=int, default=None)
parser.add_argument('--cross_attention_dropout', type=float, default=None)
parser.add_argument('--cross_attention_weight', type=float, default=None)
job_id = None

from wandb_config import (
    init_wandb,
    update_wandb,
    log_wandb_weights,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Speed switches
# ============================================================================
# Profiling forces a cuda.synchronize() per phase, which kills throughput.
# Toggle with PROFILE_TIMING=1 env var.
ENABLE_PROFILING = os.environ.get('PROFILE_TIMING', '0') == '1'

# Automatic Mixed Precision: fp16 forward + scaled fp32 backward.
# Halves activation memory for CNN/BiLSTM and speeds up matmuls on Ampere+.
USE_AMP = torch.cuda.is_available() and os.environ.get('USE_AMP', '1') == '1'
_amp_dtype_env = os.environ.get('AMP_DTYPE', 'fp16').lower()
AMP_DTYPE = torch.bfloat16 if _amp_dtype_env in ('bf16', 'bfloat16') else torch.float16

# cuDNN benchmark picks the fastest conv algo for fixed input shapes (we have one).
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # TF32 on Ampere+ for free matmul speedup with negligible accuracy hit.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# GPU TIMING PROFILER - Measures execution time of each phase
# ============================================================================
class GPUTimer:
    """GPU-aware timer using CUDA events. Disabled by default (set ENABLE_PROFILING=True)."""
    def __init__(self, enabled=True, device='cuda'):
        self.enabled = enabled
        self.device = device
        self.use_cuda = torch.cuda.is_available() and 'cuda' in device
        self.timings = {}
        self.starts = {}

    def start(self, name):
        if not self.enabled:
            return
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.starts[name] = start_event
        else:
            self.starts[name] = time.time()

    def stop(self, name):
        if not self.enabled or name not in self.starts:
            return
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            duration_ms = self.starts[name].elapsed_time(end_event)
        else:
            duration_ms = (time.time() - self.starts[name]) * 1000

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_ms)
        del self.starts[name]

    def reset(self):
        self.timings = {}
        self.starts = {}

    def print_summary(self, title="Timing Summary"):
        if not self.enabled or not self.timings:
            return
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"{'Phase':<30} {'Avg (ms)':<12} {'Total (ms)':<12} {'Count':<8}")
        print(f"{'-'*60}")
        total_time = 0
        for name, durations in self.timings.items():
            avg = sum(durations) / len(durations)
            total = sum(durations)
            total_time += total
            print(f"{name:<30} {avg:<12.2f} {total:<12.2f} {len(durations):<8}")
        print(f"{'-'*60}")
        print(f"{'TOTAL':<30} {'':<12} {total_time:<12.2f}")
        print(f"{'='*60}\n")
        return self.timings


global_timer = GPUTimer(enabled=ENABLE_PROFILING)


def _weights_dir(job_id):
    path = os.path.join(os.path.dirname(__file__), "Weights", job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _make_model_config(ws, stride, sr, ctc_vocab=None):
    """Collect current training config into a dict for the checkpoint."""
    import Parameters as _P
    cfg = {
        'alignment_loss_type': _P.alignment_loss_type,
        'window_size':        ws,
        'stride':             stride,
        'stride_ratio':       sr,
        'window_overlap_mode': _P.window_overlap_mode,
        'vector_size':        _P.vector_size,
        'sequence_encoder_type': _P.sequence_encoder_type,
        'use_bilstm':         _P.use_bilstm,
        'bilstm_layers':      _P.bilstm_layers,
        'bilstm_hidden_dim':  _P.bilstm_hidden_dim,
        'transformer_num_layers': _P.transformer_num_layers,
        'transformer_num_heads': _P.transformer_num_heads,
        'transformer_ff_dim': _P.transformer_ff_dim,
        'transformer_dropout': _P.transformer_dropout,
        'transformer_activation': _P.transformer_activation,
        'transformer_norm_first': _P.transformer_norm_first,
        'transformer_positional_encoding': _P.transformer_positional_encoding,
        'transformer_max_len': _P.transformer_max_len,
        'return_attention_weights': _P.return_attention_weights,
        'use_cross_attention': _P.use_cross_attention,
        'cross_attention_type': _P.cross_attention_type,
        'cross_attention_num_heads': _P.cross_attention_num_heads,
        'cross_attention_dropout': _P.cross_attention_dropout,
        'cross_attention_weight': _P.cross_attention_weight,
        'lang':               _P.lang,
        'text_embedder_type': _P.text_embedder_type,
        'negative_mode':      getattr(_P, 'negative_mode', 'mixed'),
        'multi_scale_enabled': _P.multi_scale_enabled,
        'char_pool_weight': _P.char_pool_weight,
        'char_pool_tau': _P.char_pool_tau,
        'char_pool_method': _P.char_pool_method,
        'char_pool_warmup_epochs': _P.char_pool_warmup_epochs,
        'char_pool_ramp_epochs': _P.char_pool_ramp_epochs,
        'char_pool_detach_alignment': _P.char_pool_detach_alignment,
        'char_pool_skip_spaces': _P.char_pool_skip_spaces,
        'char_pool_min_windows_per_char': _P.char_pool_min_windows_per_char,
        'char_pool_use_char_bank': _P.char_pool_use_char_bank,
        'use_d3tw_char_pooling': _P.use_d3tw_char_pooling,
        'text_unit_type': _P.text_unit_type,
        'ngram_min_n': _P.ngram_min_n,
        'ngram_max_n': _P.ngram_max_n,
        'ngram_min_freq': _P.ngram_min_freq,
        'ngram_max_vocab_size': _P.ngram_max_vocab_size,
        'ngram_tokenizer_mode': _P.ngram_tokenizer_mode,
        'ngram_skip_spaces': _P.ngram_skip_spaces,
        'ngram_include_ligatures': _P.ngram_include_ligatures,
        'ngram_ligatures': _P.ngram_ligatures,
        'token_pool_weight': _P.token_pool_weight,
        'token_pool_tau': _P.token_pool_tau,
        'token_pool_warmup_epochs': _P.token_pool_warmup_epochs,
        'token_pool_ramp_epochs': _P.token_pool_ramp_epochs,
        'token_pool_detach_alignment': _P.token_pool_detach_alignment,
        'token_pool_min_windows_per_token': _P.token_pool_min_windows_per_token,
        'use_char_aux_loss': _P.use_char_aux_loss,
        'char_aux_weight': _P.char_aux_weight,
        'char_aux_tau': _P.char_aux_tau,
        'char_aux_warmup_epochs': _P.char_aux_warmup_epochs,
        'char_aux_ramp_epochs': _P.char_aux_ramp_epochs,
        'ngram_vocab_size': int(getattr(_make_model_config, "_ngram_vocab_size", 0))
                            if _P.text_unit_type == "ngram" else None,
        'use_bigram_token_loss': _P.use_bigram_token_loss,
        'bigram_token_weight': _P.bigram_token_weight,
        'bigram_token_tau': _P.bigram_token_tau,
        'bigram_token_warmup_epochs': _P.bigram_token_warmup_epochs,
        'bigram_token_ramp_epochs': _P.bigram_token_ramp_epochs,
        'bigram_token_skip_spaces': _P.bigram_token_skip_spaces,
        'bigram_token_min_freq': _P.bigram_token_min_freq,
        'bigram_token_max_vocab_size': _P.bigram_token_max_vocab_size,
        'bigram_token_fusion': _P.bigram_token_fusion,
        'bigram_token_include_ligatures': _P.bigram_token_include_ligatures,
        'bigram_token_vocab_size': int(getattr(_make_model_config, "_bigram_token_vocab_size", 0)),
    }
    if ctc_vocab is not None:
        cfg.update({
            'ctc_vocab_size': len(ctc_vocab),
            'ctc_blank_idx': ctc_vocab.blank_idx,
        })
    return cfg


def save_model_weights(model, epoch, job_id, model_config=None,
                       ctc_head=None, ctc_vocab=None,
                       cross_attention_module=None,
                       bigram_fusion_mlp=None):
    weights_path = os.path.join(_weights_dir(job_id), "model_latest.pth")
    payload = {
        'model_state_dict': model.state_dict(),
        'image_model_state_dict': model.state_dict(),
    }
    if ctc_head is not None:
        payload['ctc_head_state_dict'] = ctc_head.state_dict()
    if cross_attention_module is not None:
        payload['cross_attention_state_dict'] = cross_attention_module.state_dict()
    if bigram_fusion_mlp is not None:
        payload['bigram_fusion_mlp_state_dict'] = bigram_fusion_mlp.state_dict()
    if ctc_vocab is not None:
        payload['ctc_vocab'] = ctc_vocab.to_dict()
    if model_config:
        payload['model_config'] = model_config
    if ctc_head is not None or ctc_vocab is not None or model_config:
        torch.save(payload, weights_path)
    else:
        torch.save(model.state_dict(), weights_path)
    return weights_path


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, job_id,
                    model_config=None, ctc_head=None, ctc_vocab=None,
                    cross_attention_module=None,
                    bigram_fusion_mlp=None):
    """Save full training state so --resume can pick up exactly where we left off."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'image_model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
    }
    if ctc_head is not None:
        ckpt['ctc_head_state_dict'] = ctc_head.state_dict()
    if cross_attention_module is not None:
        ckpt['cross_attention_state_dict'] = cross_attention_module.state_dict()
    if bigram_fusion_mlp is not None:
        ckpt['bigram_fusion_mlp_state_dict'] = bigram_fusion_mlp.state_dict()
    if ctc_vocab is not None:
        ckpt['ctc_vocab'] = ctc_vocab.to_dict()
    if model_config:
        ckpt['model_config'] = model_config
    torch.save(ckpt, os.path.join(_weights_dir(job_id), "checkpoint_latest.pth"))


def _extract_model_state(loaded):
    """Accept either a raw state_dict or a checkpoint dict wrapping one."""
    if isinstance(loaded, dict) and 'image_model_state_dict' in loaded:
        return loaded['image_model_state_dict']
    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
        return loaded['model_state_dict']
    return loaded


def _needs_ctc():
    return alignment_loss_type in {'ctc', 'contrastive_ctc', 'ctc_d3tw',
                                   'contrastive_ctc_d3tw'}


def _needs_d3tw():
    return alignment_loss_type in {'d3tw', 'ctc_d3tw',
                                   'contrastive_ctc_d3tw', 'd3tw_char_pool',
                                   'contrastive_d3tw_char_pool'}


def _uses_char_pool():
    return (
        alignment_loss_type in {'d3tw_char_pool', 'contrastive_d3tw_char_pool'}
        or use_d3tw_char_pooling
    )


def _uses_bigram_token_loss():
    return bool(use_bigram_token_loss)


def _uses_ngram_units():
    return str(text_unit_type).lower() == "ngram"


def _collect_training_texts(train_loader, data_dir=None):
    texts = []
    ds = getattr(train_loader, 'dataset', None)
    base_ds = getattr(ds, 'dataset', ds)
    indices = getattr(ds, 'indices', None)
    text_dir = None
    if getattr(base_ds, 'new_dataset', None):
        text_dir = base_ds.new_dataset.get('texts')
    elif data_dir is not None:
        text_dir = os.path.join(data_dir, 'texts')

    if text_dir and indices is not None:
        for idx in indices:
            path = os.path.join(text_dir, f"text1_{idx + 1}.txt")
            with open(path, "r", encoding="utf-8") as f:
                texts.append(' ' + f.read().strip() + ' ')
        return texts

    if text_dir:
        for name in sorted(os.listdir(text_dir)):
            if name.startswith("text1_") and name.endswith(".txt"):
                with open(os.path.join(text_dir, name), "r", encoding="utf-8") as f:
                    texts.append(' ' + f.read().strip() + ' ')
    return texts


def _save_char_bank(char_bank, job_id):
    path = os.path.join(_weights_dir(job_id), "char_bank.json")
    payload = {
        "idx_to_char": char_bank.idx_to_char,
        "char_to_idx": char_bank.char_to_idx,
        # Keep the frozen vectors with the mapping so evaluation/figure code
        # can reproduce character logits without rebuilding the text model.
        "embeddings": char_bank.embeddings.detach().cpu().tolist(),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def _save_token_bank(token_bank, job_id):
    path = os.path.join(_weights_dir(job_id), "token_bank.json")
    save_token_bank_json(path, token_bank.token_to_idx, token_bank.idx_to_token)
    return path


def check_grad(grad):
    if grad is None:
        print("Gradient is None!")
    elif torch.isnan(grad).any():
        print("Gradient became NaN!")
    else:
        print(f"Gradient flowing... Sum: {grad.sum().item():.5f}")


def _embed_negatives(textEmbed, neg_texts):
    """Embed all negative texts in ONE forward pass.

    neg_texts: list of B lists, each containing K strings.
    Returns: [B, K, max_len, vec_size]
    """
    B = len(neg_texts)
    K = len(neg_texts[0]) if B > 0 else 0
    flat = [t for sample in neg_texts for t in sample]  # B*K strings

    # textEmbed pads inside (pad_sequence) and returns [B*K, max_len, vec_size]
    flat_emb = textEmbed(flat)
    return flat_emb.view(B, K, flat_emb.size(1), flat_emb.size(2))


def compute_varlen_similarities(textEmbed, norm_img, pos_texts, neg_texts,
                                cross_attention_module=None,
                                cross_attention_weight=0.0,
                                ngram_tokenizer=None):
    """
    Build per-sample similarity matrices without padded text rows.

    Each text is embedded individually (no batch padding), so DTW receives
    exactly true_text_len rows — no PAD tokens corrupt the alignment.

    Gradients flow from sim tensors → norm_img → image encoder.
    Text embeddings are computed under no_grad (text branch is frozen).

    Args:
        textEmbed: frozen text embedding module
        norm_img:  [B, S, D] L2-normalised image embeddings (carries gradients)
        pos_texts: list[str], length B
        neg_texts: list[list[str]], shape B × K

    Returns:
        sim_pos_list: list[B] of Tensor[T_i, S]
        sim_neg_list: list[B] of list[K] of Tensor[T_neg_ik, S]
    """
    use_cross = (
        cross_attention_module is not None and cross_attention_weight > 0
    )
    sim_pos_list = []
    sim_neg_list = []
    pos_units_list = []
    pos_spans_list = []

    for i, text in enumerate(pos_texts):
        units, spans, txt_emb = encode_text_units(
            text=text,
            text_unit_type=text_unit_type,
            text_embedder=textEmbed,
            device=norm_img.device,
            ngram_tokenizer=ngram_tokenizer,
        )
        txt_emb = normalize_func(txt_emb)
        pos_units_list.append(units)
        pos_spans_list.append(spans)

        # sim carries grad via norm_img[i]. Dot-product remains the primary
        # D3TW matrix; optional cross-attention is only an additive auxiliary.
        dot_pos = torch.einsum("td,sd->ts", txt_emb, norm_img[i])
        if use_cross:
            attn_pos = cross_attention_module(txt_emb, norm_img[i])
            sim_pos = dot_pos + cross_attention_weight * attn_pos
        else:
            sim_pos = dot_pos
        sim_pos_list.append(sim_pos)

        sample_neg_sims = []
        for neg_text in neg_texts[i]:
            _neg_units, _neg_spans, neg_emb = encode_text_units(
                text=neg_text,
                text_unit_type=text_unit_type,
                text_embedder=textEmbed,
                device=norm_img.device,
                ngram_tokenizer=ngram_tokenizer,
            )
            neg_emb = normalize_func(neg_emb)
            dot_neg = torch.einsum("td,sd->ts", neg_emb, norm_img[i])
            if use_cross:
                attn_neg = cross_attention_module(neg_emb, norm_img[i])
                sim_neg = dot_neg + cross_attention_weight * attn_neg
            else:
                sim_neg = dot_neg
            sample_neg_sims.append(sim_neg)

        sim_neg_list.append(sample_neg_sims)

    return sim_pos_list, sim_neg_list, pos_units_list, pos_spans_list


def compute_d3tw_batch_loss(imageEmbed, textEmbed,
                            images, pos_texts, neg_texts,
                            criterion,
                            cross_attention_module=None,
                            ngram_tokenizer=None,
                            epoch=0, batch_idx=0, dataloader_length=0, debug=False, timer=None):
    """
    Compute batch loss for text-image alignment with multiple in-batch negatives.

    AMP-aware: image embeddings come out in fp16/bf16 under autocast; we cast
    them back to fp32 before similarity / DTW (DTW kernels require fp32).
    """
    if timer is None:
        timer = global_timer

    # ==================== Text Embeddings (no_grad: textEmbed is frozen) ====================
    timer.start('2_text_embedding')
    with torch.no_grad():
        pos_text_emb = textEmbed(pos_texts)              # [B, text_len, vec_size]
        stacked_neg = _embed_negatives(textEmbed, neg_texts)  # [B, K, neg_len, vec_size]
        stacked_neg = normalize_func(stacked_neg)
        norm_pos_text = normalize_func(pos_text_emb)
    timer.stop('2_text_embedding')

    def _autocast():
        return torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP)

    artifacts = None
    if multi_scale_enabled:
        # Variable-length varlen is not yet implemented for multi-scale.
        # Using padded batched similarity here until forward_varlen is extended
        # to support dual-scale inputs.
        def _similarities_at_scale(ws, sr):
            s = max(1, int(ws * sr))
            with _autocast():
                img_emb_s = imageEmbed.forward_at_scale(images, ws, s,
                                                        show_dims=False, timer=timer)
            img_emb_s = img_emb_s.float()
            norm_img_s = normalize_func(img_emb_s)
            dot_pos_s = torch.einsum('bsv,btv->bst', norm_pos_text, norm_img_s)
            dot_neg_s = torch.einsum('bktv,bsv->bkts', stacked_neg, norm_img_s)
            if (
                cross_attention_module is not None
                and use_cross_attention
                and cross_attention_weight > 0
            ):
                attn_pos_s = cross_attention_module(norm_pos_text, norm_img_s)
                B, K, T, D = stacked_neg.shape
                flat_neg = stacked_neg.reshape(B * K, T, D)
                flat_img = (
                    norm_img_s.unsqueeze(1)
                    .expand(B, K, norm_img_s.size(1), norm_img_s.size(2))
                    .reshape(B * K, norm_img_s.size(1), norm_img_s.size(2))
                )
                attn_neg_s = cross_attention_module(flat_neg, flat_img).reshape(
                    B, K, T, norm_img_s.size(1)
                )
                sim_pos_s = dot_pos_s + cross_attention_weight * attn_pos_s
                sim_neg_s = dot_neg_s + cross_attention_weight * attn_neg_s
            else:
                sim_pos_s = dot_pos_s
                sim_neg_s = dot_neg_s
            return sim_pos_s, sim_neg_s

        macro_ws, micro_ws = multi_scale_window_sizes

        timer.start('4_macro_scale')
        sim_pos_macro, sim_neg_macro = _similarities_at_scale(macro_ws, stride_ratio)
        timer.stop('4_macro_scale')

        timer.start('4_micro_scale')
        sim_pos_micro, sim_neg_micro = _similarities_at_scale(micro_ws, stride_ratio)
        timer.stop('4_micro_scale')

        timer.start('5_loss_computation')
        total_loss, loss_dict = criterion(
            sim_pos_macro, sim_neg_macro,
            sim_pos_micro, sim_neg_micro,
        )
        timer.stop('5_loss_computation')

        sim_pos_for_viz = sim_pos_macro

        if batch_idx % 10 == 0:
            print(
                f"[Multi-Scale DTW]"
                f"  macro: raw pos={loss_dict['macro_cost_pos']:.1f} neg={loss_dict['macro_cost_neg']:.1f}"
                f"  pos_wins={loss_dict['macro_pos_prob']:.3f}"
                f"  |  micro: raw pos={loss_dict['micro_cost_pos']:.1f} neg={loss_dict['micro_cost_neg']:.1f}"
                f"  pos_wins={loss_dict['micro_pos_prob']:.3f}"
                f"  |  loss={total_loss.item():.4f}",
                flush=True,
            )

    else:
        timer.start('1_image_embedding')
        with _autocast():
            img_emb = imageEmbed(images, show_dims=False, timer=timer)
        img_emb = img_emb.float()
        timer.stop('1_image_embedding')

        timer.start('3b_norm_img')
        norm_img = normalize_func(img_emb)
        timer.stop('3b_norm_img')

        # Use variable-length similarities so padded text rows never enter DTW
        timer.start('4_varlen_similarities')
        sim_pos_list, sim_neg_list, pos_units_list, pos_spans_list = compute_varlen_similarities(
            textEmbed=textEmbed,
            norm_img=norm_img,
            pos_texts=pos_texts,
            neg_texts=neg_texts,
            cross_attention_module=cross_attention_module,
            cross_attention_weight=(
                cross_attention_weight
                if use_cross_attention and cross_attention_weight > 0
                else 0.0
            ),
            ngram_tokenizer=ngram_tokenizer,
        )
        timer.stop('4_varlen_similarities')

        # One-time sanity check: print shapes for the first batch of training
        if batch_idx == 0 and epoch == 0:
            expected_units = len(pos_units_list[0])
            print(f"[SANITY] sim_pos_list[0].shape = {sim_pos_list[0].shape}  "
                  f"(expected [{expected_units}, {norm_img.shape[1]}], "
                  f"text_unit_type={text_unit_type})", flush=True)
            assert sim_pos_list[0].shape[0] == expected_units, \
                f"Padded text rows detected: got {sim_pos_list[0].shape[0]} rows but text has {expected_units} text units"

        timer.start('6_loss_computation')
        total_loss, loss_dict = criterion.forward_varlen(sim_pos_list, sim_neg_list)
        timer.stop('6_loss_computation')

        # Use first sample's sim for visualization (no padding)
        sim_pos_for_viz = sim_pos_list[0].unsqueeze(0)
        artifacts = {
            "norm_img": norm_img,
            "sim_pos_list": sim_pos_list,
            "unit_lists": pos_units_list,
            "unit_spans": pos_spans_list,
        }

        if batch_idx % 10 == 0:
            print(
                f"[DTW] raw: pos={loss_dict['cost_pos']:.1f}  neg={loss_dict['cost_neg']:.1f}"
                f"  |  prob: pos_wins={loss_dict['pos_prob']:.3f} (goal→1.0)"
                f"  |  gap={loss_dict['gap']:.3f}  loss={total_loss.item():.4f}",
                flush=True,
            )

    # Debugging: Save visualizations for the first batch every 10 epochs
    if debug and batch_idx == 0 and epoch % 10 == 0:
        # For multi-scale, sim_pos_for_viz is a padded [B, T, S] tensor.
        # For single-scale, it is sim_pos_list[0].unsqueeze(0) — shape [1, T_0, S].
        # save_debug_visualizations handles both; max_samples=1 keeps it fast.
        save_debug_visualizations(
            imageEmbed,
            pos_texts[:1],
            images[:1],
            sim_pos_for_viz,
            epoch,
            job_id=job_id
        )
        # model_config injected by Train() via a closure attribute
        _cfg = getattr(compute_batch_loss, '_model_config', None)
        if not _needs_ctc():
            save_model_weights(
                imageEmbed, epoch, job_id, model_config=_cfg,
                cross_attention_module=cross_attention_module,
            )

    return total_loss, loss_dict, artifacts


def compute_char_pool_batch_loss(
    pos_texts, artifacts, char_bank, epoch, batch_idx=0,
    token_bank=None, bigram_fusion_mlp=None
):
    """Pool one D3TW group per character and classify against the frozen bank."""
    if artifacts is None:
        raise RuntimeError("D3TW character pooling is not supported with multi_scale_enabled.")
    if char_pool_method == "soft_weighted":
        raise NotImplementedError(
            "char_pool_method='soft_weighted' requires Soft-DTW alignment weights, "
            "which the current D3TW implementation does not expose. Use 'hard_mean'."
        )

    norm_img = artifacts["norm_img"]
    sim_pos_list = artifacts["sim_pos_list"]
    sample_losses = []
    total_valid = 0
    total_chars = 0
    weighted_top1 = 0.0
    weighted_top5 = 0.0
    bigram_losses = []
    bigram_total_valid = 0
    bigram_weighted_top1 = 0.0
    bigram_weighted_top5 = 0.0
    effective_bigram_weight = get_aux_weight(
        epoch,
        bigram_token_weight,
        bigram_token_warmup_epochs,
        bigram_token_ramp_epochs,
    )
    for sample_idx, transcript in enumerate(pos_texts):
        chars = list(transcript)
        sim = sim_pos_list[sample_idx]
        visual_b = norm_img[sample_idx]
        total_chars += len(chars)
        if not chars:
            continue
        assert sim.dim() == 2
        assert visual_b.dim() == 2
        assert sim.shape[1] == visual_b.shape[0]
        assert len(chars) == sim.shape[0], (
            f"Transcript chars length {len(chars)} != sim_pos T {sim.shape[0]}. "
            "Use the exact same character list as text embeddings."
        )

        path = hard_d3tw_path_from_similarity(sim)
        if not path:
            if epoch == 0 and batch_idx == 0 and sample_idx == 0:
                print(
                    f"[CHAR POOL GROUPS] Transcript: {transcript}\n"
                    f"no restricted path exists for T={sim.shape[0]}, S={sim.shape[1]} "
                    "(this topology requires T <= S)",
                    flush=True,
                )
            continue
        assignment = path_to_assignment(
            path,
            num_chars=sim.shape[0],
            num_visual=sim.shape[1],
            device=sim.device,
        )
        pooled_visual, valid_mask, counts = pool_visual_by_assignment(
            visual_emb=visual_b,
            assignment=assignment,
            detach_assignment=char_pool_detach_alignment,
            min_windows_per_char=char_pool_min_windows_per_char,
        )
        assert pooled_visual.shape[0] == sim.shape[0]
        assert pooled_visual.shape[1] == visual_b.shape[1]

        effective_weight = get_char_pool_weight(
            epoch, char_pool_weight, char_pool_warmup_epochs, char_pool_ramp_epochs
        )
        if effective_weight > 0 and torch.is_grad_enabled():
            assert pooled_visual.requires_grad, "Pooled vectors lost their image gradient path"

        if epoch == 0 and batch_idx == 0 and sample_idx == 0:
            print(f"[CHAR POOL GROUPS] Transcript: {transcript}", flush=True)
            for char_idx, visual_indices in enumerate(groups_from_assignment(assignment)):
                print(
                    f"char[{char_idx}]={chars[char_idx]!r} windows={visual_indices}",
                    flush=True,
                )
        sample_loss, sample_stats = compute_char_pool_contrastive_loss(
            pooled_visual,
            chars,
            char_bank.embeddings,
            char_bank.char_to_idx,
            tau=char_pool_tau,
            valid_mask=valid_mask,
            skip_spaces=char_pool_skip_spaces,
        )
        valid_count = sample_stats["char_pool_valid_chars"]
        sample_losses.append(sample_loss)
        if valid_count:
            total_valid += valid_count
            weighted_top1 += sample_stats["char_pool_acc_top1"] * valid_count
            weighted_top5 += sample_stats["char_pool_acc_top5"] * valid_count

        if (
            use_bigram_token_loss
            and token_bank is not None
        ):
            pair_visuals, target_token_ids, pair_metadata = build_adjacent_pair_visuals(
                pooled_visual=pooled_visual,
                transcript_chars=chars,
                token_to_idx=token_bank.token_to_idx,
                fusion=bigram_token_fusion,
                skip_spaces=bigram_token_skip_spaces,
                fusion_mlp=bigram_fusion_mlp,
            )
            if pair_visuals.numel() > 0 and torch.is_grad_enabled():
                assert pair_visuals.requires_grad, "Bigram pair vectors lost image gradient path"
            bigram_loss_i, bigram_stats_i = compute_bigram_token_contrastive_loss(
                pair_visuals=pair_visuals,
                target_token_ids=target_token_ids,
                token_bank_embeddings=token_bank.embeddings,
                tau=bigram_token_tau,
            )
            bigram_losses.append(bigram_loss_i)
            valid_pairs = bigram_stats_i["bigram_token_valid_pairs"]
            if valid_pairs:
                bigram_total_valid += valid_pairs
                bigram_weighted_top1 += bigram_stats_i["bigram_token_acc_top1"] * valid_pairs
                bigram_weighted_top5 += bigram_stats_i["bigram_token_acc_top5"] * valid_pairs
            if epoch == 0 and batch_idx == 0 and sample_idx == 0:
                print("Bigram pairs:", flush=True)
                for pair_idx, meta in enumerate(pair_metadata[:20]):
                    print(
                        f"pair[{pair_idx}]={meta['token']!r} "
                        f"from chars [{meta['start_char_index']},{meta['end_char_index']}]",
                        flush=True,
                    )

    if sample_losses:
        char_loss = torch.stack(sample_losses).mean()
        if total_valid > 0 and torch.is_grad_enabled():
            assert char_loss.requires_grad, "Character-pool loss lost its image gradient path"
    else:
        if not getattr(compute_char_pool_batch_loss, "_warned_no_valid", False):
            print(
                "[WARN] No valid D3TW character groups in batch; "
                "char-pool loss skipped.",
                flush=True,
            )
            compute_char_pool_batch_loss._warned_no_valid = True
        char_loss = norm_img.sum() * 0.0

    effective_weight = get_char_pool_weight(
        epoch, char_pool_weight, char_pool_warmup_epochs, char_pool_ramp_epochs
    )
    if bigram_losses:
        bigram_loss = torch.stack(bigram_losses).mean()
    else:
        bigram_loss = norm_img.sum() * 0.0
    stats = {
        "char_pool_loss": float(char_loss.detach().item()),
        "char_pool_acc_top1": weighted_top1 / max(total_valid, 1),
        "char_pool_acc_top5": weighted_top5 / max(total_valid, 1),
        "char_pool_num_chars": total_chars,
        "char_pool_valid_chars": total_valid,
        "char_pool_tau": float(char_pool_tau),
        "effective_char_pool_weight": effective_weight,
        "bigram_token_loss": float(bigram_loss.detach().item()),
        "effective_bigram_token_weight": effective_bigram_weight,
        "bigram_token_acc_top1": bigram_weighted_top1 / max(bigram_total_valid, 1),
        "bigram_token_acc_top5": bigram_weighted_top5 / max(bigram_total_valid, 1),
        "bigram_token_valid_pairs": bigram_total_valid,
        "bigram_token_tau": float(bigram_token_tau),
        "sequence_length": int(norm_img.shape[1]),
    }
    return char_loss, bigram_loss, stats


def compute_token_pool_batch_loss(
    pos_texts, artifacts, token_bank, epoch, batch_idx=0, char_bank=None
):
    """Pool one D3TW group per n-gram token and classify against token bank."""
    if artifacts is None:
        raise RuntimeError("D3TW token pooling is not supported with multi_scale_enabled.")
    if token_bank is None:
        raise RuntimeError("text_unit_type='ngram' requires a frozen token bank.")

    norm_img = artifacts["norm_img"]
    sim_pos_list = artifacts["sim_pos_list"]
    unit_lists = artifacts["unit_lists"]
    unit_spans = artifacts["unit_spans"]

    token_losses = []
    char_aux_losses = []
    total_valid = 0
    total_units = 0
    weighted_top1 = 0.0
    weighted_top5 = 0.0
    window_count_weight = 0
    sum_mean_windows = 0.0
    min_windows_global = None
    max_windows_global = None
    char_aux_total_valid = 0
    char_aux_weighted_top1 = 0.0
    char_aux_weighted_top5 = 0.0

    effective_token_weight = get_aux_weight(
        epoch,
        token_pool_weight,
        token_pool_warmup_epochs,
        token_pool_ramp_epochs,
    )
    effective_char_aux_weight = get_aux_weight(
        epoch,
        char_aux_weight,
        char_aux_warmup_epochs,
        char_aux_ramp_epochs,
    )

    for sample_idx, transcript in enumerate(pos_texts):
        units = unit_lists[sample_idx]
        spans = unit_spans[sample_idx]
        sim = sim_pos_list[sample_idx]
        visual_b = norm_img[sample_idx]
        total_units += len(units)
        if not units:
            continue
        assert sim.dim() == 2
        assert visual_b.dim() == 2
        assert sim.shape[1] == visual_b.shape[0]
        assert sim.shape[0] == len(units), (
            f"Text-unit length {len(units)} != sim rows {sim.shape[0]}."
        )

        path = hard_d3tw_path_from_similarity(sim)
        if not path:
            if epoch == 0 and batch_idx == 0 and sample_idx == 0:
                print(
                    f"[TOKEN POOL GROUPS] Transcript: {transcript}\n"
                    f"no restricted path exists for K={sim.shape[0]}, S={sim.shape[1]} "
                    "(this topology requires K <= S)",
                    flush=True,
                )
            continue
        assignment = path_to_assignment(
            path,
            num_chars=sim.shape[0],
            num_visual=sim.shape[1],
            device=sim.device,
        )
        pooled_token_visual, valid_mask, counts = pool_visual_by_assignment(
            visual_emb=visual_b,
            assignment=assignment,
            detach_assignment=token_pool_detach_alignment,
            min_windows_per_char=token_pool_min_windows_per_token,
        )
        assert pooled_token_visual.shape[0] == len(units)
        assert pooled_token_visual.shape[1] == visual_b.shape[1]
        if effective_token_weight > 0 and torch.is_grad_enabled():
            assert pooled_token_visual.requires_grad, "Pooled token vectors lost image gradient path"

        if epoch == 0 and batch_idx == 0 and sample_idx == 0:
            print(f"[NGRAM TOKENIZATION] Original transcript: {transcript}", flush=True)
            for unit_idx, (unit, span) in enumerate(zip(units[:40], spans[:40])):
                print(f"[{unit_idx}] token={unit!r} span={span}", flush=True)
            print("[D3TW TOKEN GROUPS]", flush=True)
            for unit_idx, visual_indices in enumerate(groups_from_assignment(assignment)[:40]):
                print(
                    f"token[{unit_idx}]={units[unit_idx]!r} span={spans[unit_idx]} "
                    f"windows={visual_indices}",
                    flush=True,
                )
            print(
                f"Shapes: visual_emb={tuple(visual_b.shape)} "
                f"token_emb=[{len(units)},{visual_b.shape[1]}] "
                f"sim={tuple(sim.shape)} pooled_token_visual={tuple(pooled_token_visual.shape)}",
                flush=True,
            )

        token_loss_i, token_stats_i = compute_token_pool_contrastive_loss(
            pooled_visual=pooled_token_visual,
            units=units,
            token_bank_embeddings=token_bank.embeddings,
            token_to_idx=token_bank.token_to_idx,
            tau=token_pool_tau,
            valid_mask=valid_mask,
            counts=counts,
        )
        token_losses.append(token_loss_i)
        valid_count = token_stats_i["token_pool_valid_tokens"]
        if valid_count:
            total_valid += valid_count
            weighted_top1 += token_stats_i["token_pool_acc_top1"] * valid_count
            weighted_top5 += token_stats_i["token_pool_acc_top5"] * valid_count
            sum_mean_windows += token_stats_i["mean_windows_per_token"] * valid_count
            window_count_weight += valid_count
            min_w = token_stats_i["min_windows_per_token"]
            max_w = token_stats_i["max_windows_per_token"]
            min_windows_global = min_w if min_windows_global is None else min(min_windows_global, min_w)
            max_windows_global = max_w if max_windows_global is None else max(max_windows_global, max_w)

        if use_char_aux_loss and char_bank is not None:
            char_aux_loss_i, char_aux_stats_i = compute_char_aux_loss_from_token_pool(
                pooled_token_visual=pooled_token_visual,
                units=units,
                spans=spans,
                original_text=transcript,
                char_bank_embeddings=char_bank.embeddings,
                char_to_idx=char_bank.char_to_idx,
                tau=char_aux_tau,
                valid_mask=valid_mask,
            )
            char_aux_losses.append(char_aux_loss_i)
            aux_valid = char_aux_stats_i["char_aux_valid_chars"]
            if aux_valid:
                char_aux_total_valid += aux_valid
                char_aux_weighted_top1 += char_aux_stats_i["char_aux_acc_top1"] * aux_valid
                char_aux_weighted_top5 += char_aux_stats_i["char_aux_acc_top5"] * aux_valid

    if token_losses:
        token_loss = torch.stack(token_losses).mean()
        if total_valid > 0 and torch.is_grad_enabled():
            assert token_loss.requires_grad, "Token-pool loss lost its image gradient path"
    else:
        if not getattr(compute_token_pool_batch_loss, "_warned_no_valid", False):
            print("[WARN] No valid D3TW token groups in batch; token-pool loss skipped.", flush=True)
            compute_token_pool_batch_loss._warned_no_valid = True
        token_loss = norm_img.sum() * 0.0

    if char_aux_losses:
        char_aux_loss = torch.stack(char_aux_losses).mean()
    else:
        char_aux_loss = norm_img.sum() * 0.0

    stats = {
        "token_pool_loss": float(token_loss.detach().item()),
        "token_pool_acc_top1": weighted_top1 / max(total_valid, 1),
        "token_pool_acc_top5": weighted_top5 / max(total_valid, 1),
        "token_pool_valid_tokens": total_valid,
        "token_pool_num_units": total_units,
        "token_pool_tau": float(token_pool_tau),
        "effective_token_pool_weight": effective_token_weight,
        "char_aux_loss": float(char_aux_loss.detach().item()),
        "effective_char_aux_weight": effective_char_aux_weight if use_char_aux_loss else 0.0,
        "char_aux_acc_top1": char_aux_weighted_top1 / max(char_aux_total_valid, 1),
        "char_aux_acc_top5": char_aux_weighted_top5 / max(char_aux_total_valid, 1),
        "char_aux_valid_chars": char_aux_total_valid,
        "char_aux_tau": float(char_aux_tau),
        "mean_windows_per_token": sum_mean_windows / max(window_count_weight, 1),
        "min_windows_per_token": int(min_windows_global or 0),
        "max_windows_per_token": int(max_windows_global or 0),
        "sequence_length": int(norm_img.shape[1]),
    }
    return token_loss, char_aux_loss, stats


def _print_loss_summary(loss_dict, batch_idx):
    if batch_idx % 10 != 0:
        return
    parts = [f"alignment_loss_type={alignment_loss_type}"]
    for key in (
        "ctc_loss", "ctc_pos_cost", "ctc_neg_cost", "ctc_gap", "ctc_top1",
        "d3tw_loss", "d3tw_cost_pos", "d3tw_cost_neg", "total_loss",
        "char_pool_loss", "effective_char_pool_weight", "char_pool_acc_top1",
        "char_pool_acc_top5", "char_pool_valid_chars", "char_pool_num_chars",
        "bigram_token_loss", "effective_bigram_token_weight",
        "bigram_token_acc_top1", "bigram_token_acc_top5", "bigram_token_valid_pairs",
        "token_pool_loss", "effective_token_pool_weight",
        "token_pool_acc_top1", "token_pool_acc_top5", "token_pool_valid_tokens",
        "token_pool_num_units", "char_aux_loss", "effective_char_aux_weight",
        "char_aux_acc_top1", "char_aux_acc_top5", "char_aux_valid_chars",
        "mean_windows_per_token", "min_windows_per_token", "max_windows_per_token",
        "window_size", "stride", "window_overlap_mode", "num_windows",
        "sequence_length",
        "input_length", "mean_target_length", "vocab_size",
    ):
        if key in loss_dict:
            value = loss_dict[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
    print("[LOSS] " + "  ".join(parts), flush=True)


def compute_batch_loss(imageEmbed, textEmbed,
                       images, pos_texts, neg_texts,
                       criterion,
                       ctc_head=None, ctc_vocab=None,
                       char_bank=None,
                       token_bank=None,
                       bigram_fusion_mlp=None,
                       ngram_tokenizer=None,
                       cross_attention_module=None,
                       epoch=0, batch_idx=0, dataloader_length=0, debug=False, timer=None):
    if timer is None:
        timer = global_timer

    def _autocast():
        return torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP)

    loss_dict = {}
    ctc_loss_value = None
    d3tw_loss_value = None
    char_pool_loss_value = None
    bigram_token_loss_value = None
    token_pool_loss_value = None
    char_aux_loss_value = None

    if _needs_ctc():
        if ctc_head is None or ctc_vocab is None:
            raise RuntimeError("CTC mode requires ctc_head and ctc_vocab.")
        timer.start('1_image_embedding')
        with _autocast():
            img_emb = imageEmbed(images, show_dims=False, timer=timer)
        img_emb = img_emb.float()
        timer.stop('1_image_embedding')

        timer.start('6_ctc_loss_computation')
        if alignment_loss_type in {'ctc', 'ctc_d3tw'}:
            ctc_loss_value, ctc_dict = compute_ctc_loss(
                img_emb, pos_texts, ctc_head, ctc_vocab, device
            )
        else:
            ctc_loss_value, ctc_dict = compute_contrastive_ctc_loss(
                img_emb, pos_texts, neg_texts, ctc_head, ctc_vocab,
                tau=contrastive_ctc_tau,
                margin=contrastive_ctc_margin,
                loss_type=contrastive_ctc_loss_type,
                device=device,
            )
        timer.stop('6_ctc_loss_computation')
        loss_dict.update(ctc_dict)

    if _needs_d3tw():
        if textEmbed is None:
            raise RuntimeError("D3TW mode requires a text embedder.")
        d3tw_loss_value, d3tw_dict, d3tw_artifacts = compute_d3tw_batch_loss(
            imageEmbed, textEmbed, images, pos_texts, neg_texts, criterion,
            cross_attention_module=cross_attention_module,
            ngram_tokenizer=ngram_tokenizer,
            epoch=epoch, batch_idx=batch_idx,
            dataloader_length=dataloader_length, debug=debug, timer=timer
        )
        loss_dict.update({f"d3tw_{k}": v for k, v in d3tw_dict.items()})
        loss_dict["d3tw_loss"] = d3tw_loss_value.detach().item()

        if _uses_char_pool():
            if _uses_ngram_units():
                if token_bank is None:
                    raise RuntimeError("text_unit_type='ngram' requires a token bank.")
                token_pool_loss_value, char_aux_loss_value, token_pool_dict = compute_token_pool_batch_loss(
                    pos_texts, d3tw_artifacts, token_bank, epoch, batch_idx=batch_idx,
                    char_bank=char_bank,
                )
                loss_dict.update(token_pool_dict)
            else:
                if char_bank is None:
                    raise RuntimeError("D3TW character-pooling mode requires a character bank.")
                char_pool_loss_value, bigram_token_loss_value, char_pool_dict = compute_char_pool_batch_loss(
                    pos_texts, d3tw_artifacts, char_bank, epoch, batch_idx=batch_idx,
                    token_bank=token_bank,
                    bigram_fusion_mlp=bigram_fusion_mlp,
                )
                loss_dict.update(char_pool_dict)
            if char_bank is None and not _uses_ngram_units():
                raise RuntimeError("D3TW character-pooling mode requires a character bank.")
            loss_dict.update({
                "window_size": imageEmbed.window_size,
                "stride": imageEmbed.stride,
                "window_overlap_mode": window_overlap_mode,
                "num_windows": (images.shape[-1] - imageEmbed.window_size)
                               // imageEmbed.stride + 1,
            })

    if alignment_loss_type == "ctc":
        total_loss = ctc_loss_value
    elif alignment_loss_type == "contrastive_ctc":
        total_loss = ctc_loss_value
    elif alignment_loss_type == "ctc_d3tw":
        total_loss = ctc_weight * ctc_loss_value + d3tw_weight * d3tw_loss_value
        if _uses_char_pool():
            if _uses_ngram_units():
                total_loss = total_loss + loss_dict["effective_token_pool_weight"] * token_pool_loss_value
                total_loss = total_loss + loss_dict["effective_char_aux_weight"] * char_aux_loss_value
            else:
                total_loss = total_loss + loss_dict["effective_char_pool_weight"] * char_pool_loss_value
                if _uses_bigram_token_loss():
                    total_loss = total_loss + loss_dict["effective_bigram_token_weight"] * bigram_token_loss_value
    elif alignment_loss_type == "contrastive_ctc_d3tw":
        total_loss = ctc_weight * ctc_loss_value + d3tw_weight * d3tw_loss_value
        if _uses_char_pool():
            if _uses_ngram_units():
                total_loss = total_loss + loss_dict["effective_token_pool_weight"] * token_pool_loss_value
                total_loss = total_loss + loss_dict["effective_char_aux_weight"] * char_aux_loss_value
            else:
                total_loss = total_loss + loss_dict["effective_char_pool_weight"] * char_pool_loss_value
                if _uses_bigram_token_loss():
                    total_loss = total_loss + loss_dict["effective_bigram_token_weight"] * bigram_token_loss_value
    elif alignment_loss_type in {"d3tw_char_pool", "contrastive_d3tw_char_pool"}:
        if _uses_ngram_units():
            total_loss = (
                d3tw_weight * d3tw_loss_value
                + loss_dict["effective_token_pool_weight"] * token_pool_loss_value
                + loss_dict["effective_char_aux_weight"] * char_aux_loss_value
            )
        else:
            total_loss = (
                d3tw_weight * d3tw_loss_value
                + loss_dict["effective_char_pool_weight"] * char_pool_loss_value
            )
            if _uses_bigram_token_loss():
                total_loss = total_loss + loss_dict["effective_bigram_token_weight"] * bigram_token_loss_value
    elif alignment_loss_type == "d3tw":
        if _uses_char_pool():
            if _uses_ngram_units():
                total_loss = (
                    d3tw_weight * d3tw_loss_value
                    + loss_dict["effective_token_pool_weight"] * token_pool_loss_value
                    + loss_dict["effective_char_aux_weight"] * char_aux_loss_value
                )
            else:
                total_loss = (
                    d3tw_weight * d3tw_loss_value
                    + loss_dict["effective_char_pool_weight"] * char_pool_loss_value
                )
                if _uses_bigram_token_loss():
                    total_loss = total_loss + loss_dict["effective_bigram_token_weight"] * bigram_token_loss_value
        else:
            total_loss = d3tw_loss_value
    else:
        raise ValueError(f"Unknown alignment_loss_type: {alignment_loss_type}")

    loss_dict["total_loss"] = total_loss.detach().item()
    _print_loss_summary(loss_dict, batch_idx)
    return total_loss


def Train(imageEmbedding, textEmbedding, trainLoader, validLoader, criterion,
          ctc_head=None, ctc_vocab=None,
          char_bank=None,
          token_bank=None,
          bigram_fusion_mlp=None,
          ngram_tokenizer=None,
          cross_attention_module=None,
          lr=None, total_epochs=None, resume_path=None, model_config=None):
    """Train the image-text alignment model (CRNN: ResNet34 + BiLSTM)."""
    imageEmbedding.train()
    if textEmbedding is not None:
        textEmbedding.eval()
    # Make model_config accessible to compute_batch_loss for mid-epoch saves.
    _model_config = model_config
    compute_batch_loss._model_config = model_config

    if lr is None:
        lr = learning_rate
    if total_epochs is None:
        total_epochs = epochs

    # Differential learning rates: low LR for pre-trained CNN, higher for BiLSTM
    cnn_params = list(imageEmbedding.cnn_encoder.parameters())
    cnn_param_ids = set(id(p) for p in cnn_params)
    other_params = [p for p in imageEmbedding.parameters() if id(p) not in cnn_param_ids]

    optim_params = cnn_params + other_params
    if ctc_head is not None:
        optim_params += list(ctc_head.parameters())
    if cross_attention_module is not None and cross_attention_weight > 0:
        cross_params = [
            p for p in cross_attention_module.parameters() if p.requires_grad
        ]
        optim_params += cross_params
    if bigram_fusion_mlp is not None:
        optim_params += [p for p in bigram_fusion_mlp.parameters() if p.requires_grad]
    optimizer = optim.Adam(optim_params, lr=lr)
    # Cosine schedule: ReduceLROnPlateau was monitoring a loss that's bounded
    # below by the contrastive margin, so it kept halving LR to zero even
    # though training had room to learn.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=lr * 0.01
    )

    # GradScaler only does anything when AMP is on and dtype is fp16
    # (bf16 has the same dynamic range as fp32 and doesn't need scaling).
    use_scaler = USE_AMP and AMP_DTYPE == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    start_epoch = 0
    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        imageEmbedding.load_state_dict(_extract_model_state(ckpt))
        if ctc_head is not None and ckpt.get('ctc_head_state_dict') is not None:
            ctc_head.load_state_dict(ckpt['ctc_head_state_dict'])
        if (
            cross_attention_module is not None
            and ckpt.get('cross_attention_state_dict') is not None
        ):
            cross_attention_module.load_state_dict(ckpt['cross_attention_state_dict'])
        if (
            bigram_fusion_mlp is not None
            and ckpt.get('bigram_fusion_mlp_state_dict') is not None
        ):
            bigram_fusion_mlp.load_state_dict(ckpt['bigram_fusion_mlp_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if ckpt.get('scaler_state_dict') is not None and use_scaler:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}/{total_epochs}")

    loss_lst = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()

        if hasattr(criterion, 'gamma'):
            criterion.gamma = contrastive_soft_dtw_gamma
            print(f"Epoch {epoch+1} - Soft-DTW gamma: {contrastive_soft_dtw_gamma:.6f}", flush=True)

        imageEmbedding.train()
        if ctc_head is not None:
            ctc_head.train()
        if cross_attention_module is not None:
            cross_attention_module.train()
        if bigram_fusion_mlp is not None:
            bigram_fusion_mlp.train()
        global_timer.reset()

        train_loss = 0.0

        for batch_idx, (images, pos_texts, neg_texts) in enumerate(trainLoader):
            global_timer.start('0_data_to_gpu')
            images = images.to(device, non_blocking=True)
            global_timer.stop('0_data_to_gpu')

            # set_to_none=True is faster (skips zeroing) AND uses less memory
            # (releases the grad buffers until they're recreated by backward).
            optimizer.zero_grad(set_to_none=True)

            loss = compute_batch_loss(
                imageEmbedding, textEmbedding,
                images, pos_texts, neg_texts,
                criterion,
                ctc_head=ctc_head,
                ctc_vocab=ctc_vocab,
                char_bank=char_bank,
                token_bank=token_bank,
                bigram_fusion_mlp=bigram_fusion_mlp,
                ngram_tokenizer=ngram_tokenizer,
                cross_attention_module=cross_attention_module,
                epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                debug=debug, timer=global_timer
            )

            train_loss += loss.item()

            global_timer.start('9_backward')
            scaler.scale(loss).backward()
            global_timer.stop('9_backward')

            global_timer.start('10_optimizer_step')
            # Unscale before clipping so the clip threshold applies to the
            # real (unscaled) gradients.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            global_timer.stop('10_optimizer_step')

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}", flush=True)

        train_loss = train_loss / len(trainLoader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}', flush=True)

        global_timer.print_summary(f"Epoch {epoch+1} Timing Summary (Training)")

        # Validation phase
        imageEmbedding.eval()
        if ctc_head is not None:
            ctc_head.eval()
        if cross_attention_module is not None:
            cross_attention_module.eval()
        if bigram_fusion_mlp is not None:
            bigram_fusion_mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, pos_texts, neg_texts) in enumerate(validLoader):
                images = images.to(device, non_blocking=True)

                loss = compute_batch_loss(
                    imageEmbedding, textEmbedding,
                    images, pos_texts, neg_texts,
                    criterion=criterion
	                    , ctc_head=ctc_head,
	                    ctc_vocab=ctc_vocab
	                    , char_bank=char_bank
	                    , token_bank=token_bank
	                    , bigram_fusion_mlp=bigram_fusion_mlp
	                    , ngram_tokenizer=ngram_tokenizer
	                    , cross_attention_module=cross_attention_module
                    , epoch=epoch
                    , batch_idx=batch_idx
                )
                val_loss += loss.item()

        val_loss = val_loss / len(validLoader)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}', flush=True)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_minutes = int(epoch_duration // 60)
        epoch_seconds = epoch_duration % 60
        print(f'Epoch {epoch+1} completed in {epoch_minutes}m {epoch_seconds:.2f}s', flush=True)

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f'Epoch {epoch+1} - Peak GPU memory: {peak_mb:.1f} MB', flush=True)
            torch.cuda.reset_peak_memory_stats()
        print('=' * 60, flush=True)

        scheduler.step()

        # Save full checkpoint each epoch so --resume can recover exactly.
        save_checkpoint(imageEmbedding, optimizer, scheduler, scaler, epoch,
                        job_id, model_config=_model_config,
                        ctc_head=ctc_head, ctc_vocab=ctc_vocab,
                        cross_attention_module=cross_attention_module,
                        bigram_fusion_mlp=bigram_fusion_mlp)
        weights_path = save_model_weights(imageEmbedding, epoch, job_id,
                                          model_config=_model_config,
                                          ctc_head=ctc_head, ctc_vocab=ctc_vocab,
                                          cross_attention_module=cross_attention_module,
                                          bigram_fusion_mlp=bigram_fusion_mlp)

        if debug_wandb:
            # One W&B loss update per completed epoch.
            update_wandb(epoch + 1, train_loss, val_loss)
            if (epoch + 1) % 10 == 0:
                log_wandb_weights(job_id, epoch + 1, weights_path)
        loss_lst.append(train_loss)

    return loss_lst


def _apply_override(params_module, args, arg_name, param_name=None, transform=None):
    value = getattr(args, arg_name)
    if value is None:
        return
    if transform is not None:
        value = transform(value)
    target_name = param_name or arg_name
    setattr(params_module, target_name, value)
    globals()[target_name] = value


def _apply_cli_overrides(args):
    # Keep Parameters.py and train.py's imported globals in sync. Many helper
    # functions in this module read these names directly.
    import Parameters as _params

    lowercase = str.lower
    overrides = (
        ("window_size",),
        ("stride_ratio",),
        ("window_overlap_mode", None, lowercase),
        ("text_embedder", "text_embedder_type", lowercase),
        ("loss_type", "contrastive_loss_type", lowercase),
        ("epochs",),
        ("learning_rate",),
        ("alignment_loss_type", None, lowercase),
        ("use_d3tw_char_pooling",),
        ("char_pool_weight",),
        ("char_pool_tau",),
        ("char_pool_warmup_epochs",),
        ("char_pool_ramp_epochs",),
        ("char_pool_method",),
        ("char_pool_detach_alignment",),
        ("char_pool_skip_spaces",),
        ("char_pool_min_windows_per_char",),
        ("text_unit_type", None, lowercase),
        ("ngram_min_n",),
        ("ngram_max_n",),
        ("ngram_min_freq",),
        ("ngram_max_vocab_size",),
        ("ngram_tokenizer_mode",),
        ("ngram_skip_spaces",),
        ("ngram_include_ligatures",),
        ("token_pool_weight",),
        ("token_pool_tau",),
        ("token_pool_warmup_epochs",),
        ("token_pool_ramp_epochs",),
        ("token_pool_detach_alignment",),
        ("token_pool_min_windows_per_token",),
        ("use_char_aux_loss",),
        ("char_aux_weight",),
        ("char_aux_tau",),
        ("char_aux_warmup_epochs",),
        ("char_aux_ramp_epochs",),
        ("use_bigram_token_loss",),
        ("bigram_token_weight",),
        ("bigram_token_tau",),
        ("bigram_token_warmup_epochs",),
        ("bigram_token_ramp_epochs",),
        ("bigram_token_skip_spaces",),
        ("bigram_token_min_freq",),
        ("bigram_token_max_vocab_size",),
        ("bigram_token_fusion",),
        ("bigram_token_include_ligatures",),
        ("ctc_weight",),
        ("d3tw_weight",),
        ("contrastive_ctc_loss_type", None, lowercase),
        ("contrastive_ctc_tau",),
        ("contrastive_ctc_margin",),
        ("transformer_num_layers",),
        ("transformer_num_heads",),
        ("transformer_ff_dim",),
        ("transformer_dropout",),
        ("transformer_activation", None, lowercase),
        ("transformer_norm_first",),
        ("transformer_positional_encoding", None, lowercase),
        ("transformer_max_len",),
        ("return_attention_weights",),
        ("use_cross_attention",),
        ("cross_attention_type", None, lowercase),
        ("cross_attention_num_heads",),
        ("cross_attention_dropout",),
        ("cross_attention_weight",),
    )
    for override in overrides:
        _apply_override(_params, args, *override)

    if args.negative_mode is not None:
        _params.negative_mode = args.negative_mode.lower()
    if args.multi_scale:
        _params.multi_scale_enabled = True
        globals()["multi_scale_enabled"] = True
    if args.num_negatives is not None:
        _params.num_negatives = args.num_negatives
        globals()["num_negatives"] = args.num_negatives
        import newDataLoader as _ndl
        _ndl.num_negatives = args.num_negatives
    if args.sequence_encoder_type is not None:
        _params.sequence_encoder_type = args.sequence_encoder_type.lower()
        _params.use_bilstm = _params.sequence_encoder_type == "bilstm"
        globals()["sequence_encoder_type"] = _params.sequence_encoder_type
        globals()["use_bilstm"] = _params.use_bilstm

    return _params


def _validate_alignment_config(params_module):
    if alignment_loss_type in {'d3tw_char_pool', 'contrastive_d3tw_char_pool'}:
        params_module.use_d3tw_char_pooling = True
        globals()["use_d3tw_char_pooling"] = True
    if _uses_char_pool() and multi_scale_enabled:
        raise NotImplementedError("D3TW character pooling currently supports single-scale training only.")
    if _uses_char_pool() and not _needs_d3tw():
        raise ValueError("use_d3tw_char_pooling requires an alignment mode containing D3TW.")
    if _uses_char_pool() and char_pool_method == "soft_weighted":
        raise NotImplementedError(
            "char_pool_method='soft_weighted' is configured but the current Soft-DTW "
            "backend does not expose alignment weights. Use hard_mean."
        )
    if text_unit_type not in {"char", "ngram"}:
        raise ValueError("text_unit_type must be 'char' or 'ngram'.")
    if _uses_ngram_units() and not _uses_char_pool():
        raise ValueError("text_unit_type='ngram' requires d3tw_char_pool/use_d3tw_char_pooling.")
    if _uses_char_pool() and not _uses_ngram_units() and not char_pool_use_char_bank:
        raise NotImplementedError("Character-pooling currently requires char_pool_use_char_bank=True.")
    if _uses_bigram_token_loss() and _uses_ngram_units():
        print(
            "[BIGRAM TOKEN] auxiliary bigram loss is ignored in text_unit_type='ngram'; "
            "token_pool_loss is the main token-level supervision.",
            flush=True,
        )
        params_module.use_bigram_token_loss = False
        globals()["use_bigram_token_loss"] = False
    if _uses_bigram_token_loss() and not _uses_char_pool():
        raise ValueError("use_bigram_token_loss requires D3TW character pooling.")
    if _uses_bigram_token_loss() and bigram_token_fusion not in {"mean", "mlp"}:
        raise ValueError("bigram_token_fusion must be 'mean' or 'mlp'.")


def _prepare_training_inputs(args):
    # Precedence: --data_dir > --finetune (finetune_data_dir) > Parameters.py defaults.
    if args.data_dir is not None:
        resolved_data_dir = args.data_dir
    elif args.finetune:
        resolved_data_dir = finetune_data_dir
    else:
        resolved_data_dir = None

    if args.finetune:
        run_lr = finetune_learning_rate
        run_epochs = finetune_epochs
    else:
        run_lr = learning_rate
        run_epochs = epochs

    if resolved_data_dir is not None:
        print(f"Loading dataset from: {resolved_data_dir}")
        train_loader, valid_loader, test_loader = build_dataloaders(resolved_data_dir)
    else:
        train_loader, valid_loader, test_loader = (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
        )

    resume_ckpt = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=device)

    return (
        resolved_data_dir,
        run_lr,
        run_epochs,
        train_loader,
        valid_loader,
        test_loader,
        resume_ckpt,
    )


def _build_text_embedding():
    if not _needs_d3tw():
        print("[TEXT EMBED] skipped for CTC-only objective", flush=True)
        return None

    # Pick the text embedder via Parameters.text_embedder_type
    # ('char' = learned frozen table, 'fasttext' = facebook/fasttext-ar-vectors).
    text_embedding = build_text_embedder(embedding_dim=vector_size)
    text_embedding = text_embedding.to(device)
    for param in text_embedding.parameters():
        param.requires_grad_(False)
    text_embedding.eval()
    assert not any(param.requires_grad for param in text_embedding.parameters()), \
        "Text branch must be fully frozen — some parameters still have requires_grad=True"
    print(f"[TEXT EMBED] type={text_embedder_type}  out_dim={vector_size}  (frozen)", flush=True)
    return text_embedding


def _build_image_embedding(stride):
    return EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=(lang.lower() == "arabic"),
        sequence_encoder_type=sequence_encoder_type,
        use_bilstm=(sequence_encoder_type == "bilstm"),
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        dropout=model_dropout,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_ff_dim=transformer_ff_dim,
        transformer_dropout=transformer_dropout,
        transformer_activation=transformer_activation,
        transformer_norm_first=transformer_norm_first,
        transformer_positional_encoding=transformer_positional_encoding,
        transformer_max_len=transformer_max_len,
        return_attention_weights=return_attention_weights,
    )


def _build_cross_attention_module():
    if not use_cross_attention:
        return None
    if cross_attention_weight > 0 and _needs_d3tw():
        cross_attention_module = CrossAttentionSimilarity(
            dim=vector_size,
            num_heads=cross_attention_num_heads,
            dropout=cross_attention_dropout,
            attention_type=cross_attention_type,
        ).to(device)
        print(
            "[CROSS-ATTN] enabled as auxiliary similarity "
            f"type={cross_attention_type} heads={cross_attention_num_heads} "
            f"weight={cross_attention_weight}",
            flush=True,
        )
        return cross_attention_module

    print(
        "[CROSS-ATTN] requested but inactive "
        f"(needs_d3tw={_needs_d3tw()} weight={cross_attention_weight})",
        flush=True,
    )
    return None


def _prepare_image_embedding_for_training(args, image_embedding):
    # --resume restores model + optimizer + scheduler + epoch inside Train().
    # --pretrained_weights only loads model weights here (fresh optimizer/epoch).
    if args.resume is not None and args.pretrained_weights is not None:
        raise SystemExit("Pass either --resume or --pretrained_weights, not both.")

    if args.pretrained_weights is not None:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        loaded = torch.load(args.pretrained_weights, map_location=device)
        image_embedding.load_state_dict(_extract_model_state(loaded))
        print("Pretrained weights loaded successfully.")

    assert any(param.requires_grad for param in image_embedding.parameters()), \
        "Image branch has no trainable parameters — check EmbeddingModel construction"

    if show_gradients:
        for param in image_embedding.parameters():
            param.register_hook(check_grad)


def _build_ctc_components(train_loader, resolved_data_dir, resume_ckpt):
    if not _needs_ctc():
        return None, None

    if resume_ckpt is not None and resume_ckpt.get("ctc_vocab") is not None:
        ctc_vocab = CTCVocabulary.from_dict(resume_ckpt["ctc_vocab"])
    else:
        vocab_texts = _collect_training_texts(train_loader, resolved_data_dir)
        if not vocab_texts:
            raise RuntimeError("Could not collect transcripts to build CTC vocabulary.")
        ctc_vocab = CTCVocabulary.from_texts(vocab_texts, blank_token=ctc_blank_token)

    ctc_head = nn.Linear(vector_size, len(ctc_vocab)).to(device)
    if save_ctc_vocab:
        ctc_vocab_path = os.path.join(_weights_dir(job_id), "ctc_vocab.json")
        ctc_vocab.save_json(ctc_vocab_path)
        print(f"[CTC] saved vocabulary: {ctc_vocab_path}", flush=True)
    print(
        f"[CTC] vocab_size={len(ctc_vocab)} blank_idx={ctc_vocab.blank_idx} "
        f"blank_token={ctc_vocab.blank_token!r}",
        flush=True,
    )
    return ctc_vocab, ctc_head


def _build_pooling_banks(train_loader, resolved_data_dir, text_embedding):
    char_bank = None
    token_bank = None
    bigram_fusion_mlp = None
    ngram_tokenizer = None

    if not _uses_char_pool():
        return char_bank, token_bank, bigram_fusion_mlp, ngram_tokenizer

    bank_texts = _collect_training_texts(train_loader, resolved_data_dir)
    if not bank_texts:
        raise RuntimeError("Could not collect training transcripts for pooling banks.")

    if _uses_ngram_units():
        tokens = collect_ngram_tokens(
            bank_texts,
            min_n=ngram_min_n,
            max_n=ngram_max_n,
            min_freq=ngram_min_freq,
            max_vocab_size=ngram_max_vocab_size,
            skip_spaces=ngram_skip_spaces,
            include_ligatures=ngram_include_ligatures,
            ligatures=ngram_ligatures,
        )
        if not tokens:
            raise RuntimeError("No n-gram text-unit tokens were collected.")
        ngram_tokenizer = NGramTokenizer(tokens, mode=ngram_tokenizer_mode)
        token_embeddings, token_to_idx, idx_to_token = build_token_embedding_bank(
            text_embedding, tokens, device
        )
        token_bank = TokenBank(
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            embeddings=token_embeddings,
        )
        assert not token_bank.embeddings.requires_grad, "Token bank must be frozen"
        vocab_path = os.path.join(_weights_dir(job_id), "ngram_vocab.json")
        save_ngram_vocab_json(vocab_path, idx_to_token)
        token_bank_path = _save_token_bank(token_bank, job_id)
        _make_model_config._ngram_vocab_size = len(token_bank.idx_to_token)
        print(
            f"[NGRAM TOKEN] vocab size={len(token_bank.idx_to_token)} "
            f"first 30 tokens={token_bank.idx_to_token[:30]} "
            f"embedding dim={token_bank.embeddings.shape[1]} "
            f"vocab={vocab_path} token_bank={token_bank_path}",
            flush=True,
        )

        if use_char_aux_loss:
            chars = collect_unique_chars(bank_texts, skip_spaces=False)
            bank_embeddings, char_to_idx, idx_to_char = build_char_bank(
                text_embedding, chars, device
            )
            char_bank = CharacterBank(
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                embeddings=bank_embeddings,
            )
            assert not char_bank.embeddings.requires_grad, "Character bank must be frozen"
            char_bank_path = _save_char_bank(char_bank, job_id)
            print(
                f"[CHAR AUX] char bank size={len(char_bank.idx_to_char)} "
                f"first 20 chars={char_bank.idx_to_char[:20]} "
                f"embedding dim={char_bank.embeddings.shape[1]} saved={char_bank_path}",
                flush=True,
            )
    else:
        chars = collect_unique_chars(
            bank_texts, skip_spaces=char_pool_skip_spaces
        )
        bank_embeddings, char_to_idx, idx_to_char = build_char_bank(
            text_embedding, chars, device
        )
        char_bank = CharacterBank(
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            embeddings=bank_embeddings,
        )
        assert not char_bank.embeddings.requires_grad, "Character bank must be frozen"
        char_bank_path = _save_char_bank(char_bank, job_id)
        print(
            f"[CHAR POOL] char bank size={len(char_bank.idx_to_char)} "
            f"first 20 chars={char_bank.idx_to_char[:20]} "
            f"embedding dim={char_bank.embeddings.shape[1]} "
            f"method={char_pool_method} saved={char_bank_path}",
            flush=True,
        )

    if (not _uses_ngram_units()) and _uses_bigram_token_loss():
        tokens = collect_bigram_tokens(
            bank_texts,
            skip_spaces=bigram_token_skip_spaces,
            min_freq=bigram_token_min_freq,
            max_vocab_size=bigram_token_max_vocab_size,
            include_ligatures=bigram_token_include_ligatures,
        )
        if not tokens:
            print(
                "[BIGRAM TOKEN] WARNING: no bigram tokens found; disabling bigram token loss.",
                flush=True,
            )
            import Parameters as _P
            _P.use_bigram_token_loss = False
            globals()["use_bigram_token_loss"] = False
        else:
            token_embeddings, token_to_idx, idx_to_token = build_token_bank(
                text_embedding, tokens, device
            )
            token_bank = TokenBank(
                token_to_idx=token_to_idx,
                idx_to_token=idx_to_token,
                embeddings=token_embeddings,
            )
            assert not token_bank.embeddings.requires_grad, "Token bank must be frozen"
            token_bank_path = _save_token_bank(token_bank, job_id)
            _make_model_config._bigram_token_vocab_size = len(token_bank.idx_to_token)
            print(
                f"[BIGRAM TOKEN] token bank size={len(token_bank.idx_to_token)} "
                f"first 30 tokens={token_bank.idx_to_token[:30]} "
                f"embedding dim={token_bank.embeddings.shape[1]} "
                f"fusion={bigram_token_fusion} saved={token_bank_path}",
                flush=True,
            )
            if bigram_token_fusion == "mlp":
                bigram_fusion_mlp = BigramFusionMLP(vector_size).to(device)

    return char_bank, token_bank, bigram_fusion_mlp, ngram_tokenizer


def _print_run_summary(args, stride, run_lr, run_epochs, resolved_data_dir,
                       cross_attention_module):
    print(f"\n=== Architecture: Image Encoder (ResNet34 + {sequence_encoder_type}) ===")
    print(f"[OPT 1] job_id: {job_id}")
    print(f"[OPT 2] Sequence Encoder: {sequence_encoder_type}")
    if sequence_encoder_type == "bilstm":
        print(f"[OPT 2a] BiLSTM layers={bilstm_layers} hidden_dim={bilstm_hidden_dim}")
    if sequence_encoder_type == "transformer":
        print(
            f"[OPT 2a] Transformer layers={transformer_num_layers} "
            f"heads={transformer_num_heads} ff_dim={transformer_ff_dim} "
            f"dropout={transformer_dropout} pos={transformer_positional_encoding}"
        )
    print(
        f"[OPT 3] Sliding Window: window_overlap_mode={window_overlap_mode} "
        f"window_size={window_size} stride={stride} stride_ratio={stride_ratio}"
    )
    print(f"[OPT 4] In-Batch Negatives: num_negatives={num_negatives}")
    print(f"[OPT 5] alignment_loss_type={alignment_loss_type}")
    if _needs_ctc():
        print(f"[OPT 6] CTC: weight={ctc_weight} loss={contrastive_ctc_loss_type} tau={contrastive_ctc_tau} margin={contrastive_ctc_margin}")
    if _needs_d3tw():
        print(f"[OPT 7] D3TW: weight={d3tw_weight}")
        print(
            f"[OPT 7a] Cross-Attention: enabled={cross_attention_module is not None} "
            f"type={cross_attention_type} weight={cross_attention_weight}"
        )
    if _uses_char_pool():
        if _uses_ngram_units():
            print(
                f"[OPT 7b] N-gram Token D3TW: n={ngram_min_n}-{ngram_max_n} "
                f"min_freq={ngram_min_freq} max_vocab={ngram_max_vocab_size} "
                f"mode={ngram_tokenizer_mode} skip_spaces={ngram_skip_spaces}"
            )
            print(
                f"[OPT 7c] Token Pool: weight={token_pool_weight} tau={token_pool_tau} "
                f"warmup={token_pool_warmup_epochs} ramp={token_pool_ramp_epochs} "
                f"detach_alignment={token_pool_detach_alignment} "
                f"min_windows={token_pool_min_windows_per_token}"
            )
            print(
                f"[OPT 7d] Char Aux: enabled={use_char_aux_loss} "
                f"weight={char_aux_weight} tau={char_aux_tau} "
                f"warmup={char_aux_warmup_epochs} ramp={char_aux_ramp_epochs}"
            )
        else:
            print(
                f"[OPT 7b] Character Pool: weight={char_pool_weight} tau={char_pool_tau} "
                f"warmup={char_pool_warmup_epochs} ramp={char_pool_ramp_epochs} "
                f"method={char_pool_method} detach_alignment={char_pool_detach_alignment} "
                f"skip_spaces={char_pool_skip_spaces} min_windows={char_pool_min_windows_per_char}"
            )
    if _uses_bigram_token_loss():
        print(
            f"[OPT 7c] Bigram Token Loss: weight={bigram_token_weight} tau={bigram_token_tau} "
            f"warmup={bigram_token_warmup_epochs} ramp={bigram_token_ramp_epochs} "
            f"skip_spaces={bigram_token_skip_spaces} min_freq={bigram_token_min_freq} "
            f"max_vocab={bigram_token_max_vocab_size} fusion={bigram_token_fusion}"
        )
    if multi_scale_enabled:
        print(f"[OPT 8] Multi-Scale Alignment: windows={multi_scale_window_sizes}, alpha={multi_scale_alpha}")
    print(f"[SPEED] AMP={USE_AMP} (dtype={AMP_DTYPE})  Profiling={ENABLE_PROFILING}")
    print(f"[RUN]   lr={run_lr}  epochs={run_epochs}")
    if args.finetune:
        print(f"[FINETUNE] mode=ON  dataset={resolved_data_dir}")
    if args.pretrained_weights:
        print(f"[FINETUNE] Pretrained weights: {args.pretrained_weights}")
    if args.resume:
        print(f"[FINETUNE] Resuming from: {args.resume}")
    if args.data_dir:
        print(f"[FINETUNE] Dataset override: {args.data_dir}")
    print(f"================================================\n")


def main(args):
    global job_id

    _params = _apply_cli_overrides(args)
    job_id = args.job_id

    if debug_wandb:
        init_wandb(job_id)

    _validate_alignment_config(_params)

    stride = compute_stride(window_size, stride_ratio, window_overlap_mode)
    print(
        f"Using window_size={window_size}, stride={stride}, "
        f"window_overlap_mode={window_overlap_mode}"
    )

    (
        resolved_data_dir,
        run_lr,
        run_epochs,
        _train_dl,
        _valid_dl,
        _test_dl,
        resume_ckpt,
    ) = _prepare_training_inputs(args)

    textEmbedding = _build_text_embedding()

    imageEmbedding = _build_image_embedding(stride)

    cross_attention_module = _build_cross_attention_module()

    _prepare_image_embedding_for_training(args, imageEmbedding)

    ctc_vocab, ctc_head = _build_ctc_components(
        _train_dl, resolved_data_dir, resume_ckpt
    )

    char_bank, token_bank, bigram_fusion_mlp, ngram_tokenizer = _build_pooling_banks(
        _train_dl, resolved_data_dir, textEmbedding
    )

    criterion    = Loss_choice() if _needs_d3tw() else None
    _model_cfg   = _make_model_config(window_size, stride, stride_ratio, ctc_vocab)

    _print_run_summary(
        args, stride, run_lr, run_epochs, resolved_data_dir,
        cross_attention_module,
    )

    loss_lst = Train(
        imageEmbedding,
        textEmbedding,
        _train_dl,
        _valid_dl,
        criterion,
        ctc_head=ctc_head,
        ctc_vocab=ctc_vocab,
        char_bank=char_bank,
        token_bank=token_bank,
        bigram_fusion_mlp=bigram_fusion_mlp,
        ngram_tokenizer=ngram_tokenizer,
        cross_attention_module=cross_attention_module,
        lr=run_lr,
        total_epochs=run_epochs,
        resume_path=args.resume,
        model_config=_model_cfg,
    )

    if debug_wandb:
        wandb.finish()

    return loss_lst


if __name__ == '__main__':
    main(parser.parse_args())

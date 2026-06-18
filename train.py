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
# SimilarityTransformer removed - using direct cosine similarity between CNN+BiLSTM and text embeddings

import os
import gc
import time
import wandb
import warnings
import argparse

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
                             'contrastive_ctc_d3tw'],
                    help='Alignment objective to train.')
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
args = parser.parse_args()
job_id = args.job_id

from wandb_config import init_wandb, update_wandb, upload_artifacts_to_wandb

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
        'vector_size':        _P.vector_size,
        'use_bilstm':         _P.use_bilstm,
        'bilstm_layers':      _P.bilstm_layers,
        'bilstm_hidden_dim':  _P.bilstm_hidden_dim,
        'lang':               _P.lang,
        'text_embedder_type': _P.text_embedder_type,
        'negative_mode':      getattr(_P, 'negative_mode', 'mixed'),
        'multi_scale_enabled': _P.multi_scale_enabled,
    }
    if ctc_vocab is not None:
        cfg.update({
            'ctc_vocab_size': len(ctc_vocab),
            'ctc_blank_idx': ctc_vocab.blank_idx,
        })
    return cfg


def save_model_weights(model, epoch, job_id, model_config=None,
                       ctc_head=None, ctc_vocab=None):
    weights_path = os.path.join(_weights_dir(job_id), "model_latest.pth")
    payload = {
        'model_state_dict': model.state_dict(),
        'image_model_state_dict': model.state_dict(),
    }
    if ctc_head is not None:
        payload['ctc_head_state_dict'] = ctc_head.state_dict()
    if ctc_vocab is not None:
        payload['ctc_vocab'] = ctc_vocab.to_dict()
    if model_config:
        payload['model_config'] = model_config
    if ctc_head is not None or ctc_vocab is not None or model_config:
        torch.save(payload, weights_path)
    else:
        torch.save(model.state_dict(), weights_path)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, job_id,
                    model_config=None, ctc_head=None, ctc_vocab=None):
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
                                   'contrastive_ctc_d3tw'}


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


def compute_varlen_similarities(textEmbed, norm_img, pos_texts, neg_texts):
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
    sim_pos_list = []
    sim_neg_list = []

    for i, text in enumerate(pos_texts):
        with torch.no_grad():
            txt_emb = textEmbed(text)           # [T_i, D]
            txt_emb = normalize_func(txt_emb)

        # sim carries grad via norm_img[i]
        sim_pos = torch.einsum("td,sd->ts", txt_emb, norm_img[i])
        sim_pos_list.append(sim_pos)

        sample_neg_sims = []
        for neg_text in neg_texts[i]:
            with torch.no_grad():
                neg_emb = textEmbed(neg_text)   # [T_neg, D]
                neg_emb = normalize_func(neg_emb)
            sim_neg = torch.einsum("td,sd->ts", neg_emb, norm_img[i])
            sample_neg_sims.append(sim_neg)

        sim_neg_list.append(sample_neg_sims)

    return sim_pos_list, sim_neg_list


def compute_d3tw_batch_loss(imageEmbed, textEmbed,
                            images, pos_texts, neg_texts,
                            criterion,
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
            sim_pos_s = torch.einsum('bsv,btv->bst', norm_pos_text, norm_img_s)
            sim_neg_s = torch.einsum('bktv,bsv->bkts', stacked_neg, norm_img_s)
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
        sim_pos_list, sim_neg_list = compute_varlen_similarities(
            textEmbed=textEmbed,
            norm_img=norm_img,
            pos_texts=pos_texts,
            neg_texts=neg_texts,
        )
        timer.stop('4_varlen_similarities')

        # One-time sanity check: print shapes for the first batch of training
        if batch_idx == 0 and epoch == 0:
            print(f"[SANITY] sim_pos_list[0].shape = {sim_pos_list[0].shape}  "
                  f"(expected [{len(pos_texts[0])}, {norm_img.shape[1]}])", flush=True)
            assert sim_pos_list[0].shape[0] == len(pos_texts[0]), \
                f"Padded text rows detected: got {sim_pos_list[0].shape[0]} rows but text has {len(pos_texts[0])} chars"

        timer.start('6_loss_computation')
        total_loss, loss_dict = criterion.forward_varlen(sim_pos_list, sim_neg_list)
        timer.stop('6_loss_computation')

        # Use first sample's sim for visualization (no padding)
        sim_pos_for_viz = sim_pos_list[0].unsqueeze(0)

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
            save_model_weights(imageEmbed, epoch, job_id, model_config=_cfg)

    return total_loss, loss_dict


def _print_loss_summary(loss_dict, batch_idx):
    if batch_idx % 10 != 0:
        return
    parts = [f"alignment_loss_type={alignment_loss_type}"]
    for key in (
        "ctc_loss", "ctc_pos_cost", "ctc_neg_cost", "ctc_gap", "ctc_top1",
        "d3tw_loss", "d3tw_cost_pos", "d3tw_cost_neg", "total_loss",
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
                       epoch=0, batch_idx=0, dataloader_length=0, debug=False, timer=None):
    if timer is None:
        timer = global_timer

    def _autocast():
        return torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP)

    loss_dict = {}
    ctc_loss_value = None
    d3tw_loss_value = None

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
        d3tw_loss_value, d3tw_dict = compute_d3tw_batch_loss(
            imageEmbed, textEmbed, images, pos_texts, neg_texts, criterion,
            epoch=epoch, batch_idx=batch_idx,
            dataloader_length=dataloader_length, debug=debug, timer=timer
        )
        loss_dict.update({f"d3tw_{k}": v for k, v in d3tw_dict.items()})
        loss_dict["d3tw_loss"] = d3tw_loss_value.detach().item()

    if alignment_loss_type == "ctc":
        total_loss = ctc_loss_value
    elif alignment_loss_type == "contrastive_ctc":
        total_loss = ctc_loss_value
    elif alignment_loss_type == "ctc_d3tw":
        total_loss = ctc_weight * ctc_loss_value + d3tw_weight * d3tw_loss_value
    elif alignment_loss_type == "contrastive_ctc_d3tw":
        total_loss = ctc_weight * ctc_loss_value + d3tw_weight * d3tw_loss_value
    elif alignment_loss_type == "d3tw":
        total_loss = d3tw_loss_value
    else:
        raise ValueError(f"Unknown alignment_loss_type: {alignment_loss_type}")

    loss_dict["total_loss"] = total_loss.detach().item()
    _print_loss_summary(loss_dict, batch_idx)
    return total_loss


def Train(imageEmbedding, textEmbedding, trainLoader, validLoader, criterion,
          ctc_head=None, ctc_vocab=None,
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
                        ctc_head=ctc_head, ctc_vocab=ctc_vocab)
        save_model_weights(imageEmbedding, epoch, job_id,
                           model_config=_model_config,
                           ctc_head=ctc_head, ctc_vocab=ctc_vocab)

        if debug_wandb:
            update_wandb(train_loss, val_loss)
            if epoch % 10 == 0:
                upload_artifacts_to_wandb(job_id, epoch)
        loss_lst.append(train_loss)

    return loss_lst


if __name__ == '__main__':
    # Apply CLI overrides to the imported Parameters module so downstream code
    # picks them up without any other changes.
    import Parameters as _params
    if args.window_size is not None:
        _params.window_size = args.window_size
        window_size = args.window_size
    if args.stride_ratio is not None:
        _params.stride_ratio = args.stride_ratio
        stride_ratio = args.stride_ratio
    if args.text_embedder is not None:
        _params.text_embedder_type = args.text_embedder.lower()
        text_embedder_type = _params.text_embedder_type
    if args.negative_mode is not None:
        _params.negative_mode = args.negative_mode.lower()
    if args.multi_scale:
        _params.multi_scale_enabled = True
        multi_scale_enabled = True
    if args.loss_type is not None:
        _params.contrastive_loss_type = args.loss_type.lower()
        contrastive_loss_type = _params.contrastive_loss_type
    if args.epochs is not None:
        _params.epochs = args.epochs
        epochs = args.epochs
    if args.learning_rate is not None:
        _params.learning_rate = args.learning_rate
        learning_rate = args.learning_rate
    if args.num_negatives is not None:
        _params.num_negatives = args.num_negatives
        num_negatives = args.num_negatives
        import newDataLoader as _ndl
        _ndl.num_negatives = args.num_negatives
    if args.alignment_loss_type is not None:
        _params.alignment_loss_type = args.alignment_loss_type.lower()
        alignment_loss_type = _params.alignment_loss_type
    if args.ctc_weight is not None:
        _params.ctc_weight = args.ctc_weight
        ctc_weight = args.ctc_weight
    if args.d3tw_weight is not None:
        _params.d3tw_weight = args.d3tw_weight
        d3tw_weight = args.d3tw_weight
    if args.contrastive_ctc_loss_type is not None:
        _params.contrastive_ctc_loss_type = args.contrastive_ctc_loss_type.lower()
        contrastive_ctc_loss_type = _params.contrastive_ctc_loss_type
    if args.contrastive_ctc_tau is not None:
        _params.contrastive_ctc_tau = args.contrastive_ctc_tau
        contrastive_ctc_tau = args.contrastive_ctc_tau
    if args.contrastive_ctc_margin is not None:
        _params.contrastive_ctc_margin = args.contrastive_ctc_margin
        contrastive_ctc_margin = args.contrastive_ctc_margin

    if debug_wandb:
        init_wandb(job_id)

    stride = max(1, int(window_size * stride_ratio))
    print(f"Using window_size={window_size}, stride={stride} ({int((1-stride_ratio)*100)}% overlap)")

    # Resolve dataset path & training schedule. Precedence:
    #   --data_dir > --finetune (finetune_data_dir) > Parameters.py defaults
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
        _train_dl, _valid_dl, _test_dl = build_dataloaders(resolved_data_dir)
    else:
        _train_dl, _valid_dl, _test_dl = train_dataloader, valid_dataloader, test_dataloader

    resume_ckpt = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=device)

    textEmbedding = None
    if _needs_d3tw():
        # Pick the text embedder via Parameters.text_embedder_type
        # ('char' = learned frozen table, 'fasttext' = facebook/fasttext-ar-vectors).
        textEmbedding = build_text_embedder(embedding_dim=vector_size)
        textEmbedding = textEmbedding.to(device)
        for p in textEmbedding.parameters():
            p.requires_grad_(False)
        textEmbedding.eval()
        assert not any(p.requires_grad for p in textEmbedding.parameters()), \
            "Text branch must be fully frozen — some parameters still have requires_grad=True"
        print(f"[TEXT EMBED] type={text_embedder_type}  out_dim={vector_size}  (frozen)", flush=True)
    else:
        print("[TEXT EMBED] skipped for CTC-only objective", flush=True)

    imageEmbedding = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=(lang.lower() == "arabic"),
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        dropout=model_dropout,
    )

    # --resume restores model + optimizer + scheduler + epoch inside Train().
    # --pretrained_weights only loads model weights here (fresh optimizer/epoch).
    if args.resume is not None and args.pretrained_weights is not None:
        raise SystemExit("Pass either --resume or --pretrained_weights, not both.")

    if args.pretrained_weights is not None:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        loaded = torch.load(args.pretrained_weights, map_location=device)
        imageEmbedding.load_state_dict(_extract_model_state(loaded))
        print("Pretrained weights loaded successfully.")

    assert any(p.requires_grad for p in imageEmbedding.parameters()), \
        "Image branch has no trainable parameters — check EmbeddingModel construction"

    if show_gradients:
        for param in imageEmbedding.parameters():
            param.register_hook(check_grad)

    ctc_vocab = None
    ctc_head = None
    if _needs_ctc():
        if resume_ckpt is not None and resume_ckpt.get("ctc_vocab") is not None:
            ctc_vocab = CTCVocabulary.from_dict(resume_ckpt["ctc_vocab"])
        else:
            vocab_texts = _collect_training_texts(_train_dl, resolved_data_dir)
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

    criterion    = Loss_choice() if _needs_d3tw() else None
    _model_cfg   = _make_model_config(window_size, stride, stride_ratio, ctc_vocab)

    print(f"\n=== Architecture: CRNN (ResNet34 + BiLSTM) ===")
    print(f"[OPT 1] job_id: {job_id}")
    print(f"[OPT 2] BiLSTM Context: {use_bilstm} (layers={bilstm_layers})")
    print(f"[OPT 3] Sliding Window Overlap: stride_ratio={stride_ratio}")
    print(f"[OPT 4] In-Batch Negatives: num_negatives={num_negatives}")
    print(f"[OPT 5] alignment_loss_type={alignment_loss_type}")
    if _needs_ctc():
        print(f"[OPT 6] CTC: weight={ctc_weight} loss={contrastive_ctc_loss_type} tau={contrastive_ctc_tau} margin={contrastive_ctc_margin}")
    if _needs_d3tw():
        print(f"[OPT 7] D3TW: weight={d3tw_weight}")
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

    loss_lst = Train(
        imageEmbedding,
        textEmbedding,
        _train_dl,
        _valid_dl,
        criterion,
        ctc_head=ctc_head,
        ctc_vocab=ctc_vocab,
        lr=run_lr,
        total_epochs=run_epochs,
        resume_path=args.resume,
        model_config=_model_cfg,
    )

    if debug_wandb:
        wandb.finish()

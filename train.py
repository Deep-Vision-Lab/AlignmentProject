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
                    help='Path to a .pth weights file to load before training (for finetuning).')
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


def save_model_weights(model, epoch, job_id):
    weights_dir = os.path.join(os.path.dirname(__file__), "Weights", job_id)
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"model_latest.pth")
    torch.save(model.state_dict(), weights_path)


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


def compute_batch_loss(imageEmbed, textEmbed,
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
        def _similarities_at_scale(ws, sr):
            s = max(1, int(ws * sr))
            with _autocast():
                img_emb_s = imageEmbed.forward_at_scale(images, ws, s,
                                                        show_dims=False, timer=timer)
            # Cast back to fp32 for stable similarity + DTW
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
            print(f"[Multi-Scale DTW]  "
                  f"macro_pos={loss_dict['macro_cost_pos']:.2f}  macro_neg={loss_dict['macro_cost_neg']:.2f}  "
                  f"micro_pos={loss_dict['micro_cost_pos']:.2f}  micro_neg={loss_dict['micro_cost_neg']:.2f}  "
                  f"L_macro={loss_dict['loss_macro']:.4f}  L_micro={loss_dict['loss_micro']:.4f}  "
                  f"loss={total_loss.item():.4f}", flush=True)

    else:
        timer.start('1_image_embedding')
        with _autocast():
            img_emb = imageEmbed(images, show_dims=False, timer=timer)
        img_emb = img_emb.float()
        timer.stop('1_image_embedding')

        timer.start('3b_norm_img')
        norm_img = normalize_func(img_emb)
        timer.stop('3b_norm_img')

        timer.start('4_positive_similarity')
        sim_pos = torch.einsum('bsv,btv->bst', norm_pos_text, norm_img)
        timer.stop('4_positive_similarity')

        timer.start('5_negative_similarities')
        sim_neg_all = torch.einsum('bktv,bsv->bkts', stacked_neg, norm_img)
        timer.stop('5_negative_similarities')

        timer.start('6_loss_computation')
        total_loss, loss_dict = criterion(sim_pos, sim_neg_all)
        timer.stop('6_loss_computation')

        sim_pos_for_viz = sim_pos

        if batch_idx % 10 == 0:
            print(f"[DTW-NLL] pos={loss_dict['cost_pos']:.2f}  neg={loss_dict['cost_neg']:.2f}  "
                  f"norm_pos={loss_dict['norm_pos']:.4f}  norm_neg={loss_dict['norm_neg']:.4f}  "
                  f"gap={loss_dict['gap']:.4f}  loss={total_loss.item():.4f}", flush=True)

    # Debugging: Save visualizations for the first batch every 10 epochs
    if debug and batch_idx == 0 and epoch % 10 == 0:
        save_debug_visualizations(
            imageEmbed,
            pos_texts,
            images,
            sim_pos_for_viz,
            epoch,
            job_id=job_id
        )
        save_model_weights(imageEmbed, epoch, job_id)

    return total_loss


def Train(imageEmbedding, textEmbedding, trainLoader, validLoader, criterion):
    """Train the image-text alignment model (CRNN: ResNet34 + BiLSTM)."""
    imageEmbedding.train()
    textEmbedding.eval()

    # Differential learning rates: low LR for pre-trained CNN, higher for BiLSTM
    cnn_params = list(imageEmbedding.cnn_encoder.parameters())
    cnn_param_ids = set(id(p) for p in cnn_params)
    other_params = [p for p in imageEmbedding.parameters() if id(p) not in cnn_param_ids]

    optimizer = optim.Adam(cnn_params + other_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # GradScaler only does anything when AMP is on and dtype is fp16
    # (bf16 has the same dynamic range as fp32 and doesn't need scaling).
    use_scaler = USE_AMP and AMP_DTYPE == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    loss_lst = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        if hasattr(criterion, 'gamma'):
            criterion.gamma = contrastive_soft_dtw_gamma
            print(f"Epoch {epoch+1} - Soft-DTW gamma: {contrastive_soft_dtw_gamma:.6f}", flush=True)

        imageEmbedding.train()
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
                epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                debug=debug, timer=global_timer
            )

            train_loss += loss.item()

            global_timer.start('9_backward')
            scaler.scale(loss).backward()
            global_timer.stop('9_backward')

            global_timer.start('10_optimizer_step')
            scaler.step(optimizer)
            scaler.update()
            global_timer.stop('10_optimizer_step')

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}", flush=True)

        train_loss = train_loss / len(trainLoader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}', flush=True)

        global_timer.print_summary(f"Epoch {epoch+1} Timing Summary (Training)")

        # Validation phase
        imageEmbedding.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, pos_texts, neg_texts) in enumerate(validLoader):
                images = images.to(device, non_blocking=True)

                loss = compute_batch_loss(
                    imageEmbedding, textEmbedding,
                    images, pos_texts, neg_texts,
                    criterion=criterion
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

        scheduler.step(val_loss)

        if debug_wandb:
            update_wandb(train_loss, val_loss)
            if epoch % 10 == 0:
                upload_artifacts_to_wandb(job_id, epoch)
        loss_lst.append(train_loss)

    return loss_lst


if __name__ == '__main__':
    if debug_wandb:
        init_wandb(job_id)

    stride = max(1, int(window_size * stride_ratio))
    print(f"Using window_size={window_size}, stride={stride} ({int((1-stride_ratio)*100)}% overlap)")

    if args.data_dir is not None:
        print(f"Loading dataset from custom path: {args.data_dir}")
        _train_dl, _valid_dl, _test_dl = build_dataloaders(args.data_dir)
    else:
        _train_dl, _valid_dl, _test_dl = train_dataloader, valid_dataloader, test_dataloader

    textEmbedding = TextEmbedding(embedding_dim=vector_size)
    textEmbedding = textEmbedding.to(device)
    textEmbedding.eval()

    imageEmbedding = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout,
    )

    if args.pretrained_weights is not None:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        imageEmbedding.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully.")

    if show_gradients:
        for param in imageEmbedding.parameters():
            param.register_hook(check_grad)

    criterion = Loss_choice()

    print(f"\n=== Architecture: CRNN (ResNet34 + BiLSTM) ===")
    print(f"[OPT 1] job_id: {job_id}")
    print(f"[OPT 2] BiLSTM Context: {use_bilstm} (layers={bilstm_layers})")
    print(f"[OPT 3] Sliding Window Overlap: stride_ratio={stride_ratio}")
    print(f"[OPT 4] In-Batch Negatives: num_negatives={num_negatives}")
    if multi_scale_enabled:
        print(f"[OPT 5] Multi-Scale Alignment: windows={multi_scale_window_sizes}, alpha={multi_scale_alpha}")
    print(f"[SPEED] AMP={USE_AMP} (dtype={AMP_DTYPE})  Profiling={ENABLE_PROFILING}")
    if args.pretrained_weights:
        print(f"[FINETUNE] Pretrained weights: {args.pretrained_weights}")
    if args.data_dir:
        print(f"[FINETUNE] Dataset: {args.data_dir}")
    print(f"================================================\n")

    loss_lst = Train(
        imageEmbedding,
        textEmbedding,
        _train_dl,
        _valid_dl,
        criterion
    )

    if debug_wandb:
        wandb.finish()

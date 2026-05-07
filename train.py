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
args = parser.parse_args()
job_id = args.job_id

from wandb_config import init_wandb, update_wandb, upload_artifacts_to_wandb

warnings.filterwarnings("ignore")


# ============================================================================
# GPU TIMING PROFILER - Measures execution time of each phase
# ============================================================================
class GPUTimer:
    """
    GPU-aware timer that uses CUDA events for accurate GPU timing.
    Falls back to CPU timing if CUDA is not available.
    
    Usage:
        timer = GPUTimer(enabled=True)
        timer.start('phase_name')
        # ... do work ...
        timer.stop('phase_name')
        timer.print_summary()
    """
    def __init__(self, enabled=True, device='cuda'):
        self.enabled = enabled
        self.device = device
        self.use_cuda = torch.cuda.is_available() and 'cuda' in device
        self.timings = {}  # {name: [durations]}
        self.starts = {}   # {name: start_event or start_time}
        
    def start(self, name):
        if not self.enabled:
            return
        if self.use_cuda:
            # Use CUDA events for accurate GPU timing
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
            torch.cuda.synchronize()  # Wait for GPU to finish
            duration_ms = self.starts[name].elapsed_time(end_event)  # milliseconds
        else:
            duration_ms = (time.time() - self.starts[name]) * 1000  # convert to ms
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_ms)
        del self.starts[name]
    
    def get_avg(self, name):
        if name in self.timings and len(self.timings[name]) > 0:
            return sum(self.timings[name]) / len(self.timings[name])
        return 0.0
    
    def get_total(self, name):
        if name in self.timings:
            return sum(self.timings[name])
        return 0.0
    
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


# Global timer instance (set enabled=True to profile, False to disable overhead)
global_timer = GPUTimer(enabled=True)  # Set to True to enable profiling    

def interpolate_NW_matrix(NWTensor, target_shape):
    if NWTensor.dim() == 2:
        squeezed_NW_matrix = NWTensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif NWTensor.dim() == 3:
        squeezed_NW_matrix = NWTensor.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    else:
        raise ValueError(f"Unexpected NW matrix shape: {NWTensor.shape}")
    
    interpolated_NW_matrix = F.interpolate(squeezed_NW_matrix, size=target_shape, mode='nearest')
    squeezed_interpolated_NW_matrix = interpolated_NW_matrix.squeeze(1)  # Remove channel dim
    
    del NWTensor, squeezed_NW_matrix, interpolated_NW_matrix
    
    return squeezed_interpolated_NW_matrix




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

# second line of parameters is for saving visualizations during debugging 
def compute_batch_loss(imageEmbed, textEmbed, 
                       images, pos_texts, neg_texts,
                       criterion,
                       epoch=0, batch_idx=0, dataloader_length=0, debug=False, timer=None):
    """
    Compute batch loss for text-image alignment with multiple in-batch negatives.
    
    Architecture: CNN + BiLSTM for images, simple nn.Embedding for text.
    - Image embeddings have context (BiLSTM sees neighboring strokes)
    - Text embeddings are context-free (same letter = same embedding)
    
    For each positive pair (text_i, img_i), we compute:
    - Positive: sim(text_i, img_i) → DTW cost should be LOW
    - Negative (wrong img): sim(text_i, img_j) → DTW cost should be HIGH  (x num_negatives)
    - Negative (wrong txt): sim(text_j, img_i) → DTW cost should be HIGH  (x num_negatives)
    
    VRAM Efficiency: All images are embedded ONCE via CNN+BiLSTM. Negative image
    embeddings are gathered by index (no extra CNN forward pass).
    """
    # Use global timer if none provided
    if timer is None:
        timer = global_timer

    # ==================== PHASE 2: Text Embedding ====================
    timer.start('2_text_embedding')
    pos_text_emb = textEmbed(list(pos_texts))  # [B, text_len, vec_size]
    timer.stop('2_text_embedding')

    # ==================== PHASE 2b: Negative Text Embeddings ====================
    timer.start('2b_neg_text_embedding')
    neg_text_embs = []   # list of [B, neg_text_len_k, vec_size]
    for k in range(num_negatives):
        kth_neg_texts = [neg_texts[i][k] for i in range(len(neg_texts))]
        neg_text_emb_k = textEmbed(kth_neg_texts)            # [B, neg_text_len_k, vec_size]
        norm_neg_text_k = normalize_func(neg_text_emb_k)     # [B, neg_text_len_k, vec_size]
        neg_text_embs.append(norm_neg_text_k)
        del kth_neg_texts, neg_text_emb_k
    # Pad negatives to uniform length then stack: [B, K, max_neg_len, vec_size]
    max_neg_len = max(t.size(1) for t in neg_text_embs)
    for k in range(len(neg_text_embs)):
        pad_amount = max_neg_len - neg_text_embs[k].size(1)
        if pad_amount > 0:
            neg_text_embs[k] = F.pad(neg_text_embs[k], (0, 0, 0, pad_amount))
    stacked_neg = torch.stack(neg_text_embs, dim=1)  # [B, K, max_neg_len, vec_size]
    del neg_text_embs
    timer.stop('2b_neg_text_embedding')

    # ==================== PHASE 3: Normalization (text) ====================
    timer.start('3_normalization')
    norm_pos_text = normalize_func(pos_text_emb)    # [B, text_len, vec_size]
    timer.stop('3_normalization')

    # ====================================================================
    # MULTI-SCALE vs SINGLE-SCALE BRANCH
    # ====================================================================
    if multi_scale_enabled:
        # ---------- helper: embed at one scale and build similarity matrices ----------
        def _similarities_at_scale(ws, sr):
            s = max(1, int(ws * sr))
            img_emb_s = imageEmbed.forward_at_scale(images, ws, s,
                                                     show_dims=False, timer=timer)
            norm_img_s = normalize_func(img_emb_s)
            sim_pos_s = torch.einsum('bsv,btv->bst', norm_pos_text, norm_img_s)
            sim_neg_s = torch.einsum('bktv,bsv->bkts', stacked_neg, norm_img_s)
            del img_emb_s, norm_img_s
            return sim_pos_s, sim_neg_s

        macro_ws, micro_ws = multi_scale_window_sizes

        timer.start('4_macro_scale')
        sim_pos_macro, sim_neg_macro = _similarities_at_scale(macro_ws, stride_ratio)
        timer.stop('4_macro_scale')

        # Free cached GPU blocks before the heavier micro-scale pass
        torch.cuda.empty_cache()

        timer.start('4_micro_scale')
        sim_pos_micro, sim_neg_micro = _similarities_at_scale(micro_ws, stride_ratio)
        timer.stop('4_micro_scale')

        # --- loss (MultiScaleContrastiveSoftDTW) ---
        timer.start('5_loss_computation')
        total_loss, loss_dict = criterion(
            sim_pos_macro, sim_neg_macro,
            sim_pos_micro, sim_neg_micro,
        )
        timer.stop('5_loss_computation')

        # For debug visualizations, use macro-scale positive similarity
        sim_pos_for_viz = sim_pos_macro

        # Cleanup
        del sim_pos_macro, sim_neg_macro, sim_pos_micro, sim_neg_micro
        del stacked_neg
        torch.cuda.empty_cache()

        loss_value = total_loss.item()

        # Log cost breakdown for diagnosing collapse (multi-scale)
        if batch_idx % 10 == 0:
            print(f"[Multi-Scale DTW]  "
                  f"macro_pos={loss_dict['macro_cost_pos']:.2f}  macro_neg={loss_dict['macro_cost_neg']:.2f}  "
                  f"micro_pos={loss_dict['micro_cost_pos']:.2f}  micro_neg={loss_dict['micro_cost_neg']:.2f}  "
                  f"L_macro={loss_dict['loss_macro']:.4f}  L_micro={loss_dict['loss_micro']:.4f}  "
                  f"loss={loss_value:.4f}", flush=True)

    else:
        # ==================== SINGLE-SCALE (original path) ====================
        timer.start('1_image_embedding')
        img_emb = imageEmbed(images, show_dims=False, timer=timer)  # [B, seq_len, vec_size]
        timer.stop('1_image_embedding')

        timer.start('3b_norm_img')
        norm_img = normalize_func(img_emb)      # [B, seq_len, vec_size]
        timer.stop('3b_norm_img')

        timer.start('4_positive_similarity')
        sim_pos = torch.einsum('bsv,btv->bst', norm_pos_text, norm_img)
        timer.stop('4_positive_similarity')

        timer.start('5_negative_similarities')
        sim_neg_all = torch.einsum('bktv,bsv->bkts', stacked_neg, norm_img)
        total_loss, loss_dict = criterion(sim_pos, sim_neg_all)
        del stacked_neg, sim_neg_all
        torch.cuda.empty_cache()
        timer.stop('5_negative_similarities')

        timer.start('6_loss_computation')
        loss_value = total_loss.item()
        timer.stop('6_loss_computation')

        sim_pos_for_viz = sim_pos

        # Log cost breakdown for diagnosing collapse
        if batch_idx % 10 == 0:
            print(f"[DTW-NLL] pos={loss_dict['cost_pos']:.2f}  neg={loss_dict['cost_neg']:.2f}  "
                  f"norm_pos={loss_dict['norm_pos']:.4f}  norm_neg={loss_dict['norm_neg']:.4f}  "
                  f"gap={loss_dict['gap']:.4f}  loss={loss_value:.4f}", flush=True)

        del img_emb, norm_img, sim_pos
        torch.cuda.empty_cache()

    ######################################################################################################################################
    # Debugging: Save visualizations for the first batch every 10 epochs
    if debug and batch_idx == 0 and epoch % 10 == 0:
        # Use positive similarity for visualization
        save_debug_visualizations(
            imageEmbed,
            pos_texts,
            images, 
            sim_pos_for_viz,
            epoch,
            job_id=job_id
        )

        # Save model weights
        save_model_weights(imageEmbed, epoch, job_id)
    ######################################################################################################################################

    # Cleanup to free VRAM
    del pos_text_emb, norm_pos_text, sim_pos_for_viz
    torch.cuda.empty_cache()

    return total_loss



def Train(imageEmbedding, textEmbedding, trainLoader, validLoader, criterion):
    """
    Train the image-text alignment model.
    
    Architecture: CNN + BiLSTM for images, simple nn.Embedding for text.
    No Transformer - direct cosine similarity between embeddings.
    """
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
    
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Update gamma for Soft-DTW annealing
        if hasattr(criterion, 'gamma'):
            criterion.gamma = contrastive_soft_dtw_gamma
            print(f"Epoch {epoch+1} - Soft-DTW gamma: {contrastive_soft_dtw_gamma:.6f}", flush=True)
        
        # Ensure model is in training mode at start of each epoch
        imageEmbedding.train()
        
        # Reset timer for this epoch
        global_timer.reset()
            
        train_loss = 0.0

        for batch_idx, (images, pos_texts, neg_texts) in enumerate(trainLoader):
            # Timing: Data transfer to GPU
            # Note: images are already on GPU from collate, but ensure non_blocking
            global_timer.start('0_data_to_gpu')
            images = images.to(device, non_blocking=True)
            pos_texts = list(pos_texts)
            neg_texts = list(neg_texts)
            global_timer.stop('0_data_to_gpu')
            
            optimizer.zero_grad() 
  
            # Compute loss with in-batch negatives
            loss = compute_batch_loss(
                imageEmbedding, textEmbedding, 
                images, pos_texts, neg_texts,
                criterion,
                epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                debug=debug, timer=global_timer
            )
            
            train_loss += loss.item()

            # Backpropagation and optimizer step
            global_timer.start('9_backward')
            loss.backward()
            global_timer.stop('9_backward')
            
            global_timer.start('10_optimizer_step')
            optimizer.step()
            global_timer.stop('10_optimizer_step')


            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}", flush=True)
            # Optionally print gradients for debugging
            if show_gradients:
                if images.grad is not None:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1} - images gradient: {images.grad.sum():.4f}", flush=True)

            # Final cleanup
            del images
            del loss
            torch.cuda.empty_cache()

        train_loss = train_loss / len(trainLoader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}', flush=True)
        
        # Print timing summary for this epoch (only if profiling is enabled)
        global_timer.print_summary(f"Epoch {epoch+1} Timing Summary (Training)")
        
        # Validation phase
        imageEmbedding.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, pos_texts, neg_texts) in enumerate(validLoader):
                # Ensure all data is on correct device
                images = images.to(device)
                pos_texts = list(pos_texts)
                neg_texts = list(neg_texts)
                
                # Compute loss using shared function
                loss = compute_batch_loss(
                    imageEmbedding, textEmbedding,
                    images, pos_texts, neg_texts,
                    criterion=criterion
                )
                
                val_loss += loss.item()

                # Final cleanup
                del images
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(validLoader)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}', flush=True)
        
        # Calculate and print epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_minutes = int(epoch_duration // 60)
        epoch_seconds = epoch_duration % 60
        print(f'Epoch {epoch+1} completed in {epoch_minutes}m {epoch_seconds:.2f}s', flush=True)
        print('=' * 60, flush=True)
        
        # Step the LR scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Log train and validation losses and accuracies to wandb
        if debug_wandb:
            update_wandb(
                train_loss, 
                val_loss
            )
            # Upload artifacts every 10 epochs
            if epoch % 10 == 0:
                upload_artifacts_to_wandb(job_id, epoch)
        loss_lst.append(train_loss)

    return loss_lst



if __name__ == '__main__':
    if debug_wandb:
        init_wandb(job_id)

    # Calculate stride from window_size and stride_ratio (OPTIMIZATION 4: Overlap)
    stride = max(1, int(window_size * stride_ratio))
    print(f"Using window_size={window_size}, stride={stride} ({int((1-stride_ratio)*100)}% overlap)")

    # Initialize TextEmbedding FIRST (needed for space token in image embedding)
    # Text embeddings are context-free: same letter = same embedding (Label-Aware design)
    textEmbedding = TextEmbedding(embedding_dim=vector_size)
    textEmbedding = textEmbedding.to(device)

    imageEmbedding = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout,
    )

    if show_gradients:
        for param in imageEmbedding.parameters():
            param.register_hook(check_grad)

    
    criterion = Loss_choice()
    
    # Log architecture
    print(f"\n=== Architecture: CRNN (ResNet34 + BiLSTM) ===")
    print(f"[OPT 1] job_id: {job_id}")
    print(f"[OPT 2] BiLSTM Context: {use_bilstm} (layers={bilstm_layers})")
    print(f"[OPT 3] Sliding Window Overlap: stride_ratio={stride_ratio}")
    print(f"[OPT 4] In-Batch Negatives: num_negatives={num_negatives}")
    if multi_scale_enabled:
        print(f"[OPT 5] Multi-Scale Alignment: windows={multi_scale_window_sizes}, alpha={multi_scale_alpha}")
    print(f"================================================\n")
    
    # try:
    loss_lst = Train(
        imageEmbedding,
        textEmbedding,
        train_dataloader,
        valid_dataloader,
        criterion
    )
    
    if debug_wandb:
        wandb.finish()
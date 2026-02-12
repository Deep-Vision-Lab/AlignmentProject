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



def compute_accuracy(pred_path, target_path, threshold=0.5):
    # Binarize the paths
    pred_binary = (pred_path > threshold).float()
    target_binary = (target_path > threshold).float()
    
    # Calculate accuracy as the percentage of matching positions
    correct = (pred_binary == target_binary).float()
    accuracy = correct.sum().item() / correct.numel()
    
    return accuracy


def save_model_weights(model, epoch, job_id):
    weights_dir = os.path.join(os.path.dirname(__file__), "Weights", loss_type, job_id)
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), weights_path)

def check_grad(grad):
    if grad is None:
        print("Gradient is None!")
    elif torch.isnan(grad).any():
        print("Gradient became NaN!")
    else:
        print(f"Gradient flowing... Sum: {grad.sum().item():.5f}")

# second line of parameters is for saving visualizations during debugging 
def compute_batch_loss(imageEmbed, textEmbed, text1, text2, image1, image2, txt1_txt2_similar_GT, 
                       criterion,
                       epoch=0, batch_idx=0, dataloader_length=0, debug=False, timer=None):
    """
    Compute batch loss for text-image alignment.
    
    Architecture: CNN + BiLSTM for images, simple nn.Embedding for text.
    - Image embeddings have context (BiLSTM sees neighboring strokes)
    - Text embeddings are context-free (same letter = same embedding)
    
    This "Label-Aware" design ensures repeated letters (e.g., "A B A") 
    have identical embeddings, allowing Soft-DTW to correctly align 
    image patches to ALL matching text positions.
    """
    # Use global timer if none provided
    if timer is None:
        timer = global_timer

    # ==================== PHASE 1: Image Embedding (CNN + BiLSTM) ====================
    timer.start('1_image_embedding_total')
    tokens_a, tokens_b = imageEmbed(image1, image2, show_dims=False, timer=timer)
    timer.stop('1_image_embedding_total')
    
    
    # ==================== PHASE 2: Text Embedding ====================
    timer.start('2_text_embedding')
    embedded_text1 = textEmbed(text1)  # Shape: (batch_size, seq_len, embedding_dim)
    embedded_text2 = textEmbed(text2)  # Shape: (batch_size, seq_len, embedding_dim)
    timer.stop('2_text_embedding')

    # ==================== PHASE 3: Normalization ====================
    timer.start('3_normalization')
    normalized_text1 = normalize_func(embedded_text1)  # Shape: (batch_size, seq_len, embedding_dim)
    normalized_text2 = normalize_func(embedded_text2)  # Shape: (batch_size, seq_len, embedding_dim)
    normalized_tokens_a = normalize_func(tokens_a)  # Shape: (batch_size, seq_len, vector_size)
    normalized_tokens_b = normalize_func(tokens_b)
    timer.stop('3_normalization')
                                      
    # ==================== PHASE 4: Similarity Matrix (BMM) ====================
    timer.start('4_bmm_img_txt_similarity')
    similarity_1_1 = -1 * torch.bmm(normalized_text1, normalized_tokens_a.transpose(1, 2)) # Match
    similarity_2_2 = -1 * torch.bmm(normalized_text2, normalized_tokens_b.transpose(1, 2)) # Match
    similarity_1_2 = -1 * torch.bmm(normalized_text1, normalized_tokens_b.transpose(1, 2)) # Mismatch
    similarity_2_1 = -1 * torch.bmm(normalized_text2, normalized_tokens_a.transpose(1, 2)) # Mismatch
    timer.stop('4_bmm_img_txt_similarity')
    
#---------------------------------------------------------------------------------------------------------
    
    # ==================== PHASE 7: Loss Computation ====================
    timer.start('5_loss_computation')
    loss, loss_dict = criterion(
        similarity_1_1,
        similarity_2_2,
        similarity_1_2,
        similarity_2_1
    )
    loss_value = loss_dict['total']
    timer.stop('5_loss_computation')
    
    ######################################################################################################################################
    # Debugging: Save visualizations for the last batch every 10 epochs
    if debug and batch_idx == 0 and epoch % 10 == 0:
        debug_imgText1 = similarity_1_1
        debug_imgText2 = similarity_2_2

        # indexing only the first 2 samples in the batch for visualization
        text1_subset = [text1[i] for i in range(min(2, len(text1)))]
        text2_subset = [text2[i] for i in range(min(2, len(text2)))]
        image1_subset = image1[:min(2, image1.size(0))]
        image2_subset = image2[:min(2, image2.size(0))]
        TextSimilarGT_subset = txt1_txt2_similar_GT[:min(2, txt1_txt2_similar_GT.size(0))]
        similar_TxtImg1 = similarity_1_1[:min(2, similarity_1_1.size(0))]
        similar_TxtImg2 = similarity_2_2[:min(2, similarity_2_2.size(0))]
        save_debug_visualizations(
            imageEmbed, 
            text1_subset, text2_subset, image1_subset, image2_subset,
            similar_TxtImg1, similar_TxtImg2,
            epoch,
            job_id
        )
        
        del debug_imgText1, debug_imgText2
        # Save model weights
        save_model_weights(imageEmbed, epoch, job_id)

    ######################################################################################################################################

    # Delete tensors to free memory
    del tokens_a, tokens_b
    torch.cuda.empty_cache()
    
    return loss, loss_value



def Train(imageEmbedding, textEmbedding, trainLoader, validLoader, criterion):
    """
    Train the image-text alignment model.
    
    Architecture: CNN + BiLSTM for images, simple nn.Embedding for text.
    No Transformer - direct cosine similarity between embeddings.
    """
    imageEmbedding.train()
    textEmbedding.eval()
    
    # Collect all trainable parameters (only image embedding is trained)
    params_to_train = list(imageEmbedding.parameters())
    
    optimizer = optim.Adam(params_to_train, lr=learning_rate)
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Ensure model is in training mode at start of each epoch
        imageEmbedding.train()
        
        # Reset timer for this epoch
        global_timer.reset()
            
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (image1, image2, textSimilar, text1, text2) in enumerate(trainLoader):
            # Timing: Data transfer to GPU
            global_timer.start('0_data_to_gpu')
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            textSimilar = textSimilar.to(device, non_blocking=True)
            text1 = list(text1)
            text2 = list(text2)
            global_timer.stop('0_data_to_gpu')
            
            optimizer.zero_grad()
  
            # Compute loss using shared function (timing is inside)
            path_loss, loss_value = compute_batch_loss(
                imageEmbedding, textEmbedding, text1, text2, image1, image2, 
                textSimilar, criterion,
                epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                debug=debug, timer=global_timer
            )
            

            train_loss += loss_value

            # Backpropagation and optimizer step
            global_timer.start('9_backward')
            path_loss.backward()
            global_timer.stop('9_backward')
            
            global_timer.start('10_optimizer_step')
            optimizer.step()
            global_timer.stop('10_optimizer_step')

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {train_loss / (batch_idx + 1):.4f}", flush=True)
            # Optionally print gradients for debugging
            if show_gradients:
                if image1.grad is not None and image2.grad is not None:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1} - image1 gradient: {image1.grad.sum():.4f}, image2 gradient: {image2.grad.sum():.4f}", flush=True)

            # Final cleanup
            del path_loss
            del image1, image2
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1} completed. Average Loss: {train_loss / len(trainLoader):.4f}", flush=True)
        train_loss = train_loss / len(trainLoader)
        train_accuracy = train_accuracy / len(trainLoader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}', flush=True)
        
        # Print timing summary for this epoch (only if profiling is enabled)
        global_timer.print_summary(f"Epoch {epoch+1} Timing Summary (Training)")
        
        # Validation phase
        imageEmbedding.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for batch_idx, (image1, image2, textSimilar, text1, text2) in enumerate(validLoader):
                # Ensure all data is on correct device
                image1 = image1.to(device)
                image2 = image2.to(device)
                textSimilar = textSimilar.to(device)
                
                # Compute loss using shared function
                _, loss_value = compute_batch_loss(
                    imageEmbedding, textEmbedding,
                    text1, text2, image1, image2, 
                    textSimilar, criterion
                )
                
                val_loss += loss_value

                # Final cleanup
                del image1, image2
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(validLoader)
        val_accuracy = val_accuracy / len(validLoader)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}', flush=True)
        
        # Calculate and print epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_minutes = int(epoch_duration // 60)
        epoch_seconds = epoch_duration % 60
        print(f'Epoch {epoch+1} completed in {epoch_minutes}m {epoch_seconds:.2f}s', flush=True)
        print('=' * 60, flush=True)
        
        # Log train and validation losses and accuracies to wandb
        if debug_wandb:
            update_wandb(
                train_loss, 
                val_loss, 
                train_accuracy, 
                val_accuracy
            )
            # Upload artifacts every 10 epochs
            if epoch % 10 == 0:
                upload_artifacts_to_wandb(job_id, epoch)
        loss_lst.append(train_loss)

    return loss_lst



if __name__ == '__main__':
    if debug_wandb:
        init_wandb()

    # Calculate stride from window_size and stride_ratio (OPTIMIZATION 4: Overlap)
    stride = max(1, int(window_size * stride_ratio))
    print(f"Using window_size={window_size}, stride={stride} ({int((1-stride_ratio)*100)}% overlap)")

    # Initialize TextEmbedding FIRST (needed for space token in image embedding)
    # Text embeddings are context-free: same letter = same embedding (Label-Aware design)
    textEmbedding = TextEmbedding(embedding_dim=vector_size, include_spaces=include_spaces)
    textEmbedding = textEmbedding.to(device)
    # create the space token vector (detach to avoid graph retention across batches)
    space_vector = textEmbedding(' ').detach()

    imageEmbedding = EmbeddingModel(
        window_size=window_size,
        stride=stride,  # Now uses calculated overlap stride
        vector_size=vector_size,
        device=device,
        # OPTIMIZATION 1 & 3: Enable BiLSTM and Positional Encoding
        use_bilstm=use_bilstm,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_type=positional_encoding_type,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout,
        # OPTIMIZATION 5: Space Gate for black patch detection
        use_space_gate=use_space_gate,
        space_threshold=space_threshold,
        space_vector=space_vector
    )

    if show_gradients:
        for param in imageEmbedding.parameters():
            param.register_hook(check_grad)

    
    criterion = Loss_choice(loss_type)
    
    # Log architecture optimizations
    print(f"\n=== Architecture: CNN + BiLSTM (No Transformer) ===")
    print(f"[OPT 1] Positional Encoding: {use_positional_encoding} ({positional_encoding_type})")
    print(f"[OPT 2] Loss Type: {loss_type}")
    print(f"[OPT 3] BiLSTM Context: {use_bilstm} (layers={bilstm_layers})")
    print(f"[OPT 4] Sliding Window Overlap: stride_ratio={stride_ratio}")
    print(f"[OPT 5] Space Gate: {use_space_gate} (threshold={space_threshold})")
    print(f"[OPT 6] Text Spaces: include_spaces={include_spaces}")
    print(f"[OPT 7] Context-Free Text (Label-Aware): ENABLED")
    print(f"        -> Same letter = Same embedding (no context mixing)")
    if loss_type == 'ContrastiveMSE':
        print(f"[OPT 8] Contrastive Learning: ENABLED")
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
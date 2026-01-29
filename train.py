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
from embeddingModel import *
from NormalizeFuncs import *
from LossFunctionWithHelpers import *
from SimilarityTransformer import SimilarityTransformer

import os
import gc
import time
import wandb
import warnings

from wandb_config import init_wandb, update_wandb

warnings.filterwarnings("ignore")    

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


def save_model_weights(model, epoch):
    weights_dir = os.path.join(os.path.dirname(__file__), "Weights", loss_type, model_arch)
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
def compute_batch_loss(imageEmbed, textEmbed, text1, text2, image1, image2, textSimilar, 
                       criterion, similarityTransformer1, similarityTransformer2,
                       epoch=0, batch_idx=0, dataloader_length=0, debug=False):
    """
    Compute batch loss for text-image alignment.
    
    For ContrastiveSoftDTW loss type, all contrastive learning is handled inside
    the loss function - no special handling needed here.
    """

    # Image Embedding and Token Extraction    
    tokens_a, tokens_b = imageEmbed(image1, image2, show_dims=False)
    
    flip_tokens_a = torch.flip(tokens_a, dims=[-2]) # Shape: (batch_size, seq_len, vector_size)
    flip_tokens_b = torch.flip(tokens_b, dims=[-2]) # Shape: (batch_size, seq_len, vector_size)
    # -----------------------------------------------------------------------------------------
    # Text Embedding
    embedded_text1 = textEmbed(text1)  # Shape: (batch_size, seq_len, embedding_dim)
    embedded_text2 = textEmbed(text2)  # Shape: (batch_size, seq_len, embedding_dim)

    # Normalize Vetors
    normalized_text1 = F.normalize(embedded_text1, p=2, dim=-1)  # Shape: (batch_size, seq_len, embedding_dim)
    normalized_text2 = F.normalize(embedded_text2, p=2, dim=-1)  # Shape: (batch_size, seq_len, embedding_dim)
    normalized_tokens_a = F.normalize(flip_tokens_a, p=2, dim=-1)  # Shape: (batch_size, seq_len, vector_size)
    normalized_tokens_b = F.normalize(flip_tokens_b, p=2, dim=-1)
                                      
    # Compute similarity matrices using Transformer model
    img1_txt1_similarity = similarityTransformer1(normalized_text1, normalized_tokens_a)
    img2_txt2_similarity = similarityTransformer1(normalized_text2, normalized_tokens_b)

    normalized_img1_txt1_similarity = F.normalize(img1_txt1_similarity, p=2, dim=-1)
    normalized_img2_txt2_similarity = F.normalize(img2_txt2_similarity, p=2, dim=-1)
  
    # Multiplying score matrices
    finalSimilarityMatrix = torch.bmm(normalized_img1_txt1_similarity, normalized_img2_txt2_similarity.transpose(1, 2))
    
#---------------------------------------------------------------------------------------------------------
    
    # Loss computation - handle different loss types
    if loss_type == 'ContrastiveSoftDTW':
        # Contrastive Soft-DTW: InfoNCE-style contrastive learning with CUDA Soft-DTW
        # All contrastive learning is handled inside the loss function
        # We pass the embeddings and the loss computes all pairwise alignments
        loss, loss_dict = criterion(
            final_pred=finalSimilarityMatrix,
            target=textSimilar,
            img_embeddings=normalized_tokens_a,  # Image embeddings [B, N, D]
            txt_embeddings=normalized_text1       # Text embeddings [B, M, D]
        )
        loss_value = loss_dict['total']
    elif loss_type == 'MSE':
        # Simple MSE loss
        loss = criterion(finalSimilarityMatrix, textSimilar)
        loss_value = loss.item()
    else:
        # Default: try calling criterion directly
        loss = criterion(finalSimilarityMatrix, textSimilar)
        loss_value = loss.item() if hasattr(loss, 'item') else loss
    
    ######################################################################################################################################
    # Debugging: Save visualizations for the last batch every 10 epochs
    if debug and batch_idx == 0 and epoch % 10 == 0:
        debug_imgText1 = normalized_img1_txt1_similarity
        debug_imgText2 = normalized_img2_txt2_similarity

        save_debug_visualizations(
            imageEmbed, 
            text1, text2, image1, image2,
            textSimilar, finalSimilarityMatrix,
            debug_imgText1, debug_imgText2,
            epoch, batch_idx
        )
        
        del debug_imgText1, debug_imgText2
        
        # Save model weights
        save_model_weights(imageEmbed, epoch)

    ######################################################################################################################################

    # Delete tensors to free memory
    del tokens_a, tokens_b
    del flip_tokens_a, flip_tokens_b
    torch.cuda.empty_cache()
    
    return loss, loss_value, textSimilar, finalSimilarityMatrix



def Train(imageEmbedding, textEmbedding, trainLoader, validLoader, criterion,
          similarityTransformer1=None, similarityTransformer2=None):
    imageEmbedding.train()
    textEmbedding.eval()
    
    # Collect all trainable parameters
    params_to_train = list(imageEmbedding.parameters())
    if similarityTransformer1 is not None:
        similarityTransformer1.train()
        params_to_train += list(similarityTransformer1.parameters())
    if similarityTransformer2 is not None:
        similarityTransformer2.train()
        params_to_train += list(similarityTransformer2.parameters())
    
    optimizer = optim.Adam(params_to_train, lr=learning_rate)
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        # Ensure models are in training mode at start of each epoch
        imageEmbedding.train()
        if similarityTransformer1 is not None:
            similarityTransformer1.train()
        if similarityTransformer2 is not None:
            similarityTransformer2.train()
            
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (image1, image2, textSimilar, text1, text2) in enumerate(trainLoader):
            # Ensure all data is on correct device
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            textSimilar = textSimilar.to(device, non_blocking=True)
            text1 = list(text1)
            text2 = list(text2)
            optimizer.zero_grad()
            
            # Compute loss using shared function
            path_loss, loss_value, NWTextFinal, NWImageFinal = compute_batch_loss(
                imageEmbedding, textEmbedding, text1, text2, image1, image2, 
                textSimilar, criterion, 
                similarityTransformer1=similarityTransformer1, similarityTransformer2=None,
                epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                debug=debug
            )
            
            # Compute accuracy
            batch_accuracy = compute_accuracy(NWImageFinal, NWTextFinal)
            train_accuracy += batch_accuracy
            

            train_loss += loss_value

            # Backpropagation and optimizer step
            path_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {train_loss / (batch_idx + 1):.4f}", flush=True)
            # Optionally print gradients for debugging
            if show_gradients:
                if image1.grad is not None and image2.grad is not None:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1} - image1 gradient: {image1.grad.sum():.4f}, image2 gradient: {image2.grad.sum():.4f}", flush=True)

            # Final cleanup
            del path_loss
            del image1, image2
            del NWTextFinal, NWImageFinal
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1} completed. Average Loss: {train_loss / len(trainLoader):.4f}", flush=True)
        train_loss = train_loss / len(trainLoader)
        train_accuracy = train_accuracy / len(trainLoader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}', flush=True)
        
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
                _, loss_value, NWTextFinal, NWImageFinal = compute_batch_loss(
                    imageEmbedding, textEmbedding,
                    text1, text2, image1, image2, 
                    textSimilar, criterion,
                    similarityTransformer1=similarityTransformer1, similarityTransformer2=None
                )
                
                # Compute accuracy
                batch_accuracy = compute_accuracy(NWImageFinal, NWTextFinal)
                val_accuracy += batch_accuracy
                
                del NWTextFinal, NWImageFinal
                val_loss += loss_value

                # Final cleanup
                del image1, image2
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(validLoader)
        val_accuracy = val_accuracy / len(validLoader)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}', flush=True)
        
        # Log train and validation losses and accuracies to wandb
        if debug_wandb:
            update_wandb(
                train_loss, 
                val_loss, 
                train_accuracy, 
                val_accuracy
            )
        loss_lst.append(train_loss)

    return loss_lst



if __name__ == '__main__':
    if debug_wandb:
        init_wandb()

    # Calculate stride from window_size and stride_ratio (OPTIMIZATION 4: Overlap)
    stride = max(1, int(window_size * stride_ratio))
    print(f"Using window_size={window_size}, stride={stride} ({int((1-stride_ratio)*100)}% overlap)")

    imageEmbedding = EmbeddingModel(
        window_size=window_size,
        stride=stride,  # Now uses calculated overlap stride
        vector_size=vector_size,
        model_arch=model_arch,
        device=device,
        # OPTIMIZATION 1 & 3: Enable BiLSTM and Positional Encoding
        use_bilstm=use_bilstm,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_type=positional_encoding_type,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout,
        # OPTIMIZATION 5: Space Gate for black patch detection
        use_space_gate=use_space_gate,
        space_threshold=space_threshold
    )

    # Initialize TextEmbedding with space token support
    textEmbedding = TextEmbedding(embedding_dim=vector_size, include_spaces=include_spaces)

    # Initialize two separate SimilarityTransformer models for image-text similarity computation
    # The transformers take image strokes with context from previous strokes to predict which letter
    # Using different transformers allows specialized learning for different text patterns
    similarityTransformer1 = SimilarityTransformer(
        embed_dim=vector_size,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=model_dropout
    ).to(device)
    
    similarityTransformer2 = SimilarityTransformer(
        embed_dim=vector_size,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=model_dropout
    ).to(device)

    if show_gradients:
        for param in imageEmbedding.parameters():
            param.register_hook(check_grad)
        for param in similarityTransformer1.parameters():
            param.register_hook(check_grad)
        # for param in similarityTransformer2.parameters():
        #     param.register_hook(check_grad)

    
    criterion = Loss_choice(loss_type)
    
    # Log architecture optimizations
    print(f"\n=== Architecture Optimizations ===")
    print(f"[OPT 1] Positional Encoding: {use_positional_encoding} ({positional_encoding_type})")
    print(f"[OPT 2] Loss Type: {loss_type}")
    print(f"[OPT 3] BiLSTM Context: {use_bilstm} (layers={bilstm_layers})")
    print(f"[OPT 4] Sliding Window Overlap: stride_ratio={stride_ratio}")
    print(f"[OPT 5] Space Gate: {use_space_gate} (threshold={space_threshold})")
    print(f"[OPT 6] Text Spaces: include_spaces={include_spaces}")
    if loss_type == 'ContrastiveMSE':
        print(f"[OPT 7] Contrastive Learning: ENABLED")
    print(f"==================================\n")
    
    # try:
    loss_lst = Train(
        imageEmbedding,
        textEmbedding,
        train_dataloader,
        valid_dataloader,
        criterion,
        similarityTransformer1=similarityTransformer1,
        similarityTransformer2=similarityTransformer2
    )
    
    if debug_wandb:
        wandb.finish()
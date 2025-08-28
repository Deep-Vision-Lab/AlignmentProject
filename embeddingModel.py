from matplotlib import patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights, vit_b_16, ViT_B_16_Weights
import cv2
import matplotlib.pyplot as plt  
import numpy as np 


# Sliding window function to divide image into patches (subwindows)
def sliding_window(image, window_size, stride, debug_mode=False, save_dir=False):
    patches = []
    # Unfolding the image into patches of size window_size
    for i  in range(image.shape[0]):
        image_windows = []
        for j in range(0, image.shape[3] - window_size + 1, stride):
            image_i = image[i, :, :, j:j + window_size]
            image_windows.append(image_i)
            if debug_mode:
                torchvision.utils.save_image(image_i, f"{save_dir}/patch_b{i}_w{j}.png")
        patches.append(torch.stack(image_windows, dim=0))
        del image_windows
    result = torch.stack(patches, dim=0)
    del patches
    return result


# CNN model for extracting features from sliding window patches
def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1



# Transformer-based model for processing token sequences
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    
    def forward(self, tokens):
        tokens = self.pos_encoder(tokens)
        return self.transformer_encoder(tokens)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  
        return self.dropout(x)


class EmbeddingModel(nn.Module):
    def __init__(self, window_size=128, stride=64, vector_size=512, model_arch='CNN-Transformer',
                  device='cuda'):
        super(EmbeddingModel, self).__init__()
        
        self.model_arch = model_arch

        if model_arch == 'CNN-Transformer' or model_arch == 'CNN':
            self.cnn_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            cnn_feature_dim = self.cnn_encoder.fc.in_features
            self.cnn_encoder.fc = nn.Linear(cnn_feature_dim, vector_size)

        if model_arch == 'CNN-Transformer' or model_arch == 'Transformer':
            self.transformer_encoder = TransformerEncoder(d_model=vector_size, nhead=8, num_layers=6)
        
        if model_arch == 'Transformer':
            self.channel_reducer = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
            self.convert_to_vectors = nn.Sequential(
                nn.Linear(in_features=tokens_a.shape[-1], out_features=self.vector_size)
            ).to(device) 
        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
        self.model_arch = model_arch
    
    def forward(self, image_a, image_b, show_dims=False, debug=False):
       
        #################################################################################

        tokens_a = sliding_window(image_a, self.window_size, self.stride, debug_mode=debug)
        tokens_b = sliding_window(image_b, self.window_size, self.stride, debug_mode=debug)
        if show_dims: print(f"Patches_a shape: {tokens_a.shape}, Patches_b shape: {tokens_b.shape}")
        
        batches_num, windows_num, Channels, H, W = tokens_a.shape
        
        # Reshape patches to [batch_size, windows_num, Channels, H, W]
        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'CNN':        
            reshaped_tokens_a  = tokens_a.reshape(batches_num * windows_num, Channels, H, W)
            reshaped_tokens_b = tokens_b.reshape(batches_num * windows_num, Channels, H, W)
            if show_dims: print(f"Patches_a shape after reshaping: {reshaped_tokens_a.shape}, Patches_b shape after reshaping: {reshaped_tokens_b.shape}")
            del tokens_a, tokens_b

            # Pass through CNN to get feature vectors
            encoded_tokens_a = self.cnn_encoder(reshaped_tokens_a)
            encoded_tokens_b = self.cnn_encoder(reshaped_tokens_b)
            if show_dims: print(f"Tokens_a shape: {encoded_tokens_a.shape}, Tokens_b shape: {encoded_tokens_b.shape}")
            del reshaped_tokens_a, reshaped_tokens_b

            if self.model_arch == 'CNN':  
                features_vector_a = encoded_tokens_a.view(batches_num, windows_num, self.vector_size) 
                features_vector_b = encoded_tokens_b.view(batches_num, windows_num, self.vector_size) 
                if show_dims: print(f"Tokens_a shape after reshaping: {features_vector_a.shape}, Tokens_b shape after reshaping: {features_vector_b.shape}")
                del encoded_tokens_a, encoded_tokens_b

        if self.model_arch == 'Transformer':
            # Compute mean over width (W) for each row
            reduced_tokens_a = self.channel_reducer(tokens_a)  # type: ignore # [batch_size, windows_num, 1, H, W]
            reduced_tokens_b = self.channel_reducer(tokens_b)  # type: ignore # [batch_size, windows_num, 1, H, W]
            if show_dims: print(f"Patches_a shape after channel reduction: {reduced_tokens_a.shape}, Patches_b shape after channel reduction: {reduced_tokens_b.shape}")
            del tokens_a, tokens_b # type: ignore

            # Compute mean over width (W) for each row
            mean_tokens_a = reduced_tokens_a.mean(dim=2)  # [batch_size, windows_num, H, W]
            mean_tokens_b = reduced_tokens_b.mean(dim=2)  # [batch_size, windows_num, H, W]
            if show_dims: print(f"mean_a shape: {mean_tokens_a.shape}, mean_b shape: {mean_tokens_b.shape}")
            del reduced_tokens_a, reduced_tokens_b # type: ignore

            # Compute mean over width (W) for each row
            mean_a = mean_tokens_a.mean(dim=-1)  # [batch_size, windows_num, 1, H, W]
            mean_b = mean_tokens_b.mean(dim=-1)  # [batch_size, windows_num, 1, H, W]
            if show_dims: print(f"mean_a shape: {mean_a.shape}, mean_b shape: {mean_b.shape}")
            del mean_tokens_a, mean_tokens_b # type: ignore

            # Reshape for transformer: [batch_size * windows_num, Channels*H]
            reshaped_means_a = mean_a.reshape(-1, mean_a.shape[-1])
            reshaped_means_b = mean_b.reshape(-1, mean_b.shape[-1])
            if show_dims: print(f"Tokens_a shape: {reshaped_means_a.shape}, Tokens_b shape: {reshaped_means_b.shape}")
            del mean_a, mean_b

            # Convert to vectors
            encoded_tokens_a = self.convert_to_vectors(reshaped_means_a)
            encoded_tokens_b = self.convert_to_vectors(reshaped_means_b)
            if show_dims: print(f"Tokens_a shape: {encoded_tokens_a.shape}, Tokens_b shape: {encoded_tokens_b.shape}")
            del reshaped_means_a, reshaped_means_b
        
        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'Transformer':
            featured_tokens_a = self.transformer_encoder(encoded_tokens_a) # type: ignore
            featured_tokens_b = self.transformer_encoder(encoded_tokens_b) # type: ignore
            if show_dims: print(f"Tokens_a shape after permuting: {featured_tokens_a.shape}, Tokens_b shape after permuting: {featured_tokens_b.shape}")
            del encoded_tokens_a, encoded_tokens_b # type: ignore

            features_vector_a = featured_tokens_a.view(batches_num, windows_num, self.vector_size) 
            features_vector_b = featured_tokens_b.view(batches_num, windows_num, self.vector_size) 
            if show_dims: print(f"Tokens_a shape after reshaping: {features_vector_a.shape}, Tokens_b shape after reshaping: {features_vector_b.shape}")
            del featured_tokens_a, featured_tokens_b

        return features_vector_a, features_vector_b # type: ignore

import gc
import torch

# Example usage
if __name__ == "__main__":
    # Simulate grayscale image inputs
    image_a = torch.randn(8, 3, 224, 1024).to('cuda') # Batch of 32 grayscale images
    image_b = torch.randn(8, 3, 224, 1024).to('cuda')
    
    # Instantiate the alignment model
    model = EmbeddingModel(window_size=16, stride=8, vector_size=64,
                            model_arch='CNN').to('cuda') # ['CNN-Transformer','CNN','Transformer']

    # Forward pass: get token sequences for both images
    tokens_a, tokens_b = model(image_a, image_b,  show_dims=True)
    
    # Output token shapes
    print(f"Tokens A shape: {tokens_a.shape}")
    print(f"Tokens B shape: {tokens_b.shape}")
    del image_a, image_b 
    # del tokens_a, tokens_b, 
    del model
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except:
            pass
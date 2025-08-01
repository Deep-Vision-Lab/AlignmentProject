from matplotlib import patches
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return torch.stack(patches, dim=0)


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
        
        # self.channel_reducer = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.model_arch = model_arch

        if model_arch == 'CNN-Transformer' or model_arch == 'CNN':
            self.cnn_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            cnn_feature_dim = self.cnn_encoder.fc.in_features
            self.cnn_encoder.fc = nn.Linear(cnn_feature_dim, vector_size)

        if model_arch == 'CNN-Transformer' or model_arch == 'Transformer':
            self.transformer_encoder = TransformerEncoder(d_model=vector_size, nhead=8, num_layers=6)

        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
        self.model_arch = model_arch
    
    def forward(self, image_a, image_b, show_dims=False, debug=False, save_dir=None):
       
        #################################################################################
        if debug:
            os.makedirs(f'{save_dir}/A', exist_ok=True)
            os.makedirs(f'{save_dir}/B', exist_ok=True)

        patches_a = sliding_window(image_a, self.window_size, self.stride, debug_mode=debug, save_dir=f'{save_dir}/A/')
        patches_b = sliding_window(image_b, self.window_size, self.stride, debug_mode=debug, save_dir=f'{save_dir}/B/')
        if show_dims: print(f"Patches_a shape: {patches_a.shape}, Patches_b shape: {patches_b.shape}")
        
        batches_num, windows_num, Channels, H, W = patches_a.shape
        
        # Reshape patches to [batch_size, windows_num, Channels, H, W]
        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'CNN':        
            patches_a = patches_a.reshape(batches_num * windows_num, Channels, H, W)
            patches_b = patches_b.reshape(batches_num * windows_num, Channels, H, W)
            if show_dims: print(f"Patches_a shape after reshaping: {patches_a.shape}, Patches_b shape after reshaping: {patches_b.shape}")

        # # Compute mean over width (W) for each row
        # patches_a = self.channel_reducer(patches_a)  # [batch_size, windows_num, 1, H, W]
        # patches_b = self.channel_reducer(patches_b)  # [batch_size, windows_num, 1, H, W]
        # if show_dims: print(f"Patches_a shape after channel reduction: {patches_a.shape}, Patches_b shape after channel reduction: {patches_b.shape}")


        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'CNN':
            tokens_a = self.cnn_encoder(patches_a)
            tokens_b = self.cnn_encoder(patches_b)
            if show_dims: print(f"Tokens_a shape: {tokens_a.shape}, Tokens_b shape: {tokens_b.shape}")


        if self.model_arch == 'Transformer':
            # Compute mean over width (W) for each row
            mean_a = patches_a.mean(dim=2)  # [batch_size, windows_num, H, W]
            mean_b = patches_b.mean(dim=2)  # [batch_size, windows_num, H, W]
            if show_dims: print(f"mean_a shape: {mean_a.shape}, mean_b shape: {mean_b.shape}")

            # Compute mean over width (W) for each row
            mean_a = mean_a.mean(dim=-1)  # [batch_size, windows_num, 1, H, W]
            mean_b = mean_b.mean(dim=-1)  # [batch_size, windows_num, 1, H, W]
            if show_dims: print(f"mean_a shape: {mean_a.shape}, mean_b shape: {mean_b.shape}")
            
            # Reshape for transformer: [batch_size * windows_num, Channels*H]
            tokens_a = mean_a.reshape(-1, mean_a.shape[-1])
            tokens_b = mean_b.reshape(-1, mean_b.shape[-1])
            if show_dims: print(f"Tokens_a shape: {tokens_a.shape}, Tokens_b shape: {tokens_b.shape}")
            
            # Convert to vectors
            convert_to_vectors = nn.Sequential(
                nn.Linear(in_features=tokens_a.shape[-1], out_features=self.vector_size)
            ).to(patches_a.device) 
            tokens_a = convert_to_vectors(tokens_a)
            tokens_b = convert_to_vectors(tokens_b)
            if show_dims: print(f"Tokens_a shape: {tokens_a.shape}, Tokens_b shape: {tokens_b.shape}")

        
        if self.model_arch == 'CNN-Transformer' or self.model_arch == 'Transformer':
            tokens_a = self.transformer_encoder(tokens_a)
            tokens_b = self.transformer_encoder(tokens_b)
            if show_dims: print(f"Tokens_a shape after permuting: {tokens_a.shape}, Tokens_b shape after permuting: {tokens_b.shape}")
        

        tokens_a = tokens_a.view(batches_num, windows_num, self.vector_size)
        tokens_b = tokens_b.view(batches_num, windows_num, self.vector_size)
        if show_dims: print(f"Tokens_a shape after reshaping: {tokens_a.shape}, Tokens_b shape after reshaping: {tokens_b.shape}")
        
        return tokens_a, tokens_b


# Example usage
if __name__ == "__main__":
    # Simulate grayscale image inputs
    image_a = torch.randn(8, 3, 128, 1024)  # Batch of 32 grayscale images
    image_b = torch.randn(8, 3, 128, 1024)
    
    # Instantiate the alignment model
    model = EmbeddingModel(window_size=64, stride=32, vector_size=64,
                            model_arch='CNN') # ['CNN-Transformer','CNN','Transformer']
    
    # Forward pass: get token sequences for both images
    tokens_a, tokens_b = model(image_a, image_b,  show_dims=True)
    
    # Output token shapes
    print(f"Tokens A shape: {tokens_a.shape}")
    print(f"Tokens B shape: {tokens_b.shape}")
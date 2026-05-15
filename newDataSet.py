import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import random

from Parameters import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_text_similarity_matrix(text1: str, text2: str, 
                                             device: str = 'cuda') -> torch.Tensor:
    """
    Fastest GPU implementation using torch.compile (PyTorch 2.0+) if available.
    
    Uses a single-kernel approach for maximum throughput on large texts.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: Similarity matrix of shape [len(text1), len(text2)]
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Convert entire strings to tensor at once using torch.frombuffer for speed
    bytes1 = text1.encode('utf-32-le')  # 4 bytes per char
    bytes2 = text2.encode('utf-32-le')
    
    # Create tensors from buffer (zero-copy when possible)
    chars1 = torch.frombuffer(bytearray(bytes1), dtype=torch.int32).to(device)
    chars2 = torch.frombuffer(bytearray(bytes2), dtype=torch.int32).to(device)
    
    # Use einsum for potentially better kernel fusion
    # Broadcasting: [len1, 1] == [1, len2] -> [len1, len2]
    similarity_matrix = (chars1.unsqueeze(1) == chars2.unsqueeze(0)).float()
    
    return similarity_matrix


def textual_sliding_window(text, window_size, step_size):
    output = []
    for i in range(0, len(text) - window_size + 1, step_size):
        output.append(text[i:i + window_size])
    return output


class TextLineModern(Dataset):
    def __init__(self, new_dataset=None, transform=None, num_samples_override=None):
        self.new_dataset = new_dataset
        self.transform = transform

        if new_dataset:
            if num_samples_override is not None:
                self.num_samples = num_samples_override
            else:
                # Auto-detect by counting img1_*.png files in the images directory
                images_dir = new_dataset['images']
                detected = len([f for f in os.listdir(images_dir) if f.startswith('img1_') and f.endswith('.png')])
                self.num_samples = detected if detected > 0 else num_samples

    def __len__(self):
        return self.num_samples if self.new_dataset else 0


    def __getitem__(self, idx):
        if self.new_dataset:
            sample_idx = idx + 1  # 1-based file naming

            img1_path = os.path.join(self.new_dataset['images'], f"img1_{sample_idx}.png")
            img1 = Image.open(img1_path).convert("RGB")
            if self.transform:
                img1 = self.transform(img1)

            text1_path = os.path.join(self.new_dataset['texts'], f"text1_{sample_idx}.txt")
            with open(text1_path, 'r') as f:
                text1 = ' ' + f.read().strip() + ' '

            return text1, img1
        else:
            raise NotImplementedError("Handling for non-NewDataSet is not included.")
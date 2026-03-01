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
    def __init__(self, new_dataset=None, transform=None):
        self.new_dataset = new_dataset
        self.transform = transform

        if new_dataset:
            self.num_samples = 10000
            # Each sample is a positive pair (text1, img1)
            # Negative sampling is done in the collate function (in-batch negatives)

    def __len__(self):
        return self.num_samples if self.new_dataset else 0


    def __getitem__(self, idx):
        if self.new_dataset:
            sample_idx = idx + 1  # 1-based file naming

            # Load positive pair (text1, img1) - aligned
            img1_name = f"img1_{sample_idx}.png"
            text1_name = f"text1_{sample_idx}.txt"
            
            img2_name = f"img2_{sample_idx}.png"
            text2_name = f"text2_{sample_idx}.txt"

            img1_path = os.path.join(self.new_dataset['images'], img1_name)
            img1 = Image.open(img1_path).convert("RGB")
            if self.transform:
                img1 = self.transform(img1)

            text1_path = os.path.join(self.new_dataset['texts'], text1_name)
            with open(text1_path, 'r') as f:
                text1 = f.read().strip()
                text1 = ' ' + text1 + ' '

            img2_path = os.path.join(self.new_dataset['images'], img2_name)
            img2 = Image.open(img2_path).convert("RGB")
            if self.transform:
                img2 = self.transform(img2)

            text2_path = os.path.join(self.new_dataset['texts'], text2_name)
            with open(text2_path, 'r') as f:
                text2 = f.read().strip()
                text2 = ' ' + text2 + ' '

            return text1, img1, text2, img2
        else:
            raise NotImplementedError("Handling for non-NewDataSet is not included.")
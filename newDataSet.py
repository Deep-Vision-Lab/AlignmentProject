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
            num_samples = 10000
            # Pre-generate contrastive pairs for efficiency
            # For each sample i, randomly select a different sample j (j ≠ i) for negative pairing
            
            self.contrastive_pairs = []
            for i in range(num_samples):
                # Randomly select j where j ≠ i for negative sample
                available_indices = [x for x in range(num_samples) if x != i]
                j = random.choice(available_indices)
                
                # Store (img1_i, text1_i, img2_j, text2_j) where i ≠ j
                self.contrastive_pairs.append((
                    f"img1_{i+1}.png",   # img1 from sample i
                    f"text1_{i+1}.txt",  # text1 from sample i (aligned with img1)
                    f"img2_{j+1}.png",   # img2 from sample j (j ≠ i)
                    f"text2_{j+1}.txt"   # text2 from sample j (not aligned with text1/img1)
                ))

    def __len__(self):
        return len(self.contrastive_pairs)


    def __getitem__(self, idx):
        if self.new_dataset:
            # Get pre-generated contrastive pair (no random selection needed)
            img1_name, text1_name, img2_name, text2_name = self.contrastive_pairs[idx]

            # Load images
            img1_path = os.path.join(self.new_dataset['images'], img1_name)
            img2_path = os.path.join(self.new_dataset['images'], img2_name)

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # Load texts
            text1_path = os.path.join(self.new_dataset['texts'], text1_name)
            text2_path = os.path.join(self.new_dataset['texts'], text2_name)
            
            with open(text1_path, 'r') as f:
                text_line1 = f.read().strip()
                text_line1 = ' ' + text_line1 + ' '
                
            with open(text2_path, 'r') as f:
                text_line2 = f.read().strip()
                text_line2 = ' ' + text_line2 + ' '

            # Return for contrastive learning:
            # text1, img1 from sample i (aligned - positive pair)
            # text2, img2 from sample j where j ≠ i (not aligned - negative pair)
            return text_line1, img1, text_line2, img2
        else:
            raise NotImplementedError("Handling for non-NewDataSet is not included.")
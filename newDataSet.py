import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from Parameters import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def textual_sliding_window(text, window_size, step_size):
    output = []
    for i in range(0, len(text) - window_size + 1, step_size):
        output.append(text[i:i + window_size])
    return output


def word2Vec(textLine, img_width):
    embedding_dim = 512
    text_window_size = (window_size / img_width)*len(textLine)
    text_window_size = int(text_window_size)
    vocs_text = textual_sliding_window(textLine, window_size=text_window_size, step_size=text_window_size // 2 if text_window_size // 2 != 0 else 1)
    word_to_ix = {word: i for i, word in enumerate(vocs_text)}
    embedding = nn.Embedding(num_embeddings=len(vocs_text), embedding_dim=embedding_dim)
    word_idx = torch.tensor([torch.tensor(word_to_ix[text], dtype=torch.long) for text in vocs_text])
    word_vector = embedding(word_idx)

    return word_vector

class TextLineModern(Dataset):
    def __init__(self, regularScoreMatrix=True, new_dataset=None, transform=None):
        self.regularMatrix = regularScoreMatrix
        self.new_dataset = new_dataset
        self.transform = transform

        if new_dataset:
            # Reduce dataset size for memory testing
            self.image_pairs = [
                (f"img1_{i}.png", f"img2_{i}.png", f"scoreMatrix_{i}.npy", 
                 f"diffNWMatrix_{i}.npy", f"similarityMatrix_{i}.npy", 
                 f"text1_{i}.txt", f"text2_{i}.txt") for i in
                range(1, 3001)]  # Reduced from 3001 to 101

    def __len__(self):
        return len(self.image_pairs)

    def ScoreMapping(self, tensor):
        return torch.where(tensor == 1, torch.tensor(matchScore), 
                           torch.where(tensor == 0, torch.tensor(mismatchScore), 
                            tensor))


    def __getitem__(self, idx):
        if self.new_dataset:
            img1_name, img2_name, score_matrix_name, diffmatrix_name, similarity_matrix_name, text1_name, text2_name = self.image_pairs[idx]

            img1_path = os.path.join(self.new_dataset['images'], img1_name)
            img2_path = os.path.join(self.new_dataset['images'], img2_name)
            if self.regularMatrix:
                ScoreMatrix = os.path.join(self.new_dataset['matrices'], score_matrix_name)
            else:
                ScoreMatrix = os.path.join(self.new_dataset['diffNWmatrices'], diffmatrix_name)
            SimilarityMatrix = os.path.join(self.new_dataset['similarity_matrices'], similarity_matrix_name)

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Load matrices
            score_matrix = np.load(ScoreMatrix)
            score_matrix = torch.tensor(score_matrix, dtype=torch.float32)

            similar_matrix = np.load(SimilarityMatrix)
            similar_matrix = torch.tensor(similar_matrix, dtype=torch.float32)
            similar_matrix = self.ScoreMapping(similar_matrix)
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, score_matrix, similar_matrix, img1_name, img2_name
        else:
            raise NotImplementedError("Handling for non-NewDataSet is not included.")
from email.mime import image
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset


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


window_size = 64


class TextLineModern(Dataset):
    def __init__(self, datasetPaths, fonts, patchHeight, patchWidth, numberWords, new_dataset=None, transform=None):
        """
        Dataset class to handle NewDataSet structure.

        :param datasetPaths: Not used for NewDataSet.
        :param fonts: Not used for NewDataSet.
        :param patchHeight: Not used for NewDataSet.
        :param patchWidth: Not used for NewDataSet.
        :param numberWords: Not used for NewDataSet.
        :param new_dataset: Dictionary containing paths to 'images' and 'scoreMatrix' folders for NewDataSet.
        :param transform: Transformations to apply to images.
        """
        self.new_dataset = new_dataset
        self.transform = transform

        if new_dataset:
            # Preload file mappings for NewDataSet
            self.image_pairs = [
                (f"img1_{i}.png", f"img2_{i}.png", f"scoreMatrix_{i}.npy", f"text1_{i}.txt", f"text2_{i}.txt") for i in
                range(1, 10001)]
                # range(1, 3001)]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        if self.new_dataset:
            # Load images and score matrix for NewDataSet
            img1_name, img2_name, matrix_name, text1_name, text2_name = self.image_pairs[idx]

            img1_path = os.path.join(self.new_dataset['images'], img1_name)
            img2_path = os.path.join(self.new_dataset['images'], img2_name)

            matrix_path = os.path.join(self.new_dataset['scoreMatrix'], matrix_name)

            text1_path = os.path.join(self.new_dataset['texts'], text1_name)
            text2_path = os.path.join(self.new_dataset['texts'], text2_name)

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            score_matrix = np.load(matrix_path)

            with (open(text1_path, "r") as file):
                text1 = file.read()
            width1, _ = img1.size

            with (open(text2_path, "r") as file):
                text2 = file.read()
            width2, _ = img2.size


            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            score_matrix = torch.tensor(score_matrix, dtype=torch.float32,requires_grad=True).to(device)

            # read text files
            # Tokenize words from images
            seq1 = tokenize_based_on_non_connecting_letters(text1)
            seq2 = tokenize_based_on_non_connecting_letters(text2)
            
            return img1, img2, score_matrix, seq1, seq2, text1, text2
        else:
            raise NotImplementedError("Handling for non-NewDataSet is not included.")



def tokenize_based_on_non_connecting_letters(text):
    # Expanded list of non-connecting Arabic characters
    non_connecting_letters = {'ا', 'د', 'ذ', 'ر', 'ز', 'و', 'ى', ' ', 'أ', 'إ', 'ؤ', 'ء'}

    # Tokenize based on the presence of non-connecting letters or spaces
    tokens = []
    current_token = ''

    for char in text:
        current_token += char

        if char in non_connecting_letters:
            tokens.append(current_token)
            current_token = ''

    # Add the last token if it exists
    if current_token:
        tokens.append(current_token)

    # strip all componets with
    lst=[]
    for t in tokens:
        if ' ' in t and len(t)> 1:
           lst.append(t.strip())
        else:
            lst.append(t)
    lst = remove_whitespace_from_subwords(lst)
    return lst

def remove_whitespace_from_subwords(subwords_list):
    """
    Remove all empty strings or strings consisting solely of white spaces from the list of subwords.

    :param subwords_list: List of subwords which may include spaces or white spaces.
    :return: List of subwords with all white spaces removed.
    """
    # Use list comprehension to filter out empty or whitespace-only subwords
    cleaned_subwords_list = [subword for subword in subwords_list if subword.strip()]
    return cleaned_subwords_list
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont
from fetch_arabic_sentence import fetch_arabic_sentence
from splitTextLine import tokenize_based_on_non_connecting_letters
from augmentSenetece import augment_sentence
from drawTextLine import create_text_image
from matchingAlgorthim import smith_waterman
from PatchesSlicing import slicing_window, Run_CNN

import glob
import os
import random
from typing import Tuple


class TextLineModern(Dataset):
    def __init__(
            self,
            datasetPaths,
            fonts,
            patchHeight: int,
            patchWidth: int,
            numberWords: Tuple[int, int],
            is_train: bool = True
    ):
        self.datasetPaths = datasetPaths
        self.fonts = fonts
        self.patchHeight = patchHeight
        self.patchWidth = patchWidth
        self.numberWords = numberWords
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the PIL Image to a tensor (values between 0 and 1)
        ])
        self.is_train = is_train

    def resize_image(self, image):
        # Get the original dimensions
        width, height = image.size

        # Compute the new width while preserving the aspect ratio
        new_width = int((self.patchHeight / height) * width)

        # Resize the image
        resized_image = image.resize((new_width, self.patchHeight), Image.LANCZOS)
        return resized_image

    def __getitem__(self, idx):  # 120 pages
        # Assuming self.numberWords is a tuple representing the range (min_words, max_words)
        nWords = random.randint(self.numberWords[0], self.numberWords[1])
        _, sentencewithNwords = fetch_arabic_sentence(nWords, self.datasetPaths)
        subWordsOfNwords = tokenize_based_on_non_connecting_letters(sentencewithNwords)

        nWords = random.randint(self.numberWords[0], self.numberWords[1])
        _, words = fetch_arabic_sentence(nWords, self.datasetPaths)

        subWordsOfAddwords = tokenize_based_on_non_connecting_letters(words)
        sentenceAddAugmetation = augment_sentence(subWordsOfNwords, new_words=subWordsOfAddwords, operation="add")
        font1, font2 = random.sample(self.fonts, 2)
        textLineOriginal = " ".join(subWordsOfNwords)
        imgTextLineOriginal = create_text_image(textLineOriginal, font_path=font1)
        imgTextLineOriginalResized = self.resize_image(imgTextLineOriginal)
        textLineAugmented = " ".join(sentenceAddAugmetation)
        imgTextLineAugmented = create_text_image(textLineAugmented, font_path=font2)
        imgTextLineAugmentedResized = self.resize_image(imgTextLineAugmented)
        imgTextLineOriginalTensor = self.transform(imgTextLineOriginalResized)
        imgTextLineAugmentedTensor = self.transform(imgTextLineAugmentedResized)

        score, alignment1, alignment2, score_matrix, track_matrix = smith_waterman(subWordsOfNwords,
                                                                                   sentenceAddAugmetation)
        if self.is_train:
            return imgTextLineOriginalTensor, imgTextLineAugmentedTensor, score_matrix, track_matrix
        else:
            return imgTextLineOriginalTensor, imgTextLineAugmentedTensor, score_matrix, track_matrix, subWordsOfNwords, sentenceAddAugmetation

    def __len__(self):
        return 100000


def extending_image(data, batch_image_size):
    zero_vector = torch.zeros(data.shape[0], data.shape[1],
                              batch_image_size*window_width - data.shape[2])
    data = torch.cat([data.float(), zero_vector.float()], dim=2)
    return data


def extending_matrix(matrix, matrix_height_pad, matrix_width_pad):
    # pad = (batch_image_size - matrix.shape[1], 0, batch_image_size - matrix.shape[0], 0)
    pad = (matrix_width_pad - matrix.shape[1], 0, matrix_height_pad - matrix.shape[0], 0)
    score_matrix = torch.from_numpy(matrix)
    matrix = F.pad(score_matrix, pad, mode='constant', value=0)
    matrix = matrix.float()
    return matrix


def max_vetor_length(batch):
    batch1_max = max(data[0].shape[2] for data in batch)
    batch2_max = max(data[1].shape[2] for data in batch)
    return max(batch1_max, batch2_max)


batch_image_size = 2024
window_width = 150


def build_route_traceback(score_matrices, traceback_matrices):
    lst_tracback = []
    for k in range(len(score_matrices)):
        score_matrix_np = score_matrices[k].cpu().detach().numpy()
        traceback = traceback_matrices[k].cpu().detach().numpy()
        traceback_matrix = np.zeros_like(score_matrix_np)
        i, j = np.unravel_index(np.argmax(score_matrix_np), score_matrix_np.shape)
        while i > 0 and j > 0 and score_matrix_np[i, j] != 0:
            traceback_matrix[i, j] = 1
            if traceback[i,j] == 0:  # No Direction
                traceback_matrix[i, j] = 0
                break
            elif traceback[i,j] == 1:  # Diagonal
                i, j = i - 1, j - 1
            elif traceback[i,j] == 2:  # Up
                i, j = i - 1, j
            elif traceback[i,j] == 3:  # Left
                i, j = i, j - 1
        lst_tracback.append(torch.tensor(traceback_matrix))
    traceback_matrices = torch.stack(lst_tracback)
    return traceback_matrices

def collate_fn(batch):
    max_image_size = max_vetor_length(batch)

    windows_num = int(max_image_size / window_width) if max_image_size % window_width == 0 \
        else (int(max_image_size / window_width)) + 1

    max_matrix_height = max([torch.tensor(data[2]).shape[0] for data in batch])
    max_matrix_width = max([torch.tensor(data[2]).shape[1] for data in batch])

    max_matrix_size = max([max_matrix_height, max_matrix_width, windows_num, max_image_size])

    imgTextLineOriginalTensor = torch.stack([extending_image(data[0], max_matrix_size) for data in batch])
    imgTextLineAugmentedTensor = torch.stack([extending_image(data[1], max_matrix_size) for data in batch])

    score_matrix = torch.stack([extending_matrix(data[2], max_matrix_size, max_matrix_size) for data in batch])
    traceback_matrix = torch.stack([extending_matrix(data[3], max_matrix_size, max_matrix_size) for data in batch])
    traceback_matrix = build_route_traceback(score_matrix, traceback_matrix)

    return imgTextLineOriginalTensor, imgTextLineAugmentedTensor, traceback_matrix


def collate_fn_test(batch):
    max_image_size = max_vetor_length(batch)

    windows_num = int(max_image_size / window_width) if max_image_size % window_width == 0 \
        else (int(max_image_size / window_width)) + 1

    max_matrix_height = max([torch.tensor(data[2]).shape[0] for data in batch])
    max_matrix_width = max([torch.tensor(data[2]).shape[1] for data in batch])

    max_matrix_size = max([max_matrix_height, max_matrix_width, windows_num])


    imgTextLineOriginalTensor = torch.stack([extending_image(data[0], max_matrix_size) for data in batch])
    imgTextLineAugmentedTensor = torch.stack([extending_image(data[1], max_matrix_size) for data in batch])

    score_matrix = torch.stack([extending_matrix(data[2], max_matrix_size, max_matrix_size) for data in batch])
    traceback_matrix = torch.stack([extending_matrix(data[3], max_matrix_size, max_matrix_size) for data in batch])

    traceback_route_matrix = build_route_traceback(score_matrix, traceback_matrix)

    txtOriginal = [data[4] for data in batch]

    txtAugmented = [data[5] for data in batch]

    return imgTextLineOriginalTensor, imgTextLineAugmentedTensor, score_matrix, traceback_matrix, txtOriginal, txtAugmented, traceback_route_matrix

# if __name__ == '__main__':
#
#     datasetTextLines = TextLineModern(
#        ["datasets/ArabicDialect", "datasets/QuranDataset"],
#          glob.glob(os.path.join('fonts','*')),
#        50,
#          25, # sliding w'
#        [5,7]
#     )
#
#     train_loader = DataLoader(datasetTextLines, batch_size=64, collate_fn=collate_fn)
#
#     for (lin1,lin2,score_matrix) in train_loader:
#         print(lin1.shape)
#         print(lin2.shape)
#         print(score_matrix.shape)
#         lin1 = slicing_window(lin1)
#         lin2 = slicing_window(lin2)
#         print(lin1.shape)
#         print(lin2.shape)
#         model_name = 'simpleCNN'  # Change to desired model: 'simpleCNN', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'vgg16', 'vgg19'
#         lin1,lin2 = Run_CNN(model_name,lin1,lin2)
#         print(lin1.shape)
#         print(lin2.shape)
#
#         exit(0)

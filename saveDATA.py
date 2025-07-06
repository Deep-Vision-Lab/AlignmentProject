import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import FancyArrow, Rectangle, Circle


def slicing_image_show(image, index, ax, window_width):
    image = torch.permute(image, (1, 2, 0)).cpu()
    ax[index].imshow(image)

    for i in range(0, image.shape[1], window_width):
        top_left = (i, 0)
        rect = Rectangle(top_left, window_width, image.shape[0], linewidth=2, edgecolor='red' if i % 2 == 0 else 'blue',
                         facecolor='none')
        ax[index].add_patch(rect)

    ax[index].axis('off')
    


def plot_Aligned_slices(matchMatrix, img1_size, img2_size, window_width, fig, ax):
    seq1_pos = ax[0].get_position()
    seq1_window_width = (seq1_pos.width * window_width) / img1_size[2]
    seq2_pos = ax[1].get_position()
    seq2_window_width = (seq2_pos.width * window_width) / img2_size[2]

    for i in range(matchMatrix.shape[0]):
        for j in range(matchMatrix.shape[1]):

            if matchMatrix[i, j] == 1:
                new_x_seq1 = seq1_pos.x0 + i * seq1_window_width
                arrow_point_seq1 = new_x_seq1 + seq1_window_width / 2
                new_x_seq1 = arrow_point_seq1 if arrow_point_seq1 < seq1_pos.x1 \
                    else new_x_seq1 + (seq1_pos.x1 - new_x_seq1) / 2

                new_x_seq2 = seq2_pos.x0 + j * seq2_window_width
                arrow_point_seq2 = new_x_seq2 + seq2_window_width / 2
                new_x_seq2 = arrow_point_seq2 if arrow_point_seq2 < seq2_pos.x1 \
                    else new_x_seq2 + (seq2_pos.x1 - new_x_seq2) / 2

                seq1_xy = [new_x_seq1, seq1_pos.y0]
                seq2_xy = [new_x_seq2, seq2_pos.y1 + 0.03]

                dx = seq2_xy[0] - seq1_xy[0]
                dy = seq2_xy[1] - seq1_xy[1]

                arrow = FancyArrow(seq1_xy[0], seq1_xy[1], dx, dy, width=0.003, head_width=0.03,
                                head_length=0.05, color='red', transform=fig.transFigure)
                fig.add_artist(arrow)
                break


def buildAlignedImages(Seq1Img, Seq2Img, traceback_matrix, window_width, plot_pathFile):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    slicing_image_show(Seq1Img.squeeze(), 0, ax, window_width)
    slicing_image_show(Seq2Img.squeeze(), 1, ax, window_width)
    plot_Aligned_slices(traceback_matrix, Seq1Img.shape, Seq2Img.shape, window_width, fig, ax)
    plt.savefig(plot_pathFile)


def vectors_similarity(token_a, token_b, plot_path, batch_idx, i):
    with open(plot_path,'w') as file:
        for k, _ in enumerate(token_a):
            for l, _ in enumerate(token_b):
                cosine_sim = F.cosine_similarity(token_a[k], token_b[l], dim=0, eps=1e-8)
                file.write(f'vectors {batch_idx}_{i} index: [{k}][{l}] similarity: {cosine_sim}\n')


def print_elements(token, xlsx_path):
    token = token.detach().cpu().numpy()
    df = pd.DataFrame(token)
    df.to_excel(xlsx_path, index=False)


def save_Vectors_plot(vector1, vector2, path1, path2):
    vector1 = vector1.reshape(-1, 1)
    vector2 = vector2.reshape(-1, 1)

    _, ax = plt.subplots(1, 2, figsize=(20, 10))

    sns.heatmap(vector1, cmap='viridis', annot=True, fmt=".2f", ax=ax[0], cbar=False)
    ax[0].set_title(f"Alignment Path")
    ax[0].axis("off")

    sns.heatmap(vector2, cmap='viridis', annot=True, fmt=".2f", ax=ax[1], cbar=False)
    ax[1].set_title(f"Smith Path")
    ax[1].axis("off")

    plt.savefig(path1)

    _, ax = plt.subplots(1, 1, figsize=(20, 10))

    sns.heatmap(np.abs(vector1-vector2), cmap='viridis', annot=True, fmt=".2f", ax=ax, cbar=False)
    ax.set_title(f"Alignment Path")
    ax.axis("off")

    plt.savefig(path2)
import glob
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress visualization
import seaborn as sns
from PatchesSlicing import select_model, slicing_window, Run_CNN
from dataset import TextLineModern, collate_fn, collate_fn_test, window_width
from AlignModel import Alignment, diff_smith_waterman
from drawTextLine import plot_heatmap, plot_traceback

height = 50
width = window_width
model_name = 'ResNet18'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 2



def train_model(model, alignmentModel, train_loader, optimizer, criterion, num_samples=10000, epochs=10,
                threshold=0.9):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        correct = 0
        total = 0
        epoch_loss = 0
        train_iter = iter(train_loader)

        # Wrap train_loader with tqdm to visualize batch progress
        for _ in tqdm(range(num_samples), desc=f"Training Epoch {epoch + 1}", file=sys.stdout):
            lin1, lin2, traceback_matrix = next(train_iter)

            lin1 = lin1.to(device)
            lin2 = lin2.to(device)

            optimizer.zero_grad()
            lin1 = slicing_window(lin1, width).to(device)
            lin2 = slicing_window(lin2, width).to(device)

            lin1, lin2 = Run_CNN(model, lin1, lin2)
            outputs = diff_smith_waterman(alignmentModel, lin1, lin2, lin1.shape[1]).to(device)


            loss = criterion(outputs, traceback_matrix)
            loss.backward(retain_graph=True)
            optimizer.step()

            # loss
            epoch_loss += loss.item()

            # accuracy
            differ = torch.abs(outputs - traceback_matrix)
            correct += torch.sum(differ < threshold)
            total += torch.sum(differ < threshold) + torch.sum(differ >= threshold)

            del lin1, lin2, outputs,traceback_matrix  # Delete the tensors
            torch.cuda.empty_cache()  # Release unused memory from GPU cache

        avg_epoch_loss = epoch_loss / (batch_size * num_samples)
        print(f'Loss: {avg_epoch_loss:.4f}')

        accuracy = (correct / total) * 100
        print(f'accuracy: {accuracy:.4f}%')

    return model


def evaluation_model(model, alignmentModel, test_loader, num_samples, threshold = 0.9):
    model.eval()
    correct = 0
    total = 0
    test_iter = iter(test_loader)
    output_groundTruth = []

    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc=f"Evaluate samples", file=sys.stdout):
            lin1, lin2, score_matrix, traceback_matrix, txtOriginal, txtAugmented,backtrace_route = next(test_iter)

            lin1 = lin1.to(device)
            lin2 = lin2.to(device)
            score_matrix = score_matrix.to(device)
            traceback_matrix = traceback_matrix.to(device)

            lin1 = slicing_window(lin1, width).to(device)
            lin2 = slicing_window(lin2, width).to(device)

            lin1, lin2 = Run_CNN(model, lin1, lin2)

            outputs = diff_smith_waterman(alignmentModel, lin1, lin2, lin1.shape[1]).to(device)

            # convert the traceback matrix for the score matrix into a binary
            fig, ax = plt.subplots(1, 1, figsize=(18, 10))
            sns.heatmap(backtrace_route[0], cmap='viridis', annot=True,ax=ax)
            plt.show()
            fig, ax = plt.subplots(1, 1, figsize=(18, 10))
            sns.heatmap(traceback_matrix[0], cmap='viridis', annot=True,ax=ax)
            plt.show()
            # accuracy
            differ = torch.abs(outputs - backtrace_route)
            correct += torch.sum(differ < threshold)
            total += torch.sum(differ < threshold) + torch.sum(differ >= threshold)

            for i in range(batch_size):
                output_groundTruth.append(
                    [score_matrix[i], traceback_matrix[i], outputs[i], txtOriginal[i], txtAugmented[i]])

            del lin1, lin2, score_matrix, traceback_matrix, outputs  # Delete the tensors
            torch.cuda.empty_cache()  # Release unused memory from GPU cache
        accuracy = (correct / total) * 100
        print(f'accuracy: {accuracy:.4f}%')
    return output_groundTruth


def make_backtrack_matrix(score_matrix):
    score_matrix_np = score_matrix.cpu().detach().numpy()
    traceback_matrix = np.zeros_like(score_matrix_np)
    i, j = np.unravel_index(np.argmax(score_matrix_np), score_matrix_np.shape)
    while i > 0 and j > 0 and score_matrix_np[i, j] != 0:
        direction = np.argmax(np.array([0,score_matrix[i - 1, j - 1],score_matrix[i - 1, j],score_matrix[i, j - 1]]))

        if direction == 0:  # No Direction
            traceback_matrix[i, j] = 0
            break
        elif direction == 1:  # Diagonal
            traceback_matrix[i, j] = 1
            i,j = i-1,j-1
        elif direction == 2:  # Up
            traceback_matrix[i, j] = 2
            i,j = i-1,j
        elif direction == 3:  # Left
            traceback_matrix[i, j] = 3
            i,j = i,j-1
    return traceback_matrix


def show_score_matrix(score_matrix, track_matrix, seq1=[],seq2=[]):
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    if seq1 and seq2:
        plot_traceback(score_matrix, track_matrix, ax, seq1, seq2)
    else:
        plot_traceback(score_matrix, track_matrix, ax)
    plt.show()


if __name__ == '__main__':
    train_flag = False
    datasetTextLines = TextLineModern(
        ["datasets/ArabicDialect", "datasets/QuranDataset"],
        glob.glob(os.path.join('fonts', '*')),
        height,
        width,
        [5, 7],
        is_train=train_flag
    )
    weights_filename = 'model_weights_0001.pth'
    # Train/Evaluation parameters
    model = select_model(model_name, height, width)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    alignmentModel = Alignment(match_score=3, miss_score=-6)

    if train_flag:
        train_loader = DataLoader(datasetTextLines, batch_size=batch_size, collate_fn=collate_fn)
        model = train_model(model, alignmentModel, train_loader, optimizer, criterion, num_samples=8000, epochs=20,
                            threshold=0.9)
        print('Training finished')

        torch.save(model.state_dict(), weights_filename)
        print('Weights are saved')
    else:
        test_loader = DataLoader(datasetTextLines, batch_size=batch_size, collate_fn=collate_fn_test)
        state_dict = torch.load(weights_filename, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        print('Weights are loaded')

        output_groundTruth = evaluation_model(model, alignmentModel, test_loader, num_samples=1)
        for output in output_groundTruth:
            show_score_matrix(output[0], output[1], output[3], output[4])
            traceback_matrix = make_backtrack_matrix(output[2])
            show_score_matrix(output[2], traceback_matrix)

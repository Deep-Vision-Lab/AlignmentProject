import torch
import numpy as np
import sys
import os

# Assuming you're importing models and helpers from the root project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soft_dtw_cuda import SoftDTW
from NWAlgo import _path_to_matrix  # or whatever helper gives you the exact alignment path

def calculate_recall_and_mrr(distance_matrix):
    """
    distance_matrix: [N, M] array where N = query images, M = database target images.
    Returns Recall@1, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).
    Assumes correct alignment is when query `i` matches target `i`.
    """
    N = distance_matrix.shape[0]
    ranks = np.zeros(N)
    
    for i in range(N):
        sorted_indices = np.argsort(distance_matrix[i, :])
        ranks[i] = np.where(sorted_indices == i)[0][0] + 1  # 1-based indexing
        
    recall_1 = np.sum(ranks <= 1) / N
    recall_5 = np.sum(ranks <= 5) / N
    recall_10 = np.sum(ranks <= 10) / N
    mrr = np.mean(1 / ranks)
    
    return {
        'Recall@1': recall_1,
        'Recall@5': recall_5,
        'Recall@10': recall_10,
        'MRR': mrr
    }

def calculate_average_diagonal_deviation(predicted_path, seq_len_a, seq_len_b):
    """
    Calculates the Average Diagonal Deviation (ADD) between an expected chronolocial perfect diagonal
    and the actual dynamic time warping predicted path.
    predicted_path: list of (x, y) coordinates representing the alignment path.
    seq_len_a: Arabic segment length
    seq_len_b: English segment length
    """
    deviations = []
    # Perfect diagonal line: y = (seq_len_b / seq_len_a) * x
    slope = seq_len_b / seq_len_a if seq_len_a > 0 else 1
    
    for (x, y) in predicted_path:
        expected_y = slope * x
        deviation = abs(y - expected_y)
        deviations.append(deviation)
        
    return np.mean(deviations) if deviations else 0.0

if __name__ == "__main__":
    print("Executing Phase 2: Quantitative Inference (The Hard Numbers)")
    # TODO: Load Test DataLoader and Model
    # Loop over the test dataset, cross-compute DTW distances, and aggregate these metrics across your test set.
    print("Testing framework defined. Map this to your specific dataset loaders to get the final hard numbers for the paper.")

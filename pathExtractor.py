import numpy as np
import torch

#NW Path

def compute_traceback_path(matrix, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
    """Compute the optimal alignment path using traceback.
    
    At each step from (i,j), compute scores for each direction:
    - diag:   matrix[i-1,j-1] + aij  (aij from similarity_matrix directly)
    - up:     matrix[i-1,j] + gap_penalty
    - left:   matrix[i,j-1] + gap_penalty
    Then use softmax + argmax to determine the direction.
    Priority on ties: diag > up > left. Ends at (0,0).
    """
    # Start at bottom-right if no explicit starting position provided
    if position is None:
        n, m = matrix.shape
        i, j = n - 1, m - 1
    else:
        i, j = position
    path = []
    start_i, start_j = i, j  # Store starting position

    # Trace strictly toward (0,0)
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 and j == 0:
            break

        aij = match_score if similarity_matrix[i, j] > 0 else miss_score
        # Check diagonal, up, and left moves
        diag_score = matrix[i-1, j-1] + aij if i > 0 and j > 0 else 0
        up_score = matrix[i, j-1] + gap_penalty if j > 0 else 0 
        left_score = matrix[i-1, j] + gap_penalty if i > 0 else 0 
        
        # Find the maximum score using softmax + argmax
        max_score_idx = (torch.tensor([diag_score, up_score, left_score])).argmax().item()

        # Priority: diagonal -> up -> left when scores are equal (standard Smith-Waterman)
        if max_score_idx == 0:
            i -= 1
            j -= 1
        elif max_score_idx == 1:
            j -= 1
        elif max_score_idx == 2:
            i -= 1
        # Clean up
        del diag_score, up_score, left_score, max_score_idx

    # Reverse to get path from start to end
    forward_path = path[::-1]

    return forward_path, (start_i, start_j)


def NW_Path(matrices, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
    path_matrix = torch.zeros_like(matrices,device=matrices.device)
    starting_points = []
    for i, matrix in enumerate(matrices):
        if position is None:
            path, (start_i, start_j) = compute_traceback_path(matrix, similarity_matrix[i],
                                                            match_score, miss_score, gap_penalty)
        else:    
            path, (start_i, start_j) = compute_traceback_path(matrix, similarity_matrix[i],
                                                            match_score, miss_score, gap_penalty,
                                                            position[i])
        for (x, y) in path:
            path_matrix[i, x, y] = 1
        starting_points.append((start_i, start_j))
        # Clean up
        del path, start_i, start_j

    return path_matrix, starting_points


#==================================================================================================
# Diff NW Path

def compute_diff_traceback_path(matrix, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None, temp=100.0):
    """Compute the optimal alignment path using traceback with temperature-scaled softmax"""
    # Start at bottom-right if no explicit starting position provided
    if position is None:
        n, m = matrix.shape
        i, j = n - 1, m - 1
    else:
        i, j = position
    path = []
    start_i, start_j = i, j  # Store starting position
    while i >= 0 and j >= 0:
        path.append((i, j))
        aij = similarity_matrix[i, j]
        # Check diagonal, up, and left moves
        diag_score = matrix[i-1, j-1] + aij if i > 0 and j > 0 else 0
        up_score = matrix[i, j-1] + gap_penalty if j > 0 else 0 
        left_score = matrix[i-1, j] + gap_penalty if i > 0 else 0 
        
        # Find the maximum score using temperature-scaled softmax + argmax
        scores = torch.tensor([diag_score, up_score, left_score]) / temp
        max_score_idx = torch.softmax(scores, dim=0).argmax().item()

        # Priority: diagonal -> up -> left when scores are equal (standard Smith-Waterman)
        if max_score_idx == 0:
            i -= 1
            j -= 1
        elif max_score_idx == 1:
            j -= 1
        elif max_score_idx == 2:
            i -= 1
        # Clean up
        del diag_score, up_score, left_score, max_score_idx, scores

    return path[::-1], (start_i, start_j) # Reverse to get path from start to end


def diff_NW_Path(matrices, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None, temp=100.0):
    path_matrix = torch.zeros_like(matrices,device=matrices.device)
    starting_points = []
    for i, matrix in enumerate(matrices):
        if position is None:
            path, (start_i, start_j) = compute_diff_traceback_path(matrix, similarity_matrix[i],
                                                            match_score, miss_score, gap_penalty,
                                                            temp=temp)
        else:    
            path, (start_i, start_j) = compute_diff_traceback_path(matrix, similarity_matrix[i],
                                                            match_score, miss_score, gap_penalty,
                                                            position[i], temp=temp)
        for (x, y) in path:
            path_matrix[i, x, y] = 1
        starting_points.append((start_i, start_j))
        # Clean up
        del path, start_i, start_j
        
    return path_matrix, starting_points
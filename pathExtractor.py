import numpy as np
import torch

#SW Path
# Diff SW Path

def compute_traceback_path(matrix, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
    """Compute the optimal alignment path using traceback"""
    matrix_np = matrix.detach().cpu().numpy()
    # Find the maximum score position as starting point
    i, j = np.unravel_index(np.argmax(matrix_np), matrix.shape) if position is None else position
    path = []
    start_i, start_j = i, j  # Store starting position
    while i >= 0 and j >= 0 and matrix[i, j] > 0:
        path.append((i, j))
        aij = match_score if similarity_matrix[i, j] == 1 else miss_score
        # Check diagonal, up, and left moves
        diag_score = matrix[i-1, j-1] + aij if i > 0 and j > 0 else 0
        up_score = matrix[i-1, j] + gap_penalty if i > 0 else 0 
        left_score = matrix[i, j-1] + gap_penalty if j > 0 else 0 
        
        # Find the index of maximum score
        scores = [diag_score, up_score, left_score]
        max_score_idx = scores.index(max(scores))

        # Priority: diagonal -> up -> left when scores are equal (standard Smith-Waterman)
        if max_score_idx == 0:
            i -= 1
            j -= 1
        elif max_score_idx == 1:
            i -= 1
        elif max_score_idx == 2:
            j -= 1
        # Clean up
        del diag_score, up_score, left_score, scores, max_score_idx

    return path[::-1], (start_i, start_j) # Reverse to get path from start to end


def SW_Path(matrices, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
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
# Diff SW Path

def compute_diff_traceback_path(matrix, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
    """Compute the optimal alignment path using traceback"""
    matrix_np = matrix.detach().cpu().numpy()
    # Find the maximum score position as starting point
    i, j = np.unravel_index(np.argmax(matrix_np), matrix.shape) if position is None else position
    path = []
    start_i, start_j = i, j  # Store starting position
    while i >= 0 and j >= 0 and matrix[i, j] > 0:
        path.append((i, j))
        aij = similarity_matrix[i, j]
        # Check diagonal, up, and left moves
        diag_score = matrix[i-1, j-1] + aij if i > 0 and j > 0 else 0
        up_score = matrix[i-1, j] + gap_penalty if i > 0 else 0 
        left_score = matrix[i, j-1] + gap_penalty if j > 0 else 0 
        
        # Find the maximum score using simple max
        max_score_idx = (torch.exp(torch.tensor([diag_score, up_score, left_score]))
                        / torch.exp(torch.tensor([diag_score, up_score, left_score])).sum()).argmax().item()

        # Priority: diagonal -> up -> left when scores are equal (standard Smith-Waterman)
        if max_score_idx == 0:
            i -= 1
            j -= 1
        elif max_score_idx == 1:
            i -= 1
        elif max_score_idx == 2:
            j -= 1
        # Clean up
        del diag_score, up_score, left_score, max_score_idx

    return path[::-1], (start_i, start_j) # Reverse to get path from start to end


def diff_SW_Path(matrices, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
    path_matrix = torch.zeros_like(matrices,device=matrices.device)
    starting_points = []
    for i, matrix in enumerate(matrices):
        if position is None:
            path, (start_i, start_j) = compute_diff_traceback_path(matrix, similarity_matrix[i],
                                                            match_score, miss_score, gap_penalty)
        else:    
            path, (start_i, start_j) = compute_diff_traceback_path(matrix, similarity_matrix[i],
                                                            match_score, miss_score, gap_penalty,
                                                            position[i])
        for (x, y) in path:
            path_matrix[i, x, y] = 1
        starting_points.append((start_i, start_j))
        # Clean up
        del path, start_i, start_j
        
    return path_matrix, starting_points
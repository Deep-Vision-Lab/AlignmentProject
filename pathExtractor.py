import numpy as np
import torch


def build_backtrace_matrix_2d(matrix):
    path_matrix = np.zeros_like(matrix)
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    # i, j = check_max_on_axes(matrix)
    while i > 0 and j > 0:
        
        if i != 0 and j != 0:
            direction = np.argmax(
                np.array([0, matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]))
        elif i > 0 and j == 0:
            direction = 1
        elif i == 0 and j > 0:
            direction = 2
        else:
            direction = 0  # No Direction
        
        
        if direction == 0:  # No Direction  
            path_matrix[i, j] = 1
            if i > 0 and j > 0:
                i, j = i - 1, j - 1
            elif i > 0 and j < 0:
                i = i - 1
                # break
            elif i < 0 and j > 0:
                j = j - 1
            elif i <= 0 and j <= 0:
                break

        elif direction == 1:  # Up
            path_matrix[i, j] = 2
            if i > 0:
                i = i - 1

        elif direction == 2:  # Left
            path_matrix[i, j] = 3
            if j > 0:
                j = j - 1

        elif direction == 3:  # Diagonal
            path_matrix[i, j] = 1
            if i > 0 and j > 0:
                i, j = i - 1, j - 1

    return path_matrix


def build_forward_traceback_matrix_2d(matrix, path_matrix):
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    # i, j = check_max_on_axes(matrix)
    while i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
        if i != matrix.shape[0] - 1 and j != matrix.shape[1] - 1:
            direction = np.argmax(
                np.array([0, matrix[i + 1, j], matrix[i, j + 1], matrix[i + 1, j + 1]]))
        elif i < matrix.shape[0] - 1 and j == matrix.shape[1] - 1:
            direction = 1
        elif i == matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
            direction = 2
        else:
            direction = 0  # No Direction
        
        if direction == 0:  # No Direction
            path_matrix[i, j] = 0
            if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                i, j = i + 1, j + 1
            elif i < matrix.shape[0] - 1 and j >= matrix.shape[1] - 1:
                i = i + 1
            elif i >= matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                j = j + 1
            elif i >= matrix.shape[0] - 1 and j >= matrix.shape[1] - 1:
                break
        elif direction == 1:  # Down
            path_matrix[i, j] = 2
            if i < matrix.shape[0] - 1:
                i = i + 1
        elif direction == 2:  # Right
            path_matrix[i, j] = 3
            if j < matrix.shape[1] - 1:
                j = j + 1
        elif direction == 3:  # Diagonal
            path_matrix[i, j] = 1
            if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                i, j = i + 1, j + 1

    return path_matrix


def extract_traceback_path(matrix):
    path_matrix = build_backtrace_matrix_2d(matrix)
    path_matrix = build_forward_traceback_matrix_2d(matrix, path_matrix)
    return path_matrix


#==================================================================================================

def build_backtrace_matrix_3d(matrices):
    path_matrix = np.zeros_like(matrices)
    # i, j = check_max_on_axes(matrix)
    for batch_idx, matrix in enumerate(matrices):
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        # i, j = check_max_on_axes(matrix)
        while i > 0 and j > 0:
            if i != 0 and j != 0:
                direction = np.argmax(
                    np.array([0, matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]))
            elif i > 0 and j == 0:
                direction = 1
            elif i == 0 and j > 0:
                direction = 2
            else:
                direction = 0  # No Direction
            
            
            if direction == 0:  # No Direction  
                path_matrix[batch_idx, i, j] = 0
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1
                elif i > 0 and j < 0:
                    i = i - 1
                    # break
                elif i < 0 and j > 0:
                    j = j - 1
                elif i <= 0 and j <= 0:
                    break

            elif direction == 1:  # Up
                path_matrix[batch_idx, i, j] = 2
                if i > 0:
                    i = i - 1

            elif direction == 2:  # Left
                path_matrix[batch_idx, i, j] = 3
                if j > 0:
                    j = j - 1

            elif direction == 3:  # Diagonal
                path_matrix[batch_idx, i, j] = 1
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1

    return path_matrix


def build_forward_traceback_matrix_3d(matrices, path_matrix):
    for batch_idx, matrix in enumerate(matrices):
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        # i, j = check_max_on_axes(matrix)
        while i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
            if i != matrix.shape[0] - 1 and j != matrix.shape[1] - 1:
                direction = np.argmax(
                    np.array([0, matrix[i + 1, j], matrix[i, j + 1], matrix[i + 1, j + 1]]))
            elif i < matrix.shape[0] - 1 and j == matrix.shape[1] - 1:
                direction = 1
            elif i == matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                direction = 2
            else:
                direction = 0  # No Direction
            

            if direction == 0:  # No Direction
                path_matrix[batch_idx, i, j] = 0
                if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                    i, j = i + 1, j + 1
                elif i < matrix.shape[0] - 1 and j >= matrix.shape[1] - 1:
                    i = i + 1
                elif i >= matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                    j = j + 1
                elif i >= matrix.shape[0] - 1 and j >= matrix.shape[1] - 1:
                    break

            elif direction == 1:  # Down
                path_matrix[batch_idx, i, j] = 2
                if i < matrix.shape[0] - 1:
                    i = i + 1

            elif direction == 2:  # Right
                path_matrix[batch_idx, i, j] = 3
                if j < matrix.shape[1] - 1:
                    j = j + 1

            elif direction == 3:  # Diagonal
                path_matrix[batch_idx, i, j] = 1
                if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                    i, j = i + 1, j + 1

    return path_matrix


def makeTracerouteMatrix(matrices):
    path_matrix = build_backtrace_matrix_3d(matrices)
    path_matrix = build_forward_traceback_matrix_3d(matrices, path_matrix)
    return path_matrix


#==================================================================================================


def build_backtrace_matrix_Binary_3d(matrices):
    path_matrix = np.zeros_like(matrices)
    for batch_idx, matrix in enumerate(matrices):
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        # i, j = check_max_on_axes(matrix)
        while i >= 0 or j >= 0:
            if i > 0 and j > 0:
                direction = np.argmax(
                    np.array([0, matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]))
            elif i > 0 and j == 0:
                direction = 1
            elif i == 0 and j > 0:
                direction = 2
            else:
                direction = 0  # No Direction
            
            
            path_matrix[batch_idx, i, j] = 1
            
            if direction == 0:  # No Direction  
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1
                elif i > 0 and j <= 0:
                    i = i - 1
                elif i <= 0 and j > 0:
                    j = j - 1
                elif i <= 0 and j <= 0:
                    break

            elif direction == 1:  # Up
                if i > 0:
                    i = i - 1

            elif direction == 2:  # Left
                if j > 0:
                    j = j - 1

            elif direction == 3:  # Diagonal
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1

    return path_matrix


def build_forward_traceback_matrix_Binary_3d(matrices, path_matrix):
    for batch_idx, matrix in enumerate(matrices):
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        # i, j = check_max_on_axes(matrix)
        while i <= matrix.shape[0] - 1 or j <= matrix.shape[1] - 1:
            if i != matrix.shape[0] - 1 and j != matrix.shape[1] - 1:
                direction = np.argmax(
                    np.array([0, matrix[i + 1, j], matrix[i, j + 1], matrix[i + 1, j + 1]]))
            elif i < matrix.shape[0] - 1 and j == matrix.shape[1] - 1:
                direction = 1
            elif i == matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                direction = 2
            else:
                direction = 0  # No Direction
            

            if direction == 0:  # No Direction
                path_matrix[batch_idx, i, j] = 1
                if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                    i, j = i + 1, j + 1
                elif i < matrix.shape[0] - 1 and j >= matrix.shape[1] - 1:
                    i = i + 1
                elif i >= matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                    j = j + 1
                elif i >= matrix.shape[0] - 1 and j >= matrix.shape[1] - 1:
                    break

            elif direction == 1:  # Down
                path_matrix[batch_idx, i, j] = 1
                if i < matrix.shape[0] - 1:
                    i = i + 1

            elif direction == 2:  # Right
                path_matrix[batch_idx, i, j] = 1
                if j < matrix.shape[1] - 1:
                    j = j + 1

            elif direction == 3:  # Diagonal
                path_matrix[batch_idx, i, j] = 1
                if i < matrix.shape[0] - 1 and j < matrix.shape[1] - 1:
                    i, j = i + 1, j + 1

    return path_matrix


def makeTracerouteMatrixBinary(matrices):
    """
    matrices: should be a PyTorch tensor (not numpy array)
    """
    path_matrix = build_backtrace_matrix_Binary_3d(matrices)
    # Always pass the original PyTorch tensor as the first argument
    path_matrix = build_forward_traceback_matrix_Binary_3d(matrices, path_matrix)
    return path_matrix


#==================================================================================================

def check_max_on_axes(matrix_2d):
    rows, cols = matrix_2d.shape

    max_value_row = -1
    max_index_rows = (rows - 1, cols - 1)
    for idx in range(cols):
        val_row = matrix_2d[rows - 1, idx]
        if val_row >= max_value_row:
            max_value_row = val_row
            max_index_rows = (rows - 1, idx)
    

    max_value_col = -1
    max_index_cols = (rows - 1, cols - 1)
    for idx in range(rows):
        val_col = matrix_2d[idx, cols - 1]
        if val_col >= max_value_col:
            max_value_col = val_col
            max_index_cols = (idx, cols - 1)
    
    return max_index_rows if max_value_row >= max_value_col else max_index_cols

#==================================================================================================

def compute_traceback_path(matrix, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
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

    return path[::-1], (start_i, start_j) # Reverse to get path from start to end


def diff_SW_Path(matrices, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
    path_matrix = torch.zeros_like(matrices)
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
        
    return path_matrix, starting_points
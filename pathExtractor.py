import numpy as np


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
import numpy as np


def extract_traceback_path(matrix):
    traceback_matrix = np.zeros_like(matrix)
    i, j = check_max_on_axes(matrix)
    # i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    while i > 0 and j > 0:
        if i != 0 and j != 0:
            direction = np.argmax(
                np.array([0, matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]))
        elif i > 0 and j == 0:
            direction = np.argmax(np.array([0, matrix[i - 1, j]]))
        elif i == 0 and j > 0:
            direction = np.argmax(np.array([0, matrix[i, j - 1]]))
        else:
            direction = 0  # No Direction


        if direction == 0:  # No Direction
            traceback_matrix[i, j] = 0
            if i > 0 and j > 0:
                    i, j = i - 1, j - 1
            elif i > 0 and j < 0:
                i = i - 1
                # break
            elif i < 0 and j > 0:
                j = j - 1
            elif i < 0 and j < 0:
                break
        elif direction == 1:  # Up
            traceback_matrix[i, j] = 1
            if i > 0:
                i = i - 1
        elif direction == 2:  # Left
            traceback_matrix[i, j] = 1
            if i > 0:
                j = j - 1
        elif direction == 3:  # Diagonal
            traceback_matrix[i, j] = 1
            if i > 0 and j > 0:
                i, j = i - 1, j - 1

    return traceback_matrix



def makeTracerouteMatrix(matrices):
    matrices = matrices.detach().cpu().numpy()
    traceback_matrix = np.zeros_like(matrices)
    for batch_idx, matrix in enumerate(matrices):
        i, j = check_max_on_axes(matrix)  
        # i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        while i > 0 and j > 0:
            if i != 0 and j != 0:
                direction = np.argmax(
                    np.array([0, matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]))
            elif i > 0 and j == 0:
                direction = np.argmax(np.array([0, matrix[i - 1, j]]))
            elif i == 0 and j > 0:
                direction = np.argmax(np.array([0, matrix[i, j - 1]]))
            else:
                direction = 0  # No Direction


            if direction == 0:  # No Direction
                traceback_matrix[batch_idx, i, j] = 1
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1
                elif i > 0 and j < 0:
                    i = i - 1
                    # break
                elif i < 0 and j > 0:
                    j = j - 1
                elif i < 0 and j < 0:
                    break
            elif direction == 1:  # Up
                traceback_matrix[batch_idx, i, j] = 2
                if i > 0:
                    i = i - 1
            elif direction == 2:  # Left
                traceback_matrix[batch_idx, i, j] = 3
                if j > 0:
                    j = j - 1
            elif direction == 3:  # Diagonal
                traceback_matrix[batch_idx, i, j] = 1
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1

    return traceback_matrix


def makeTracerouteMatrixBinary(matrices):
    traceback_matrix = np.zeros_like(matrices)
    for batch_idx, matrix in enumerate(matrices):
        i, j = check_max_on_axes(matrix)
        # i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        while i > 0 and j > 0:
            if i != 0 and j != 0:
                direction = np.argmax(
                    np.array([0, matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]))
            elif i > 0 and j == 0:
                direction = np.argmax(np.array([0, matrix[i - 1, j]]))
            elif i == 0 and j > 0:
                direction = np.argmax(np.array([0, matrix[i, j - 1]]))
            else:
                direction = 0

            # print(f'i: {i}, j: {j}, direction: {direction}')
            if direction == 0:  # No Direction
                traceback_matrix[batch_idx, i, j] = 1
                if i > 0 and j > 0:
                    i, j = i - 1, j - 1
                elif i > 0 and j < 0:
                    i = i - 1
                    # break
                elif i < 0 and j > 0:
                    j = j - 1
                elif i < 0 and j < 0:
                    break
            elif direction == 1:  # Up
                traceback_matrix[batch_idx, i, j] = 1
                # if i > 0:
                i = i - 1
            elif direction == 2:  # Left
                traceback_matrix[batch_idx, i, j] = 1
                # if j > 0:
                j = j - 1
            elif direction == 3:  # Diagonal
                traceback_matrix[batch_idx, i, j] = 1
                # if i > 0 and j > 0:
                i, j = i - 1, j - 1

    return traceback_matrix


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
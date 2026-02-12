
import numpy as np
import torch

def compute_nw_score_matrix(similarity_matrix: torch.Tensor,
                            match_score: float = 3.0,
                            miss_score: float = -27.0,
                            gap_penalty: float = -10.0,
                            return_numpy: bool = True):
    """
    Compute the Needleman-Wunsch score matrix for global alignment.
    
    Args:
        similarity_matrix: A 2D tensor or numpy array of shape [N, M] containing pairwise 
                          similarity/substitution scores between sequences.
        gap_penalty: Penalty for gaps (typically negative). Default: -1.0
        return_numpy: If True, return numpy array instead of torch tensor.
        
    Returns:
        score_matrix: The NW score matrix of shape [N+1, M+1] where:
                     - First row/column contain cumulative gap penalties
                     - H[i,j] = max(H[i-1,j-1] + sim[i-1,j-1], 
                                    H[i-1,j] + gap, 
                                    H[i,j-1] + gap)
    """
    import torch
    
    # Handle inputs
    if isinstance(similarity_matrix, torch.Tensor):
        if similarity_matrix.dim() != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D")
        sim = similarity_matrix.detach().cpu().numpy()
    elif isinstance(similarity_matrix, np.ndarray):
        if len(similarity_matrix.shape) != 2:
            raise ValueError(f"Expected 2D similarity matrix, got {len(similarity_matrix.shape)}D")
        sim = similarity_matrix
    else:
        raise ValueError("similarity_matrix must be a torch.Tensor or numpy.ndarray")

    N, M = sim.shape
    
    # Initialize score matrix with extra row/column for boundary conditions
    H = np.zeros((N + 1, M + 1), dtype=np.float32)
    
    # Initialize first row and column with cumulative gap penalties
    for i in range(1, N + 1):
        H[i, 0] = i * gap_penalty
    for j in range(1, M + 1):
        H[0, j] = j * gap_penalty
    
    # Fill the score matrix using dynamic programming
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Diagonal: match/mismatch - sim matrix is 0-indexed corresponding to 1-indexed H
            score_diag = H[i - 1, j - 1] + sim[i - 1, j - 1]
            
            # Left: gap in sequence 1 (move from j-1 to j)
            score_left = H[i, j - 1] + gap_penalty
            
            # Up: gap in sequence 2 (move from i-1 to i)
            score_up = H[i - 1, j] + gap_penalty
            
            # Take maximum
            H[i, j] = max(score_diag, score_left, score_up)
    
    if return_numpy:
        return H
        
    # Convert back to tensor if input was tensor or if numpy result not explicitly requested
    return torch.from_numpy(H) 

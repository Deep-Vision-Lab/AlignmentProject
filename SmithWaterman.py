import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SmithWaterman(nn.Module):
    """
    Smith-Waterman local sequence alignment algorithm implemented in PyTorch.
    """
    
    def __init__(self, match_score=2, mismatch_penalty=-1, gap_penalty=-1):
        super(SmithWaterman, self).__init__()
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
    
    def forward(self, seq1, seq2):
        """
        Compute Smith-Waterman alignment matrix.
        
        Args:
            seq1 (torch.Tensor): First sequence tensor of shape [batch_size, seq1_len, feature_dim]
            seq2 (torch.Tensor): Second sequence tensor of shape [batch_size, seq2_len, feature_dim]
            
        Returns:
            torch.Tensor: Alignment score matrix of shape [batch_size, seq1_len+1, seq2_len+1]
        """
        batch_size, seq1_len, feature_dim = seq1.shape
        _, seq2_len, _ = seq2.shape
        
        # Initialize score matrix with zeros
        score_matrix = torch.zeros(batch_size, seq1_len + 1, seq2_len + 1, 
                                 device=seq1.device, dtype=seq1.dtype)
        
        # Compute similarity matrix between all pairs
        similarity_matrix = self.compute_similarity(seq1, seq2)
        
        # Fill the score matrix using dynamic programming
        for i in range(1, seq1_len + 1):
            for j in range(1, seq2_len + 1):
                # Get similarity score for current position
                sim_score = similarity_matrix[:, i-1, j-1]
                
                # Calculate scores for three possible moves
                diagonal = score_matrix[:, i-1, j-1] + sim_score
                up = score_matrix[:, i-1, j] + self.gap_penalty
                left = score_matrix[:, i, j-1] + self.gap_penalty
                
                # Take maximum of the three scores and 0 (local alignment)
                score_matrix[:, i, j] = torch.clamp(
                    torch.max(torch.max(diagonal, up), left), min=0
                )
        
        return score_matrix
    
    def compute_similarity(self, seq1, seq2):
        """
        Compute similarity matrix between sequences.
        
        Args:
            seq1 (torch.Tensor): First sequence [batch_size, seq1_len, feature_dim]
            seq2 (torch.Tensor): Second sequence [batch_size, seq2_len, feature_dim]
            
        Returns:
            torch.Tensor: Similarity matrix [batch_size, seq1_len, seq2_len]
        """
        # Expand dimensions for broadcasting
        seq1_expanded = seq1.unsqueeze(2)  # [batch_size, seq1_len, 1, feature_dim]
        seq2_expanded = seq2.unsqueeze(1)  # [batch_size, 1, seq2_len, feature_dim]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(seq1_expanded, seq2_expanded, dim=-1)
        
        # Convert cosine similarity to match/mismatch scores
        # If similarity > threshold, it's a match, otherwise mismatch
        threshold = 0.5
        similarity_matrix = torch.where(
            cos_sim > threshold,
            torch.tensor(self.match_score, device=seq1.device, dtype=seq1.dtype),
            torch.tensor(self.mismatch_penalty, device=seq1.device, dtype=seq1.dtype)
        )
        
        return similarity_matrix
    
    def traceback(self, score_matrix, seq1, seq2):
        """
        Perform traceback to find optimal local alignment path.
        
        Args:
            score_matrix (torch.Tensor): Score matrix from forward pass
            seq1 (torch.Tensor): First sequence
            seq2 (torch.Tensor): Second sequence
            
        Returns:
            list: List of alignment paths for each batch
        """
        batch_size = score_matrix.shape[0]
        paths = []
        
        for batch_idx in range(batch_size):
            matrix = score_matrix[batch_idx].cpu().numpy()
            
            # Find the maximum score position
            max_pos = np.unravel_index(np.argmax(matrix), matrix.shape)
            i, j = max_pos
            
            path = []
            
            # Traceback until we reach a score of 0 or boundary
            while i > 0 and j > 0 and matrix[i, j] > 0:
                path.append((i, j))
                
                # Check three possible previous positions
                diagonal = matrix[i-1, j-1] if i > 0 and j > 0 else -np.inf
                up = matrix[i-1, j] if i > 0 else -np.inf
                left = matrix[i, j-1] if j > 0 else -np.inf
                
                # Find the maximum score among the three
                max_score = max(diagonal, up, left)
                
                # Move to the position with maximum score
                # Priority: up -> left -> diagonal (as requested)
                if up == max_score:
                    i -= 1
                elif left == max_score:
                    j -= 1
                elif diagonal == max_score:
                    i -= 1
                    j -= 1
                else:
                    break
            
            # Reverse path to get start-to-end order
            paths.append(path[::-1])
        
        return paths


class SmithWatermanVisualization:
    """
    Utility class for visualizing Smith-Waterman results.
    """
    
    @staticmethod
    def plot_score_matrix(score_matrix, seq1_labels=None, seq2_labels=None, 
                         paths=None, batch_idx=0, save_path=None):
        """
        Plot the score matrix as a heatmap with optional alignment path.
        
        Args:
            score_matrix (torch.Tensor): Score matrix to visualize
            seq1_labels (list): Labels for sequence 1 (y-axis)
            seq2_labels (list): Labels for sequence 2 (x-axis)
            paths (list): Alignment paths from traceback
            batch_idx (int): Which batch to visualize
            save_path (str): Path to save the plot
        """
        matrix = score_matrix[batch_idx].detach().cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='Alignment Score')
        
        # Add labels if provided
        if seq1_labels:
            plt.yticks(range(len(seq1_labels) + 1), [''] + seq1_labels)
        if seq2_labels:
            plt.xticks(range(len(seq2_labels) + 1), [''] + seq2_labels)
        
        plt.xlabel('Sequence 2')
        plt.ylabel('Sequence 1')
        plt.title(f'Smith-Waterman Score Matrix - Batch {batch_idx}')
        
        # Plot alignment path if provided
        if paths and len(paths) > batch_idx:
            path = paths[batch_idx]
            if path:
                path_y = [p[0] for p in path]
                path_x = [p[1] for p in path]
                plt.plot(path_x, path_y, 'r-o', linewidth=3, markersize=6,
                        alpha=0.8, label='Optimal Alignment Path')
                plt.legend()
        
        # Add values to cells for small matrices
        if matrix.shape[0] <= 20 and matrix.shape[1] <= 20:
            for (i, j), val in np.ndenumerate(matrix):
                plt.text(j, i, f"{val:.1f}", ha='center', va='center',
                        color='white' if val < matrix.mean() else 'black', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# Example usage and testing
if __name__ == "__main__":
    # Create sample sequences (random embeddings)
    batch_size = 2
    seq1_len, seq2_len = 10, 8
    feature_dim = 64
    
    # Generate random sequence embeddings
    seq1 = torch.randn(batch_size, seq1_len, feature_dim)
    seq2 = torch.randn(batch_size, seq2_len, feature_dim)
    
    # Make some positions similar for demonstration
    seq1[:, 2:4, :] = seq2[:, 3:5, :] + 0.1 * torch.randn_like(seq2[:, 3:5, :])
    seq1[:, 6:8, :] = seq2[:, 1:3, :] + 0.1 * torch.randn_like(seq2[:, 1:3, :])
    
    # Initialize Smith-Waterman
    sw = SmithWaterman(match_score=3, mismatch_penalty=-1, gap_penalty=-2)
    
    # Compute alignment
    score_matrix = sw(seq1, seq2)
    
    # Perform traceback
    paths = sw.traceback(score_matrix, seq1, seq2)
    
    # Visualize results
    visualizer = SmithWatermanVisualization()
    
    for batch_idx in range(batch_size):
        print(f"\nBatch {batch_idx}:")
        print(f"Score matrix shape: {score_matrix[batch_idx].shape}")
        print(f"Max score: {score_matrix[batch_idx].max().item():.2f}")
        print(f"Alignment path length: {len(paths[batch_idx])}")
        
        # Plot the score matrix with alignment path
        visualizer.plot_score_matrix(
            score_matrix, 
            paths=paths, 
            batch_idx=batch_idx,
            save_path=f'smith_waterman_batch_{batch_idx}.png'
        )
    
    print(f"\nAlignment complete! Score matrix shape: {score_matrix.shape}")

import torch
import torch.utils.dlpack
import jax
import jax.dlpack
from matplotlib import pyplot as plt
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import numpy as np


# A generic mechanism for turning a JAX function into a PyTorch function.

def sw_simple(batch=True, unroll=2):
    '''smith-waterman (local alignment) with no gap, no temp, no mask'''

    # rotate to vectorize
    def sw_rotate(x, mask=None):
        # solution from jake vanderplas (thanks!)
        a, b = x.shape
        ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        zero = jnp.zeros([n, m])
        if mask is None: mask = 1.0
        output = {"x": zero.at[i, j].set(x),
                  "m": zero.at[i, j].set(mask),
                  "o": (jnp.arange(n) + a % 2) % 2}
        prev = (jnp.zeros(m), jnp.zeros(m))
        return output, prev, (i, j)

    # comute scoring (hij) matrix
    def sw_sco(x):
        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _step(prev, sm):
            h2, h1 = prev  # previous two rows of scoring (hij) mtx
            h1_T = _cond(sm["o"], jnp.pad(h1[:-1], [1, 0]), jnp.pad(h1[1:], [0, 1]))
            h0 = sm["m"] * jax.nn.logsumexp(jnp.stack([h2 + sm["x"], h1, h1_T]), 0)
            return (h1, h0), h0

        sm, prev, idx = sw_rotate(x)
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
        return hij.max()

    # traceback (aka backprop) to get alignment
    traceback = jax.grad(sw_sco)

    # add batch dimensionW
    if batch:
        return jax.vmap(traceback)
    else:
        return traceback    
def sw_with_gap(batch=True, unroll=2, gap_penalty=-1):
    '''smith-waterman (local alignment) with gap support'''

    # rotate to vectorize
    def sw_rotate(x, mask=None):
        # solution from jake vanderplas (thanks!)
        a, b = x.shape
        ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        # jax.debug.print("{x}", x=i)
        # jax.debug.print("{x}", x=j)
        n, m = (a + b - 1), (a + b) // 2
        # print(f'n: {n}')
        # print(f'm: {m}')
        zero = jnp.zeros([n, m])
        if mask is None: mask = 1.0
        output = {
            "x": zero.at[i, j].set(x),  # Set values in the alignment matrix
            "m": zero.at[i, j].set(mask),  # Set mask values
            "o": (jnp.arange(n) + a % 2) % 2  # For alternating row shifts
        }
        # print(f'output: {output["x"].shape}')
        # jax.debug.print("{x}",x=output['m'])
        prev = (jnp.zeros(m), jnp.zeros(m))  # Initial previous values
        return output, prev, (i, j)

    # compute scoring (hij) matrix
    def sw_sco(x):
        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _step(prev, sm):
            h2, h1 = prev  # previous two rows of scoring (hij) mtx

            # Gap handling: introduce a gap penalty
            h1_T = _cond(sm["o"], jnp.pad(h1[:-1], [1, 0]), jnp.pad(h1[1:], [0, 1]))

            # Align: normal diagonal movement with no gap
            align = h2 + sm["x"]  # Alignment score (match/mismatch)

            # Horizontal/vertical moves with gap penalty
            insert = h1 + gap_penalty  # Insertion (moving vertically)
            delete = h1_T + gap_penalty  # Deletion (moving horizontally)

            # Take the maximum of alignment, insert, and delete
            # jax.debug.print("align: {align}")
            h0 = sm["m"] * jax.nn.logsumexp(jnp.stack([align, insert, delete]), 0)
            return (h1, h0), h0

        # Apply the rotate function and calculate the score
        sm, prev, idx = sw_rotate(x)
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
        return hij.max()

    # traceback (aka backprop) to get alignment
    traceback = jax.grad(sw_sco)

    # add batch dimension
    if batch:
        return jax.vmap(traceback)
    else:
        return traceback



def j2t(x_jax):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))

def t2j(x_torch):
    x_torch = x_torch.contiguous()  # Ensure tensor is contiguous for conversion
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))

def jax2torch(fun):
  class JaxFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
      y_, ctx.fun_vjp = jax.vjp(fun, t2j(x))
      return j2t(y_)

    @staticmethod
    def backward(ctx, grad_y):
      grad_x_, = ctx.fun_vjp(t2j(grad_y))
      return j2t(grad_x_),

  return JaxFun.apply



###########################################################################
# Cosine Similarity Layer

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLayer(nn.Module):
    def __init__(self, matchscore, missscore):
        super(CosineSimilarityLayer, self).__init__()
        self.matchscore = matchscore
        self.missscore = missscore
        # self.powerbase = 3 # This was defined in your code but not used.
                           # If you intend to use it, ensure it's part of the computation.

    def forward(self, x1, x2):
        # Ensure input tensors have the same feature dimension (last dimension)
        # Original check was x1.shape[1] (sequence length), which might not be the intent.
        if x1.shape[2] != x2.shape[2]: 
            raise ValueError(
                f"Input tensors must have the same feature dimension (last dim). "
                f"x1.shape: {x1.shape}, x2.shape: {x2.shape}"
            )

        # Compute the dot product (batch matrix multiplication)
        # x1: [B, N, D], x2: [B, M, D]
        # x2.transpose(1, 2) gives [B, D, M]
        # torch.bmm([B, N, D], [B, D, M]) results in [B, N, M]
        # This correctly computes dot products over the feature dimension D
        # for each pair of vectors from sequence N and sequence M.
        dot_product = torch.bmm(x1, x2.transpose(1, 2))  # shape (batch_size, N, M)
        
        # Compute the magnitudes of the vectors (norm over feature dimension D)
        magnitude_x1 = torch.norm(x1, dim=2, keepdim=True)  # shape (batch_size, N, 1)
        magnitude_x2 = torch.norm(x2, dim=2, keepdim=True)  # shape (batch_size, M, 1)

        # Calculate cosine similarity
        # magnitude_x1: [B, N, 1]
        # magnitude_x2.transpose(1, 2): [B, 1, M] (after transposing [B, M, 1])
        # Denominator broadcasts to [B, N, M]
        denominator = magnitude_x1 * magnitude_x2.transpose(1, 2) + 1e-8
        cosine_similarity = dot_product / denominator  # shape (batch_size, N, M)

        # Compute L2 distance
        # x1.unsqueeze(2): [B, N, 1, D]
        # x2.unsqueeze(1): [B, 1, M, D]
        # diff: [B, N, M, D] (broadcasted difference between each vector pair)
        diff = x1.unsqueeze(2) - x2.unsqueeze(1)
        # l2_distance: Euclidean distance for each pair, shape [B, N, M]
        l2_distance = torch.norm(diff, dim=3)

        # Normalize L2 distance to roughly [0, 1]
        # l2_distance.max() is a scalar tensor. Gradient flows through this.
        l2_distance_max = l2_distance.max() 
        normalized_distance = l2_distance / (l2_distance_max + 1e-10) # Avoid division by zero

        # Invert normalized distance to create length similarity
        length_contribution = 1 - normalized_distance

        # Combine cosine similarity with length contribution
        weight_cosine = 1
        weight_length = 0
        combined_similarity = weight_cosine * cosine_similarity + weight_length * length_contribution
  
        # Transform similarity to a probability (ensuring it's in [0,1] range)
        # Cosine sim: [-1, 1], Length contrib: [0, 1] (approx)
        # Combined: (0.7*[-1,1]) + (0.3*[0,1]) = [-0.7, 0.7] + [0, 0.3] = [-0.7, 1.0]
        # (combined_similarity + 1) / 2 maps [-0.7, 1.0] to [0.15, 1.0], which is a valid prob range.
        similarity_prob = (combined_similarity + 1) / 2

        # Calculate dissimilarity as the complement of similarity
        dissimilarity_prob = 1 - similarity_prob

        # Stack similarity and dissimilarity probabilities for softmax
        scores = torch.stack([similarity_prob, dissimilarity_prob], dim=-1) # shape (B, N, M, 2)

        # Apply softmax along the last dimension
        softmax_scores = F.softmax(scores, dim=-1)

        # Extract similarity and dissimilarity from softmax results
        similarity_softmaxed = softmax_scores[..., 0]
        dissimilarity_softmaxed = softmax_scores[..., 1]

        # Multiply by matchscore and missscore
        match_score_contribution = similarity_softmaxed * self.matchscore
        miss_score_contribution = dissimilarity_softmaxed * self.missscore # missscore is typically negative

        # Sum match and miss scores to get the final score
        final_score = match_score_contribution + miss_score_contribution

        return final_score
    


###########################################################################
# Alignment Algorithm

class Alignment(nn.Module):
    def __init__(self, match_score, miss_score):
        super(Alignment, self).__init__()
        self.match_score = match_score
        self.miss_score = miss_score
        self.cosine_similarity_layer = CosineSimilarityLayer(matchscore= self.match_score,
                                                             missscore=self.miss_score )
        self.sw_fn_torch = jax2torch(jax.jit(sw_with_gap())) 
        
        # self.sw_fn_torch = jax2torch(jax.jit(sw_simple()))
    def forward(self, x1=None, x2=None, calc_output=None, calc_cosine=True):
        if calc_cosine:
            self.output = self.cosine_similarity_layer(x1, x2)  
            new_output = torch.squeeze(self.output, dim=0)
            # visualize_heatmap_with_values(new_output[0], title="Cosine Similarity Heatmap")
            self.align = self.sw_fn_torch(new_output)
            # visualize_heatmap_with_values(self.align[0], title="Alignment Heatmap")
        else:
            self.align = self.sw_fn_torch(calc_output)
        return self.align

###########################################################################
# Test
def visualize_heatmap_with_values(tensor, title="Heatmap", cmap="viridis"):
    arr = tensor.detach().cpu().numpy()
    plt.figure(figsize=(30, 20))
    plt.imshow(arr, cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    # Show values in each cell
    for (i, j), val in np.ndenumerate(arr):
        plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=8)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


if __name__ == '__main__':
    # Example usage - output CNN or transformer
    # # Create tensors with 4 consecutive nonzero elements, rest zeros
    x1 = torch.ones(2, 20, 512)
    x2 = torch.ones(2, 20, 512)
    # # Set elements 8-11 to random values for both x1 and x2
    # x1[:, 0:2, :] = torch.rand(2, 2, 512)
    # x2[:, 0:2, :] = torch.rand(2, 2, 512)
    # # Set elements 8-11 to random values for both x1 and x2
    # x1[:, 7:9, :] = torch.rand(2, 2, 512)
    # x2[:, 6:10, :] = torch.rand(2, 4, 512)
    # # Set elements 8-11 to random values for both x1 and x2
    # x1[:, 16:18, :] = torch.rand(2, 2, 512)
    # x2[:, 17:19, :] = torch.rand(2, 2, 512)
    x2[:, 0:18, :] = torch.zeros(2, 18, 512)
    x1.requires_grad = True
    x2.requires_grad = True
    # x1.data *= 2 
    # x2.data *= 0.5
    # Create the Cosine Similarity layer
    alignment = Alignment(match_score=7, miss_score=-7)
    # Get the cosine similarity output
    output = alignment(x1, x2)
    print(output.shape)
    
    # Visualize the output as a heatmap
    output_np = output.detach().cpu().numpy()


    # Function to compute traceback path
    def compute_traceback_path(matrix):
        """Compute the optimal alignment path using traceback"""
        # Find the maximum score position as starting point
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
        path = []
        
        while i > 0 and j > 0 and matrix[i, j] > 0:
            path.append((i, j))
            
            # Check diagonal, up, and left moves
            diag_score = matrix[i-1, j-1] if i > 0 and j > 0 else -np.inf
            up_score = matrix[i-1, j] if i > 0 else -np.inf
            left_score = matrix[i, j-1] if j > 0 else -np.inf
            
            # Find the maximum score
            max_score = max(up_score, left_score, diag_score)
            
            # Priority: up -> right (left) -> diagonal when scores are equal
            if up_score == max_score:
                i -= 1
            elif left_score == max_score:
                j -= 1
            elif diag_score == max_score:
                i -= 1
                j -= 1
                
        return path[::-1]  # Reverse to get path from start to end
    
    for batch_idx in range(output_np.shape[0]):
        plt.figure(figsize=(12, 8))
        plt.imshow(output_np[batch_idx], cmap='viridis', aspect='auto')
        plt.colorbar(label='Alignment Score')
        plt.title(f'Alignment Output Heatmap with Traceback Path - Batch {batch_idx}')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        
        # Compute and plot the traceback path
        path = compute_traceback_path(output_np[batch_idx])
        if path:
            path_y = [p[0] for p in path]  # Row indices
            path_x = [p[1] for p in path]  # Column indices
            plt.plot(path_x, path_y, color='red', linewidth=3, marker='o', 
                    markersize=6, alpha=0.8, label='Optimal Alignment Path')
            plt.legend()
        
        # Add values to each cell for better readability (skip for large matrices)
        if output_np[batch_idx].shape[0] <= 20 and output_np[batch_idx].shape[1] <= 20:
            for (i, j), val in np.ndenumerate(output_np[batch_idx]):
                plt.text(j, i, f"{val:.4f}", ha='center', va='center', 
                        color='white' if val < output_np[batch_idx].mean() else 'black', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'alignment_output_heatmap_with_path_batch_{batch_idx}.png', dpi=150)
        plt.close() 
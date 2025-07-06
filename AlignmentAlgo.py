import torch
import torch.utils.dlpack
import jax
import jax.dlpack
from matplotlib import pyplot as plt
import jax.numpy as jnp 


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
        n, m = (a + b - 1), (a + b) // 2
        zero = jnp.zeros([n, m])
        if mask is None: mask = 1.0
        output = {
            "x": zero.at[i, j].set(x),  # Set values in the alignment matrix
            "m": zero.at[i, j].set(mask),  # Set mask values
            "o": (jnp.arange(n) + a % 2) % 2  # For alternating row shifts
        }
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
    def forward(self, x1, x2):      
        self.output = self.cosine_similarity_layer(x1, x2)  
        new_output = torch.squeeze(self.output, dim=0)
        self.align = self.sw_fn_torch(new_output)
        return self.align

###########################################################################
# Test

if __name__ == '__main__':
    # Example usage - output CNN or transformer
    x1 = torch.rand(8, 20, 512, requires_grad=True)  # Random tensor for the first input
    x2 = torch.rand(8, 20, 512, requires_grad=True)  # Random tensor for the second input
    # x1.data *= 2 
    # x2.data *= 0.5
    # Create the Cosine Similarity layer
    cosine_similarity_layer = CosineSimilarityLayer(matchscore=3, missscore=-3)
    # Get the cosine similarity output
    output = cosine_similarity_layer(x1, x2)
    y = output.sum()
    y.backward()
    # print(output)
    print(f'x1 gradient: {x1.grad.sum().item()}')
    print(f'x2 gradient: {x2.grad.sum().item()}')
    
    exit(0)
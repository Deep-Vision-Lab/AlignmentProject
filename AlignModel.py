# import torch
# import torch.nn as nn
#
#
# class ShiftLayer(nn.Module):
#     def __init__(self):
#         super(ShiftLayer, self).__init__()
#
#     def forward(self, x, row_shift=0, col_shift=0):
#         # Get the shape of the input tensor
#
#         # Create an output tensor initialized to zeros
#         output = torch.zeros_like(x)
#
#         # Shift rows
#         if row_shift > 0:
#             output[:, row_shift:, :] = x[:, :-row_shift, :]
#         elif row_shift < 0:
#             output[:, :row_shift, :] = x[:, -row_shift:, :]
#
#         # Shift columns
#         if col_shift > 0:
#             output[:, :, col_shift:] = x[:, :, :-col_shift]
#         elif col_shift < 0:
#             output[:, :, :col_shift] = x[:, :, -col_shift:]
#
#         # Return output tensor which retains gradient information
#         return output
#
# # Example usage
# x = torch.tensor(torch.randn(1,50,50), requires_grad=True)
#
# # Create a shift layer
# shift_layer = ShiftLayer()

import torch
import torch.utils.dlpack
import jax
import jax.dlpack
# import sw_fn as sw
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

    # add batch dimension
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
    x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
    return x_torch


def t2j(x_torch):
    x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
    x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
    return x_jax


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


#
# sw_fn_torch = jax2torch(jax.jit(sw_simple()))
# x = torch.ones((2,10,10),requires_grad=True)
# sw_fn_torch(x)
# path_torch = torch.sum(sw_fn_torch(x) - torch.randn(size=(1,100,100)))
# path_torch.backward()
# print(torch.unique(path_torch))
# print(x.grad)
# for i in range(0,2):
#     plt.imshow(torch.log(path_torch + 1e-8)[i].detach().cpu())
#     plt.show()
#
# print()

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLayer(nn.Module):
    def __init__(self, matchscore, missscore):
        super(CosineSimilarityLayer, self).__init__()
        self.matchscore = matchscore
        self.missscore = missscore
        self.powerbase = 3

    def forward(self, x1, x2):
        # Ensure input tensors have the same feature dimension
        if x1.shape[2] != x2.shape[2]:
            raise ValueError("Input tensors must have the same feature dimension.")

        # Compute the dot product (batch matrix multiplication)
        dot_product = torch.bmm(x1, x2.transpose(1, 2))  # shape (batch_size, 20, 20)

        # Compute the magnitudes of the vectors
        magnitude_x1 = torch.norm(x1, dim=2, keepdim=True)  # shape (batch_size, 20, 1)

        magnitude_x2 = torch.norm(x2, dim=2, keepdim=True)  # shape (batch_size, 20, 1)

        # Calculate cosine similarity
        cosine_similarity = dot_product / (
                    magnitude_x1 * magnitude_x2.transpose(1, 2) + 1e-8)  # shape (batch_size, 20, 20)

        # Compute L2 distance (length-based)
        diff = x1.unsqueeze(2) - x2.unsqueeze(1)  # shape (batch_size, 20, 20, feature_dim)
        l2_distance = torch.norm(diff, dim=3)  # shape (batch_size, 20, 20)

        # Normalize L2 distance to [0, 1]
        l2_distance_max = l2_distance.max()  # Get the maximum distance to normalize
        normalized_distance = l2_distance / (l2_distance_max + 1e-10)  # Avoid division by zero

        # Invert normalized distance to create length similarity (larger similarity for closer lengths)
        length_contribution = 1 - normalized_distance  # shape (batch_size, 20, 20)

        # Combine cosine similarity with length contribution
        # You can adjust the weight (e.g., 0.5 and 0.5) as needed for your use case
        weight_cosine = 0.7  # weight for direction (cosine similarity)
        weight_length = 0.3  # weight for length (length contribution)
        combined_similarity = weight_cosine * cosine_similarity + weight_length * length_contribution

        # Transform similarity to a probability between 0 and 1
        similarity_prob = (combined_similarity + 1) / 2  # shape (batch_size, 20, 20)
        # Calculate dissimilarity as the complement of similarity
        dissimilarity_prob = 1 - similarity_prob  # shape (batch_size, 20, 20)

        # Stack similarity and dissimilarity probabilities for softmax
        #  self.powerbase is to increase the error between both of them
        scores = torch.stack([similarity_prob, dissimilarity_prob], dim=-1)  # shape (batch_size, 20, 20, 2)

        # Apply softmax along the last dimension
        softmax_scores = F.softmax(scores, dim=-1)  # shape (batch_size, 20, 20, 2)

        # Extract similarity and dissimilarity from softmax results
        similarity_softmaxed = softmax_scores[..., 0]  # shape (batch_size, 20, 20)
        dissimilarity_softmaxed = softmax_scores[..., 1]  # shape (batch_size, 20, 20)

        # Multiply by matchscore and missscore
        match_score = similarity_softmaxed * self.matchscore  # shape (batch_size, 20, 20)
        miss_score = dissimilarity_softmaxed * self.missscore  # shape (batch_size, 20, 20)

        # Sum match and miss scores to get the final score
        final_score = match_score + miss_score  # shape (batch_size, 20, 20)

        return final_score


# Example usage - output CNN or transformer
x1 = torch.ones(8, 20, 512, requires_grad=True)  # Random tensor for the first input
x2 = 0.5 * torch.ones(8, 20, 512, requires_grad=True)  # Random tensor for the second input


# # Create the Cosine Similarity layer
# cosine_similarity_layer = CosineSimilarityLayer(matchscore=3, missscore=-3)
# # # Get the cosine similarity output
# output = cosine_similarity_layer(x1, x2)
# print(output)
#
# exit(0)


class Alignment(nn.Module):
    def __init__(self, match_score, miss_score):
        super(Alignment, self).__init__()
        self.match_score = match_score
        self.miss_score = miss_score
        self.cosine_similarity_layer = CosineSimilarityLayer(matchscore=self.match_score,
                                                             missscore=self.miss_score)

        self.sw_fn_torch = jax2torch(jax.jit(sw_with_gap()))
        # self.sw_fn_torch = jax2torch(jax.jit(sw_simple()))

    def forward(self, x1, x2):
        self.output = self.cosine_similarity_layer(x1, x2)
        self.align = self.sw_fn_torch(self.output)
        return self.align


def diff_smith_waterman(alignmentModel, seq1, seq2, padding=0):
    output = alignmentModel(seq1, seq2)
    pad = (padding - output.shape[2], 0, padding - output.shape[1], 0, 0, 0)
    output = torch.nn.functional.pad(output, pad, mode='constant', value=0)  # Pad with 1 on each side
    return output

# if __name__ == '__main__':
#
#     # Example usage
#     x1 = torch.ones(4, 5, 512, requires_grad=True)  # Random tensor for the first input
#     print(x1.grad)
#     x2 = torch.ones(4, 7, 512, requires_grad=True)  # Random tensor for the second input
#     print(x2.grad)
#     alignmentModel = Alignment(match_score=3, miss_score=-6)
#     output = alignmentModel(x1, x2)
#     gt = torch.ones((4,6,6))
#     print(output.shape)
#     output = torch.nn.functional.pad(output, pad=(1, 0, 1, 0, 0, 0))  # Pad with 1 on each side
#     print(output.shape)
#     y = torch.sum(torch.abs(output-gt))
#     print(y.grad_fn)
#     y.backward()
#
#     print(x1.grad)
#     print(x2.grad)

# print(output)
#
#
#
# # we should take two lines , split them to (8,n,dimVector) , dimVector = 512

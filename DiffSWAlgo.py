# Set JAX to use 32-bit precision for faster compilation


import gc
import os
from typing import Optional

import jax
import jax.dlpack
import jax.numpy as jnp 

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from Parameters import *
from pathExtractor import diff_SW_Path, SW_Path
try:
    import psutil  # optional, used for memory logging
except Exception:
    psutil = None

os.environ['JAX_ENABLE_X64'] = 'False'
os.environ['XLA_FLAGS'] = "--xla_dump_to=/tmp/foo"

# A generic mechanism for turning a JAX function into a PyTorch function.

def sw_simple(batch=True, unroll=2):
    '''smith-waterman (local alignment) with no gap, no temp, no mask'''

    # rotate to vectorize
    def sw_rotate(x, mask=None):
        # solution from jake vanderplas (thanks!)
        a, b = x.shape
        ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        del ar, br  # Clean up intermediate arrays
        n, m = (a + b - 1), (a + b) // 2
        zero = jnp.zeros([n, m])
        if mask is None: mask = 1.0
        output = {
            "x": zero.at[i, j].set(x), 
            "m": zero.at[i, j].set(mask), # Mask Tensor
            "o": (jnp.arange(n) + a % 2) % 2
        }
        print('output[x] shape:', output["x"].shape)
        jax.debug.print("{x}",x=output['x'])
        print('output[m] shape:', output["m"].shape)
        jax.debug.print("{x}",x=output['m'])
        print('output[o] shape:', output["o"].shape)
        jax.debug.print("{x}",x=output['o'])


        del zero  # Clean up zero array after use
        prev = (jnp.zeros(m), jnp.zeros(m))
        return output, prev, (i, j)

    # comute scoring (hij) matrix
    def sw_sco(x):
        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _step(prev, sm):
            h2, h1 = prev  # previous two rows of scoring (hij) mtx
            h1_T = _cond(sm["o"], jnp.pad(h1[:-1], [1, 0]), jnp.pad(h1[1:], [0, 1]))
            stacked_values = jnp.stack([h2 + sm["x"], h1, h1_T])
            h0 = sm["m"] * jax.nn.logsumexp(stacked_values, 0)
            del h1_T, stacked_values  # Clean up intermediate calculations
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

        # Hold element [a,b] is the index in the rotated matrix
        # i[a,b] = m & j[a,b] = n => rotated_matrix[m,n] = original[a,b]
        i, j = (br - ar) + (a - 1), (ar + br) // 2

        n, m = (a + b - 1), (a + b) // 2
        zero = jnp.zeros([n, m])
        if mask is None: mask = 1.0
        output = {
            "x": zero.at[i, j].set(x),  # Rotated matrix with the similarity weigts
            "m": zero.at[i, j].set(mask),  # Mask tensor where are we going to put the elements in the rotated matrix
            "o": (jnp.arange(n) + a % 2) % 2  # For alternating row shifts
        }
        print('output[x] shape:', output["x"].shape)
        jax.debug.print("{x}",x=output['x'])
        print('output[m] shape:', output["m"].shape)
        jax.debug.print("{x}",x=output['m'])
        print('output[o] shape:', output["o"].shape)
        jax.debug.print("{x}",x=output['o'])
        
        prev = (jnp.zeros(m), jnp.zeros(m))  # Initial previous values
        
        print('prev[0] shapes:', prev[0].shape)
        jax.debug.print("prev[0]: {x}", x=prev[0])
        print('prev[1] shapes:', prev[1].shape)
        jax.debug.print("prev[1]: {x}", x=prev[1])
        
        return output, prev, (i, j)

    # compute scoring (hij) matrix
    def sw_sco(x):
        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _step(prev, sm):
            h2, h1 = prev  # previous two rows of scoring (hij) matrix

            # Gap handling: introduce a gap penalty
            h1_T = _cond(sm["o"], jnp.pad(h1[:-1], [1, 0]),
                            jnp.pad(h1[1:], pad_width=[0, 1]))

            # Align: normal diagonal movement with no gap
            align = h2 + sm["x"]  # Alignment score (match/mismatch)

            # Horizontal/vertical moves with gap penalty
            insert = h1 + gap_penalty  # Insertion (moving vertically)
            delete = h1_T + gap_penalty  # Deletion (moving horizontally)

            # Take the maximum of alignment, insert, and delete
            
            print('align shape:', align.shape)
            jax.debug.print("align: {x}", x=align)
            print('insert shape:', insert.shape)
            jax.debug.print("insert: {x}", x=insert)
            print('delete shape:', delete.shape)
            jax.debug.print("delete: {x}", x=delete)
            
            stacked_scores = jnp.stack([align, insert, delete])
            print('stacked_scores shape:', stacked_scores.shape)
            jax.debug.print("stacked_scores: {x}", x=stacked_scores)
            h0 = sm["m"] * jax.nn.logsumexp(stacked_scores, 0)
            # h0 = sm["m"] * stacked_scores
            del align, insert, delete, h1_T, stacked_scores  # Clean up intermediate calculations
            return (h1, h0), h0

        # Apply the rotate function and calculate the score
        sm, prev, idx = sw_rotate(x)
        hij = jax.lax.scan(_step, prev, sm, unroll=unroll)[-1][idx]
        
        # print('hij shape:', hij.shape)
        # jax.debug.print("{x}",x=hij)

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
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch)) # type: ignore

def jax2torch(fun):
  class JaxFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
      y_, ctx.fun_vjp = jax.vjp(fun, t2j(x))
      return j2t(y_)

    @staticmethod
    def backward(ctx, grad_y): # type: ignore
      grad_x_, = ctx.fun_vjp(t2j(grad_y))
      return j2t(grad_x_)

  return JaxFun.apply


# Memory / RAM helper -----------------------------------------------------
def log_memory(prefix: str = "", log_file: str | None = None):
    """Print RAM and GPU memory usage with a short prefix.

    - Uses `psutil` if available to report system RAM.
    - Uses `torch.cuda` APIs if CUDA is available to report GPU memory.
    - Always prints to stdout (use `python -u` or `flush=True` when running under batch systems).
    - If `log_file` is provided, appended to that file.
    """
    try:
        lines = []
        # System RAM
        if psutil is not None:
            vm = psutil.virtual_memory()
            lines.append(f"RAM: {vm.used // (1024**2)}MB/{vm.total // (1024**2)}MB ({vm.percent}%)")
        else:
            lines.append("RAM: psutil unavailable")

        # Python memory objects (rough indicator)
        try:
            objs = gc.get_objects()
            lines.append(f"Python objects: {len(objs)}")
        except Exception:
            lines.append("Python objects: n/a")

        # GPU memory
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(dev) // (1024**2)
                reserved = torch.cuda.memory_reserved(dev) // (1024**2)
                max_alloc = torch.cuda.max_memory_allocated(dev) // (1024**2)
                lines.append(f"CUDA device {dev}: allocated={allocated}MB reserved={reserved}MB max_alloc={max_alloc}MB")
            except Exception as e:
                lines.append(f"CUDA stats error: {e}")
        else:
            lines.append("CUDA: not available")

        msg = f"[MEM] {prefix} | " + " | ".join(lines)
        print(msg, flush=True)
        if log_file is not None:
            try:
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
    except Exception as e:
        # Be robust: don't crash the main program if memory logging fails
        print(f"[MEM] logging failed ({e})", flush=True)


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
        x2_transposed = x2.transpose(1, 2)
        dot_product = torch.bmm(x1, x2_transposed)  # shape (batch_size, N, M)
        del x2_transposed
        
        # Compute the magnitudes of the vectors (norm over feature dimension D)
        magnitude_x1 = torch.norm(x1, dim=2, keepdim=True)  # shape (batch_size, N, 1)
        magnitude_x2 = torch.norm(x2, dim=2, keepdim=True)  # shape (batch_size, M, 1)

        # Calculate cosine similarity
        # magnitude_x1: [B, N, 1]
        # magnitude_x2.transpose(1, 2): [B, 1, M] (after transposing [B, M, 1])
        # Denominator broadcasts to [B, N, M]
        magnitude_x2_transposed = magnitude_x2.transpose(1, 2)
        denominator = magnitude_x1 * magnitude_x2_transposed + 1e-8
        del magnitude_x1, magnitude_x2, magnitude_x2_transposed
        
        cosine_similarity = dot_product / denominator  # shape (batch_size, N, M)
        del dot_product, denominator

        comp_cosine_similarity = 1 - cosine_similarity

        # Multiply by matchscore and missscore
        match_score_contribution = cosine_similarity * self.matchscore
        miss_score_contribution = comp_cosine_similarity * self.missscore # missscore is typically negative
        del cosine_similarity, comp_cosine_similarity

        # Sum match and miss scores to get the final score
        final_score = match_score_contribution + miss_score_contribution
        del match_score_contribution, miss_score_contribution

        return final_score
    


###########################################################################
# Differentiable Smith-Waterman Algorithm

class DiffSWAlgo(nn.Module):
    def __init__(self, match_score, miss_score,gap=-1):
        super(DiffSWAlgo, self).__init__()
        self.match_score = match_score
        self.miss_score = miss_score
        self.cosine_similarity_layer = CosineSimilarityLayer(matchscore= match_score,
                                                             missscore=miss_score)
        # self.cosine_similarity_layer = nn.CosineSimilarity(dim=1, eps=1e-6)
        # Use non-batched SW because we pass a 2D [N,M] similarity matrix
        self.sw_fn_torch = jax2torch(jax.jit(sw_with_gap(gap_penalty=gap)))

        # self.sw_fn_torch = jax2torch(jax.jit(sw_simple()))
    
    def reset_cosine_similarity(self):
        self.cosine_similarity = None
        
    def forward(self, 
                x1: Optional[torch.Tensor] = None, 
                x2: Optional[torch.Tensor] = None, 
                similarity_matrix = None, 
                calc_cosine = True,
                show_dims = False):
        if show_dims and x1 is not None and x2 is not None:
            print(f"x1 shape: {x1.shape}")
            print(f"x2 shape: {x2.shape}")

        if calc_cosine:
            self.cosine_similarity = self.cosine_similarity_layer(x1, x2)
            if show_dims:
                print(f"Cosine similarity shape: {self.cosine_similarity.shape}")
            new_output = torch.squeeze(self.cosine_similarity, dim=0)
            if show_dims:
                print(f"Squeezed cosine similarity shape: {new_output.shape}")
            # visualize_heatmap_with_values(new_output[0], title="Cosine Similarity Heatmap")
            self.align = self.sw_fn_torch(new_output)
            if show_dims:
                print(f"align shape: {self.align.shape}")
        else:
            self.cosine_similarity = similarity_matrix
            if show_dims:
                print(f"Cosine similarity shape: {self.cosine_similarity.shape}")
            self.align = self.sw_fn_torch(self.cosine_similarity)
            if show_dims:
                print(f"align shape: {self.align.shape}")
        
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

# Helper to annotate numeric values on every cell of a numpy 2D array
def annotate_all(arr: np.ndarray, fmt: str = ".4f", fontsize: int = 6):
    for (i, j), val in np.ndenumerate(arr):
        color = 'white' if val < arr.mean() else 'black'
        plt.text(j, i, f"{val:{fmt}}", ha='center', va='center', color=color, fontsize=fontsize)


if __name__ == '__main__':
    # Example usage - convert two sentences to ASCII one-hot vectors and align

    def sentence_to_ascii_onehot(s: str, vocab_size: int = 128) -> torch.Tensor:
        """Convert a sentence to a [len(s), vocab_size] one-hot matrix by ASCII code.

        - Non-ASCII chars (>= vocab_size) are mapped to index 0.
        - Returns float32 tensor suitable as features (D = vocab_size).
        """
        v = torch.zeros(len(s), vocab_size, dtype=torch.float32)
        for i, ch in enumerate(s):
            code = ord(ch)
            if code >= vocab_size:
                code = 0
            v[i, code] = 1.0
        return v

    # Sentences to align (edit as needed)
    sentence1 = "HELLOAERLD"
    sentence2 = "HELLOWORLD"

    # Build per-character ASCII one-hot feature sequences: [N, D] and [M, D]
    vec1 = sentence_to_ascii_onehot(sentence1)  # [N, 128]
    vec2 = sentence_to_ascii_onehot(sentence2)  # [M, 128]

    # Expand to batched tensors expected by the model: [B, N, D], [B, M, D]
    # Use B=1 to keep shapes simple and consistent here
    x1 = vec1.unsqueeze(0)
    x2 = vec2.unsqueeze(0)

    x1.requires_grad = True
    x2.requires_grad = True
    # Create the Cosine Similarity layer
    # cosin_layer = CosineSimilarityLayer(matchscore=1, missscore=-1)
    # # Get the cosine similarity output
    # output = cosin_layer(x1, x2)
    # print(output.shape)
    # # Visualize the output as a heatmap
    # # output_np = output.detach().cpu().numpy()
    # visualize_heatmap_with_values(output[0], title="Cosine Similarity Heatmap")
    
    # Create the Alignment layer
    alignment = DiffSWAlgo(match_score=matchScore, miss_score=mismatchScore, 
                           gap=gapScore)
    # Get the cosine similarity output
    output = alignment(x1, x2)
    print(output.shape)
    
    # Visualize the output as a heatmap
    output_np = output.detach().cpu().numpy()
    
    # Clean up input tensors after forward pass
    del x1, x2


    # Function to compute traceback path
    def compute_traceback_path(matrix, similarity_matrix, match_score=1, miss_score=-1, gap_penalty=-1, position=None):
        """Compute the optimal alignment path using traceback"""
        # Find the maximum score position as starting point
        i, j = np.unravel_index(np.argmax(matrix), matrix.shape) if position is None else position
        path = []
        
        while i >= 0 and j >= -1 and matrix[i, j] > 0:
            path.append((i, j))
            aij = similarity_matrix[i, j]
            # Check diagonal, up, and left moves
            diag_score = matrix[i-1, j-1] + aij if i > 0 and j > 0 else 0
            up_score = matrix[i-1, j] + gap_penalty if i > 0 else 0 
            left_score = matrix[i, j-1] + gap_penalty if j > 0 else 0 
            
            # Find the maximum score using simple max
            scores_tensor = torch.tensor([diag_score, up_score, left_score])
            exp_scores = torch.exp(scores_tensor)
            softmax_scores = exp_scores / exp_scores.sum()
            max_score_idx = softmax_scores.argmax().item()
            del scores_tensor, exp_scores, softmax_scores  # Clean up intermediate tensors

            # Priority: diagonal -> up -> left when scores are equal (standard Smith-Waterman)
            if max_score_idx == 0:
                i -= 1
                j -= 1
            elif max_score_idx == 1:
                i -= 1
            elif max_score_idx == 2:
                j -= 1

        return path[::-1]  # Reverse to get path from start to end
    
    # Prepare matrices and indices whether output is 2D ([N,M]) or 3D ([B,N,M])
    if output_np.ndim == 2:
        matrices = [output_np]
        batch_indices = [0]
    else:
        matrices = [output_np[i] for i in range(output_np.shape[0])]
        batch_indices = list(range(output_np.shape[0]))

    for matrix, batch_idx in zip(matrices, batch_indices):
        plt.figure(figsize=(12, 8))
        plt.imshow(matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Alignment Score')
        plt.title(f'Alignment Output Heatmap with Traceback Path - Batch {batch_idx}')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        annotate_all(matrix)
        
        # Compute and plot the traceback path
        # We need to get the similarity matrix from the alignment output
        # For simplicity, we'll use the cosine similarity as a proxy
        if hasattr(alignment, 'cosine_similarity') and alignment.cosine_similarity is not None:
            cs = alignment.cosine_similarity.detach().cpu().numpy()
            if cs.ndim == 3:
                similarity_matrix = cs[batch_idx]
            else:
                similarity_matrix = cs
        else:
            # Fallback: use the output matrix itself as similarity matrix
            similarity_matrix = matrix
        
        # Visualize the similarity matrix
        plt.figure(figsize=(12, 8))
        plt.imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Cosine Similarity Score')
        plt.title(f'Cosine Similarity Matrix - Batch {batch_idx}')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        annotate_all(similarity_matrix, fmt=".3f")
        
        # Add values to each cell for better readability (skip for large matrices)
        if similarity_matrix.shape[0] <= 20 and similarity_matrix.shape[1] <= 20:
            for (i, j), val in np.ndenumerate(similarity_matrix):
                plt.text(j, i, f"{val:.3f}", ha='center', va='center', 
                        color='white' if abs(val) < 0.1 else 'black', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'similarity_matrix_batch_{batch_idx}.png', dpi=150)
        plt.close()
        
        # Compute traceback path starting from the global maximum by default
        path = compute_traceback_path(matrix, similarity_matrix,
                          match_score=7, miss_score=-3, gap_penalty=-1)
        if path:
            path_y = [p[0] for p in path]  # Row indices
            path_x = [p[1] for p in path]  # Column indices
            plt.plot(path_x, path_y, color='red', linewidth=3, marker='o', 
                    markersize=6, alpha=0.8, label='Optimal Alignment Path')
            plt.legend()
            # Clean up path variables
            del path_y, path_x
        
        # Add values to each cell for better readability (skip for large matrices)
        if matrix.shape[0] <= 20 and matrix.shape[1] <= 20:
            for (i, j), val in np.ndenumerate(matrix):
                plt.text(j, i, f"{val:.4f}", ha='center', va='center', 
                        color='white' if val < matrix.mean() else 'black', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'alignment_output_heatmap_with_path_batch_{batch_idx}.png', dpi=150)
        plt.close()
        
        # Clean up batch-specific variables
        del similarity_matrix, path

    # --- Regular Smith-Waterman (non-differentiable) over raw characters ---
    def smith_waterman_scores(s1: str, s2: str, match=1, mismatch=-1, gap=-1):
        n, m = len(s1), len(s2)
        H = np.zeros((n + 1, m + 1), dtype=np.float32)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_diag = H[i - 1, j - 1] + (match if s1[i - 1] == s2[j - 1] else mismatch)
                score_up = H[i - 1, j] + gap
                score_left = H[i, j - 1] + gap
                H[i, j] = max(0.0, score_diag, score_up, score_left)
        return H

    H = smith_waterman_scores(sentence1, sentence2, match=matchScore, mismatch=mismatchScore, gap=gapScore)

    # Heatmap for DiffSWAlgo result (first matrix if batched)
    diff_matrix = output_np if output_np.ndim == 2 else output_np[0]
    plt.figure(figsize=(12, 8))
    plt.imshow(diff_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='DiffSW Alignment Score')
    plt.title('DiffSWAlgo Alignment Heatmap')
    plt.xlabel('Sequence 2 Position')
    plt.ylabel('Sequence 1 Position')
    annotate_all(diff_matrix)
    plt.tight_layout()
    plt.savefig('diffsw_alignment_heatmap.png', dpi=150)
    plt.close()

    # Heatmap for Regular Smith-Waterman (trim zero row/col to N x M)
    reg_matrix = H[1:, 1:]
    plt.figure(figsize=(12, 8))
    plt.imshow(reg_matrix, cmap='magma', aspect='auto')
    plt.colorbar(label='Regular SW Score')
    plt.title('Regular Smith-Waterman Alignment Heatmap')
    plt.xlabel('Sequence 2 Position')
    plt.ylabel('Sequence 1 Position')
    annotate_all(reg_matrix)
    plt.tight_layout()
    plt.savefig('regular_sw_alignment_heatmap.png', dpi=150)
    plt.close()
    
    # --- Overlay paths using pathExtractor on both results ---
    try:
        # DiffSW path overlay
        diff_matrix_t = torch.as_tensor(diff_matrix, dtype=torch.float32)
        diff_batch = diff_matrix_t.unsqueeze(0) if diff_matrix_t.ndim == 2 else diff_matrix_t
        # Use cosine similarity from the model as similarity matrix
        cs_t = getattr(alignment, 'cosine_similarity', None)
        if cs_t is None:
            cs_batch = diff_batch  # fallback if similarity not available
        else:
            if cs_t.ndim == 2:
                cs_batch = cs_t.unsqueeze(0)
            else:
                cs_batch = cs_t
        path_mat_diff, _ = diff_SW_Path(diff_batch, cs_batch, match_score=matchScore,
                                        miss_score=mismatchScore, gap_penalty=gapScore)
        path_np = path_mat_diff[0].detach().cpu().numpy()
        ry, rx = np.where(path_np > 0)
        plt.figure(figsize=(12, 8))
        plt.imshow(diff_matrix, cmap='viridis', aspect='auto')
        if ry.size > 0:
            plt.scatter(rx, ry, c='red', s=12, label='Path')
            plt.legend()
        annotate_all(diff_matrix)
        plt.colorbar(label='DiffSW Alignment Score')
        plt.title('DiffSWAlgo Alignment Heatmap (pathExtractor path)')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        plt.tight_layout()
        plt.savefig('diffsw_alignment_heatmap_with_path.png', dpi=150)
        plt.close()

        # Regular SW path overlay
        reg_matrix_t = torch.as_tensor(reg_matrix, dtype=torch.float32)
        # Build binary similarity matrix for character matches
        N, M = reg_matrix_t.shape
        sim_bin = torch.zeros((N, M), dtype=torch.float32)
        for i, ch1 in enumerate(sentence1):
            for j, ch2 in enumerate(sentence2):
                sim_bin[i, j] = 1.0 if ch1 == ch2 else 0.0
        reg_batch = reg_matrix_t.unsqueeze(0)
        sim_batch = sim_bin.unsqueeze(0)
        path_mat_sw, _ = SW_Path(reg_batch, sim_batch, match_score=matchScore,
                                 miss_score=mismatchScore, gap_penalty=gapScore)
        path_sw_np = path_mat_sw[0].detach().cpu().numpy()
        sy, sx = np.where(path_sw_np > 0)
        plt.figure(figsize=(12, 8))
        plt.imshow(reg_matrix, cmap='magma', aspect='auto')
        if sy.size > 0:
            plt.scatter(sx, sy, c='cyan', s=12, label='Path')
            plt.legend()
        annotate_all(reg_matrix)
        plt.colorbar(label='Regular SW Score')
        plt.title('Regular Smith-Waterman Heatmap (pathExtractor path)')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        plt.tight_layout()
        plt.savefig('regular_sw_alignment_heatmap_with_path.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"Path overlay failed: {e}")

    # Final cleanup
    del output_np, alignment 
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

from NormalizeFuncs import average_normalize
from Parameters import *
from pathExtractor import *
try:
    import psutil  # optional, used for memory logging
except Exception:
    psutil = None

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/10.1.243-fasrc01/"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Centralized image output directory
# Use a path relative to this file so runs from other CWDs still save in the repo.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "NWAlgoIMG")
os.makedirs(OUTPUT_DIR, exist_ok=True)
### Control saving of standalone alignment heatmaps
SAVE_ALIGNMENT_HEATMAPS = False

# A generic mechanism for turning a JAX function into a PyTorch function.

def nw(batch=True, unroll=2, gap=-10.0, temp=1.0):
  # rotate matrix for vectorized dynamic-programming
  def rotate(x, lengths, gap, temp):
    def _ini_global(L):
      return gap*jnp.arange(L)

    a,b = x.shape
    real_a, real_b = lengths
    mask = (jnp.arange(a)<real_a)[:,None] * (jnp.arange(b)<real_b)[None,:]
    
    real_L = lengths
    mask = jnp.pad(mask,[[1,0],[1,0]])
    x = jnp.pad(x,[[1,0],[1,0]])
    
    # solution from jake vanderplas (thanks!)
    a,b = x.shape
    ar,br = jnp.arange(a)[::-1,None], jnp.arange(b)[None,:]
    i,j = (br-ar)+(a-1),(ar+br)//2
    n,m = (a+b-1),(a+b)//2
    zero = jnp.zeros((n,m))
    output = {"x":zero.at[i,j].set(x),
              "mask":zero.at[i,j].set(mask),
              "o":(jnp.arange(n)+a%2)%2}
    
    ini_a, ini_b = _ini_global(a), _ini_global(b)      
    ini = jnp.zeros((a,b)).at[:,0].set(ini_a).at[0,:].set(ini_b)
    output["ini"] = zero.at[i,j].set(ini)

    return {
        "x":output,
        "prev":(jnp.zeros(m),jnp.zeros(m)),
        "idx":(i,j),
        "mask":mask,
        "L":real_L
    }

  # fill the scoring matrix
  def sco(x, lengths, gap=-10.0, temp=1.0):

    def _logsumexp(x, axis=None, mask=None):
      if mask is None: return jax.nn.logsumexp(x,axis=axis)
      else: return x.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(x - x.max(axis,keepdims=True)),axis=axis))

    def _soft_maximum(x, axis=None, mask=None):
      return temp*_logsumexp(x/temp, axis, mask)

    def _cond(cond, true, false):
      return cond*true + (1-cond)*false

    def _step(prev, sm): 
      h2,h1 = prev   # previous two rows of scoring (hij) mtx
      
      Align = h2 + sm["x"]
      Turn = _cond(sm["o"],jnp.pad(h1[:-1],[1,0]),jnp.pad(h1[1:],[0,1]))
      h0 = [Align, h1+gap, Turn+gap]
      h0 = jnp.stack(h0) / temp
      h0 = temp * sm["mask"] * _soft_maximum(h0,0)
      h0 += sm["ini"]
      return (h1,h0),h0

    a,b = x.shape
    sm = rotate(x, lengths=lengths, gap=gap, temp=temp)
    hij = jax.lax.scan(_step, sm["prev"], sm["x"], unroll=unroll)[-1][sm["idx"]]

    return hij[sm["L"][0],sm["L"][1]]
  
  # traceback to get alignment (aka. get marginals)
  traceback = jax.grad(sco)

  # add batch dimension
  # Only vmap over x (axis 0), lengths/gap/temp are not batched (None)
  if batch: return jax.vmap(traceback, in_axes=(0, None, None, None))
  else: return traceback


def j2t(x_jax):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))

def t2j(x_torch):
    x_torch = x_torch.contiguous()  # Ensure tensor is contiguous for conversion
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch)) # type: ignore

def jax2torch(fun):
  class JaxFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
      # Convert all torch tensors to JAX arrays
      jax_args = tuple(t2j(a) if isinstance(a, torch.Tensor) else a for a in args)
      y_, ctx.fun_vjp = jax.vjp(fun, *jax_args)
      ctx.num_args = len(args)
      ctx.is_tensor = [isinstance(a, torch.Tensor) for a in args]
      return j2t(y_)

    @staticmethod
    def backward(ctx, grad_y): # type: ignore
      grad_x_ = ctx.fun_vjp(t2j(grad_y))
      # Return gradients for each input, None for non-tensor args
      return tuple(j2t(g) if ctx.is_tensor[i] else None for i, g in enumerate(grad_x_))

  return JaxFun.apply


###########################################################################
# Needleman-Wunsch Score Matrix Calculation

def compute_nw_score_matrix(similarity_matrix: torch.Tensor, 
                            gap_penalty: float = -1.0,
                            return_numpy: bool = False):
    """
    Compute the Needleman-Wunsch score matrix for global alignment.
    
    Args:
        similarity_matrix: A 2D tensor of shape [N, M] containing pairwise 
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
    if similarity_matrix.dim() != 2:
        raise ValueError(f"Expected 2D similarity matrix, got {similarity_matrix.dim()}D")
    
    N, M = similarity_matrix.shape
    
    # Convert to numpy for easier indexing during DP
    sim = similarity_matrix.detach().cpu().numpy()
    
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
            # Diagonal: match/mismatch
            score_diag = H[i - 1, j - 1] + sim[i - 1, j - 1]
            # Left: gap in sequence 1
            score_left = H[i, j - 1] + gap_penalty
            # Up: gap in sequence 2
            score_up = H[i - 1, j] + gap_penalty
            # Take maximum
            H[i, j] = max(score_diag, score_left, score_up)
    
    if return_numpy:
        return H
    return torch.from_numpy(H)


def compute_nw_score_matrix_from_sequences(seq1: str, 
                                            seq2: str,
                                            match_score: float = 1.0,
                                            mismatch_score: float = -1.0,
                                            gap_penalty: float = -1.0,
                                            return_numpy: bool = False):
    """
    Compute the Needleman-Wunsch score matrix directly from two sequences.
    
    Args:
        seq1: First sequence (string).
        seq2: Second sequence (string).
        match_score: Score for matching characters. Default: 1.0
        mismatch_score: Score for mismatching characters. Default: -1.0
        gap_penalty: Penalty for gaps (typically negative). Default: -1.0
        return_numpy: If True, return numpy array instead of torch tensor.
        
    Returns:
        score_matrix: The NW score matrix of shape [N+1, M+1].
    """
    N, M = len(seq1), len(seq2)
    
    # Initialize score matrix
    H = np.zeros((N + 1, M + 1), dtype=np.float32)
    
    # Initialize first row and column with cumulative gap penalties
    for i in range(1, N + 1):
        H[i, 0] = i * gap_penalty
    for j in range(1, M + 1):
        H[0, j] = j * gap_penalty
    
    # Fill the score matrix using dynamic programming
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Determine match/mismatch score
            if seq1[i - 1] == seq2[j - 1]:
                sub_score = match_score
            else:
                sub_score = mismatch_score
            
            # Diagonal: match/mismatch
            score_diag = H[i - 1, j - 1] + sub_score
            # Left: gap in sequence 1
            score_left = H[i, j - 1] + gap_penalty
            # Up: gap in sequence 2
            score_up = H[i - 1, j] + gap_penalty
            # Take maximum
            H[i, j] = max(score_diag, score_left, score_up)
    
    if return_numpy:
        return H
    return torch.from_numpy(H)


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

    def forward(self, x1, x2, threshold=0.5):
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
        comp_cosine_similarity = 1 - cosine_similarity
        del dot_product, denominator

        # threshold_cosine_similarity = comp_cosine_similarity
        threshold_cosine_similarity = torch.sigmoid(10e6 * (cosine_similarity - threshold))
        threshold_comp_cosine_similarity = torch.sigmoid(10e6 * (comp_cosine_similarity - threshold))
        # Multiply by matchscore and missscore
        match_score_contribution = threshold_cosine_similarity * self.matchscore
        miss_score_contribution = threshold_comp_cosine_similarity * self.missscore # missscore is typically negative
        del cosine_similarity, comp_cosine_similarity

        # Sum match and miss scores to get the final score
        final_score = match_score_contribution + miss_score_contribution
        del match_score_contribution, miss_score_contribution

        return final_score
    


###########################################################################
# Differentiable Smith-Waterman Algorithm

class DiffNWAlgo(nn.Module):
    def __init__(self, match_score, miss_score, gap, temp=1.0, batch=True):
        super(DiffNWAlgo, self).__init__()
        self.match_score = match_score
        self.miss_score = miss_score
        self.temp = temp
        self.cosine_similarity_layer = CosineSimilarityLayer(matchscore= match_score,
                                                             missscore=miss_score)
        # Use non-batched NW because we pass a 2D [N,M] similarity matrix
        self.nw_fn_torch = jax2torch(jax.jit(nw(batch=batch, gap=gap, temp=temp)))

    
    def reset_cosine_similarity(self):
        self.ImageSimilar = None
        
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
            self.ImageSimilar = self.cosine_similarity_layer(x1, x2)
            if show_dims:
                print(f"Cosine similarity shape: {self.ImageSimilar.shape}")
            new_output = torch.squeeze(self.ImageSimilar, dim=0)
            if show_dims:
                print(f"Squeezed cosine similarity shape: {new_output.shape}")
            # visualize_heatmap_with_values(new_output[0], title="Cosine Similarity Heatmap")
            # Pass lengths as a tuple (N, M) and gap as positional args
            lengths = (new_output.shape[-2], new_output.shape[-1])
            self.align = self.nw_fn_torch(new_output, lengths, gapScore, self.temp)
            if show_dims:
                print(f"align shape: {self.align.shape}")
        else:
            self.ImageSimilar = similarity_matrix
            if show_dims:
                print(f"Cosine similarity shape: {self.ImageSimilar.shape}")
            lengths = (self.ImageSimilar.shape[-2], self.ImageSimilar.shape[-1])
            self.align = self.nw_fn_torch(self.ImageSimilar, lengths, gapScore, 100.0)
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
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_')}.png"))
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
    alignment = DiffNWAlgo(match_score=matchScore, miss_score=mismatchScore, 
                           gap=gapScore, batch=False)
    # Get the cosine similarity output
    output = alignment(x1, x2, calc_cosine=True)
    print(output.shape)
    
    # Visualize the output as a heatmap
    
    output = average_normalize(output.unsqueeze(0))
    output_np = output.detach().cpu().numpy()

    
    # Clean up input tensors after forward pass
    del x1, x2
    
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
        cs = alignment.ImageSimilar.detach().cpu().numpy()
        if cs.ndim == 3:
            similarity_matrix = cs[batch_idx]
        else:
            similarity_matrix = cs
        
        # Visualize the similarity matrix
        plt.figure(figsize=(12, 8))
        plt.imshow(similarity_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Cosine Similarity Score')
        plt.title(f'Cosine Similarity Matrix')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        # Use characters as axis tick labels
        try:
            ax = plt.gca()
            ax.set_xticks(np.arange(len(sentence2)))
            ax.set_xticklabels(list(sentence2), rotation=90, fontsize=8)
            ax.set_yticks(np.arange(len(sentence1)))
            ax.set_yticklabels(list(sentence1), fontsize=8)
        except Exception:
            pass
        annotate_all(similarity_matrix, fmt=".3f")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'similarity_matrix.png'), dpi=150)
        plt.close()

    # Heatmap for DiffNWAlgo result (first matrix if batched)
    diff_matrix = output_np if output_np.ndim == 2 else output_np[0]
    plt.figure(figsize=(12, 8))
    plt.imshow(diff_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='DiffNW Alignment Score')
    plt.title('DiffNWAlgo Alignment Heatmap')
    plt.xlabel('Sequence 2 Position')
    plt.ylabel('Sequence 1 Position')
    annotate_all(diff_matrix)
    plt.tight_layout()
    if SAVE_ALIGNMENT_HEATMAPS:
        plt.savefig(os.path.join(OUTPUT_DIR, 'diffnw_alignment_heatmap.png'), dpi=150)
    plt.close()

    # Regular Needleman-Wunsch heatmap omitted; using compute_traceback_path directly for path.
    
    # --- Overlay paths using pathExtractor on both results ---
    try:
        # DiffNW path overlay
        diff_matrix_t = torch.as_tensor(diff_matrix, dtype=torch.float32)
        diff_batch = diff_matrix_t.unsqueeze(0) if diff_matrix_t.ndim == 2 else diff_matrix_t
        # Use cosine similarity from the model as similarity matrix
       
        cs_batch = alignment.ImageSimilar  # fallback if similarity not available
        # Build a path matrix using compute_diff_traceback_path per batch item
        if diff_batch.ndim == 3:
            path_mat_diff = torch.zeros_like(diff_batch)
            for bi in range(diff_batch.shape[0]):
                path_list, _start = compute_diff_traceback_path(diff_batch[bi], cs_batch[bi],
                                                               match_score=matchScore,
                                                               miss_score=mismatchScore,
                                                               gap_penalty=gapScore)
                for (x, y) in path_list:
                    path_mat_diff[bi, x, y] = 1
        else:
            # Single matrix case
            path_list, _start = compute_diff_traceback_path(diff_batch, cs_batch,
                                                           match_score=matchScore,
                                                           miss_score=mismatchScore,
                                                           gap_penalty=gapScore)
            path_mat_diff = torch.zeros_like(diff_batch)
            for (x, y) in path_list:
                path_mat_diff[x, y] = 1

        path_np = (path_mat_diff if path_mat_diff.ndim == 2 else path_mat_diff[0]).detach().cpu().numpy()
        ry, rx = np.where(path_np > 0)
        plt.figure(figsize=(12, 8))
        plt.imshow(diff_matrix, cmap='viridis', aspect='auto')
        if ry.size > 0:
            # Dots
            plt.scatter(rx, ry, c='red', s=12, label='Path')
            # Connect dots in traversal order (sort by i+j which increases by 1 per step)
            order = np.argsort(ry + rx)
            plt.plot(rx[order], ry[order], c='red', linewidth=1.5, alpha=0.9)
            plt.legend()
        annotate_all(diff_matrix)
        plt.colorbar(label='DiffNW Alignment Score')
        plt.title('DiffNWAlgo Alignment Heatmap (pathExtractor path)')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        # Use characters as axis tick labels
        try:
            ax = plt.gca()
            ax.set_xticks(np.arange(len(sentence2)))
            ax.set_xticklabels(list(sentence2), rotation=90, fontsize=8)
            ax.set_yticks(np.arange(len(sentence1)))
            ax.set_yticklabels(list(sentence1), fontsize=8)
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'diffnw_alignment_heatmap_with_path.png'), dpi=150)
        plt.close()

        # Regular NW path overlay
        # Build a binary similarity matrix for character matches
        N, M = len(sentence1), len(sentence2)
        sim_bin = torch.zeros((N, M), dtype=torch.float32)
        for i, ch1 in enumerate(sentence1):
            for j, ch2 in enumerate(sentence2):
                sim_bin[i, j] = 1.0 if ch1 == ch2 else 0.0
        # Build the Needleman-Wunsch score matrix using the helper function
        H = compute_nw_score_matrix_from_sequences(
            sentence1, sentence2,
            match_score=matchScore,
            mismatch_score=mismatchScore,
            gap_penalty=gapScore,
            return_numpy=True
        )
        reg_matrix = H[1:, 1:]
        # Use compute_traceback_path from pathExtractor on the NW score matrix
        path_NW, _ = compute_traceback_path(torch.as_tensor(reg_matrix), torch.as_tensor(sim_bin),
                                            match_score=matchScore, miss_score=mismatchScore, gap_penalty=gapScore)
        sy, sx = (np.array([p[0] for p in path_NW]), np.array([p[1] for p in path_NW])) if len(path_NW) > 0 else (np.array([]), np.array([]))
        plt.figure(figsize=(12, 8))
        plt.imshow(reg_matrix, cmap='magma', aspect='auto')
        if sy.size > 0:
            # Dots
            plt.scatter(sx, sy, c='cyan', s=12, label='Path')
            # Connect dots in traversal order (sort by i+j)
            order2 = np.argsort(sy + sx)
            plt.plot(sx[order2], sy[order2], c='cyan', linewidth=1.5, alpha=0.9)
            plt.legend()
        annotate_all(reg_matrix)
        plt.colorbar(label='Regular NW Score')
        plt.title('Regular Needleman-Wunsch Heatmap (pathExtractor path)')
        plt.xlabel('Sequence 2 Position')
        plt.ylabel('Sequence 1 Position')
        # Use characters as axis tick labels
        try:
            ax = plt.gca()
            ax.set_xticks(np.arange(len(sentence2)))
            ax.set_xticklabels(list(sentence2), rotation=90, fontsize=8)
            ax.set_yticks(np.arange(len(sentence1)))
            ax.set_yticklabels(list(sentence1), fontsize=8)
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'regular_nw_alignment_heatmap_with_path.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Path overlay failed: {e}")

    # Final cleanup
    del output_np, alignment 
# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import sys
import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math
import jax

# ----------------------------------------------------------------------------------------------------------------------

@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma

    for p in range(n_passes):
        J = max(0, min(p - tid, max_j - 1))
        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                # 1. Restricted D3TW topology (intentional design choice):
                #    - diagonal transition: advance both text char (i) and image step (j)
                #    - stay transition: keep the same text char (i), advance image step (j)
                # This supports one-character-to-many-windows alignment, which is suitable
                # for Arabic connected-script where one letter spans multiple overlapping
                # windows. It is NOT full 3-neighbor DTW (which would also allow skipping
                # a text character with an upward transition).
                # We negate them because we want Soft-Min, not Soft-Max
                val_match = R[b, i - 1, j - 1]
                val_stay  = R[b, i, j - 1]
                
                r0 = -val_match * inv_gamma
                r2 = -val_stay * inv_gamma
                
                # 2. Find maximum for stability
                rmax = max(r0, r2)
                
                # =========================================================
                # THE FIX: THE INFINITY GATE
                # If rmax is -inf, it means BOTH neighbors are +inf (unreachable).
                # We CANNOT do math on them. Just set current cell to inf.
                # -1e20 is a safe threshold for "negative infinity"
                # =========================================================
                if rmax < -1e20:
                    R[b, i, j] = math.inf
                else:
                    # Safe to proceed: at least one path is valid
                    rsum = math.exp(r0 - rmax) + math.exp(r2 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin

        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))
        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):
            # 1. Initialize gradient to 0
            E[k, i, j] = 0.0
            
            # 2. Safety Check: If this cell was unreachable in forward pass, 
            # it has no gradient. Skip it.
            if math.isinf(R[k, i, j]):
                # Do nothing, E[k,i,j] remains 0.0
                pass
            
            elif not (abs(i - j) > bandwidth > 0):
                # 3. Check Child 1: Right (Stay) -> corresponds to R[i, j+1]
                # We only receive gradient if the child was reachable (not inf)
                val_stay = R[k, i, j + 1]
                if not math.isinf(val_stay):
                    # Formula: exp( (R_child - R_current - D_child) / gamma )
                    # This calculates the probability of the path going this way.
                    b = math.exp((val_stay - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                    E[k, i, j] += E[k, i, j + 1] * b

                # 4. Check Child 2: Diagonal (Match) -> corresponds to R[i+1, j+1]
                val_match = R[k, i + 1, j + 1]
                if not math.isinf(val_match):
                    c = math.exp((val_match - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                    E[k, i, j] += E[k, i + 1, j + 1] * c

        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0
        
        sys.stdout.flush()

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, 
                                                   n_passes, cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        # jax.debug.print("R = {}", R)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------

@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < np.abs(i - j):
                    continue
                
                # Get negated values
                r0 = -R[b, i - 1, j - 1] / gamma
                r2 = -R[b, i, j - 1] / gamma
                
                rmax = max(r0, r2)
                
                # =========================================================
                # THE FIX: THE INFINITY GATE
                # =========================================================
                if rmax < -1e20:
                    R[b, i, j] = np.inf
                else:
                    rsum = np.exp(r0 - rmax) + np.exp(r2 - rmax)
                    softmin = - gamma * (np.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    
    # Boundary cleanup
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                
                # --- FIX 1: If forward was Inf, Gradient is 0 ---
                if np.isinf(R[k, i, j]):
                    E[k, i, j] = 0.0
                    continue

                if 0 < bandwidth < np.abs(i - j):
                    continue
                
                # --- FIX 2: Check neighbors before subtraction ---
                
                # Child 1: Stay (i, j+1)
                val_stay = R[k, i, j + 1]
                if not np.isinf(val_stay):
                    b0 = (val_stay - R[k, i, j] - D[k, i, j + 1]) / gamma
                    E[k, i, j] += E[k, i, j + 1] * np.exp(b0)

                # Child 2: Match (i+1, j+1)
                val_match = R[k, i + 1, j + 1]
                if not np.isinf(val_match):
                    c0 = (val_match - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                    E[k, i, j] += E[k, i + 1, j + 1] * np.exp(c0)
                    
    return E[:, 1:N + 1, 1:M + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is None:
            self.dist_func = SoftDTW._euclidean_dist_func
        else:
            self.dist_func = SoftDTW._cosine_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)
    
    @staticmethod
    def _cosine_dist_func(x, y):
        """
        Calculates the cosine distance (1 - cosine_similarity) between each element in x and y per timestep.
        Cosine distance ranges from 0 (identical) to 2 (opposite).
        """
        # Normalize along the feature dimension
        x_norm = x / (x.norm(dim=2, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(dim=2, keepdim=True) + 1e-8)
        
        # Compute cosine similarity matrix: [batch, n, m]
        # x_norm: [batch, n, d], y_norm: [batch, m, d]
        cosine_sim = torch.bmm(x_norm, y_norm.transpose(1, 2))
        
        # Convert to distance (1 - similarity), so 0 = identical, 2 = opposite
        return 1.0 - cosine_sim
    

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            sys.stdout.flush() 
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    forward = sdtw(a, b)
    print(forward)
    end = timer()
    t = end - start

    grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, forward, grads

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    # sdtw = SoftDTW(False, gamma=1.0, normalize=False)
    sdtw_cuda = SoftDTW(True, gamma=1.0, normalize=False)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, sdtw_cuda)

        # CPU
        # t_cpu, forward_cpu, backward_cpu = timed_run(a_cpu, b_cpu, sdtw)

        # Verify the results
        # assert torch.allclose(forward_cpu, forward_gpu.cpu())
        # assert torch.allclose(backward_cpu, backward_gpu.cpu(), atol=tol_backward)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            # times_cpu += [t_cpu]
            times_gpu += [t_gpu]

    # Average and log
    # avg_cpu = np.mean(times_cpu)
    avg_gpu = np.mean(times_gpu)
    # print("  CPU:     ", avg_cpu)
    print("  GPU:     ", avg_gpu)
    # print("  Speedup: ", avg_cpu / avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    profile(1, 15, 17, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    # profile(512, 256, 256, 2, tol_backward=1e-3)
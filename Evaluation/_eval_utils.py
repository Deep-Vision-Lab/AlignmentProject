"""
Shared utilities for all evaluation scripts.

Provides:
  - load_image_model()   : load EmbeddingModel from a .pth weights file
  - load_text_model()    : instantiate TextEmbedding
  - get_image_embedding(): run EmbeddingModel on one image -> [1, S, D]
  - get_text_embedding() : run TextEmbedding on a string   -> [1, T, D]
  - compute_sim_matrix() : cosine similarity [T, S]
  - soft_dtw_path()      : D3TW alignment path via SoftDTW kernel
  - soft_dtw_cost()      : SoftDTW alignment cost normalised by (T+S)
  - dtw_path_classic()   : standard DTW path (no gaps, no SoftDTW)
  - hard_dtw_cost()      : NW DP cost (kept for reference)
  - dtw_path()           : NW DP path (kept for reference)
  - load_test_pairs()    : yield (img_path, text) pairs from a data directory
"""

import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---- make project root importable ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Parameters import *
from embeddingModel import EmbeddingModel, sliding_window
from textEmbedding import TextEmbedding, build_text_embedder
from soft_dtw_cuda import compute_softdtw


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_image_model(weights_path: str, dev: str = device) -> EmbeddingModel:
    """Load EmbeddingModel from a checkpoint file."""
    model = EmbeddingModel(
        window_size=window_size,
        stride=window_size,  # Now uses calculated overlap stride
        vector_size=vector_size,
        device=device,
        # OPTIMIZATION 1 & 3: Enable BiLSTM and Positional Encoding
        use_bilstm=use_bilstm,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_type=positional_encoding_type,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout
    ).to(dev)
    checkpoint = torch.load(weights_path, map_location=dev)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_text_model(dev: str = device):
    """Instantiate the text embedder selected in Parameters.py.

    Returns either a TextEmbedding (char) or a FastTextCharEmbedding,
    both of which expose the same forward/char_to_index interface.
    """
    model = build_text_embedder(embedding_dim=vector_size).to(dev)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def get_image_embedding(model: EmbeddingModel, image_path: str,
                        dev: str = device) -> torch.Tensor:
    """
    Return normalised image embeddings of shape [1, S, D].

    S = number of sliding-window patches, D = vector_size.
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = _IMG_TRANSFORM(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        emb = model(img_tensor)           # [1, S, D]
    emb = F.normalize(emb, p=2, dim=-1)
    return emb


def get_text_embedding(model: TextEmbedding, text: str,
                       dev: str = device) -> torch.Tensor:
    """
    Return normalised text character embeddings of shape [1, T, D].

    T = len(text), D = vector_size.
    """
    with torch.no_grad():
        emb = model(text)                 # [T, D]
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.unsqueeze(0)              # [1, T, D]


def compute_sim_matrix(img_emb: torch.Tensor,
                       txt_emb: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity matrix of shape [T, S].

    img_emb: [1, S, D]
    txt_emb: [1, T, D]
    """
    # [T, S]
    sim = torch.bmm(txt_emb, img_emb.transpose(1, 2)).squeeze(0)
    return sim


# ---------------------------------------------------------------------------
# Hard DTW (pure numpy DP – no JAX / CUDA dependency)
# ---------------------------------------------------------------------------

def hard_dtw_cost(sim: torch.Tensor,
                  gap_penalty: float = -10.0,
                  match_score: float = 10.0,
                  mismatch_score: float = -27.0,
                  match_threshold: float = 0.0) -> float:
    """
    Run hard Needleman-Wunsch DP on a similarity matrix and return the
    normalised alignment cost.

    sim              : [T, S] cosine-similarity matrix
    gap_penalty      : penalty for inserting a gap (default -10)
    match_score      : score for a matching cell (sim > match_threshold)
    mismatch_score   : score for a mismatching cell
    match_threshold  : cosine-similarity threshold that divides match / mismatch
    Returns: scalar – total score normalised by (T+S)  (higher = better)
    """
    if isinstance(sim, torch.Tensor):
        s = sim.detach().cpu().numpy().astype(np.float32)
    else:
        s = np.array(sim, dtype=np.float32)

    T, S = s.shape
    H = np.full((T + 1, S + 1), -1e9, dtype=np.float32)
    H[0, 0] = 0.0
    for i in range(1, T + 1):
        H[i, 0] = H[i - 1, 0] + gap_penalty
    for j in range(1, S + 1):
        H[0, j] = H[0, j - 1] + gap_penalty

    for i in range(1, T + 1):
        for j in range(1, S + 1):
            cell_score = match_score if s[i - 1, j - 1] > match_threshold else mismatch_score
            H[i, j] = max(
                H[i - 1, j - 1] + cell_score,   # match / mismatch
                H[i - 1, j]     + gap_penalty,  # gap in S
                H[i, j - 1]     + gap_penalty,  # gap in T
            )

    return float(H[T, S]) / (T + S)


def dtw_path(sim: torch.Tensor,
             gap_penalty: float = -10.0,
             match_score: float = 10.0,
             mismatch_score: float = -27.0,
             match_threshold: float = 0.0) -> list:
    """
    Return the optimal NW alignment path as a list of (t_idx, s_idx) pairs.

    sim              : [T, S] cosine-similarity matrix
    gap_penalty      : penalty for inserting a gap (default -10)
    match_score      : NW score for a matched cell
    mismatch_score   : NW score for a mismatched cell
    match_threshold  : cosine-similarity threshold that divides match / mismatch
    """
    if isinstance(sim, torch.Tensor):
        s = sim.detach().cpu().numpy().astype(np.float32)
    else:
        s = np.array(sim, dtype=np.float32)

    T, S = s.shape

    H = np.full((T + 1, S + 1), -1e9, dtype=np.float32)
    H[0, 0] = 0.0
    for i in range(1, T + 1):
        H[i, 0] = H[i - 1, 0] + gap_penalty
    for j in range(1, S + 1):
        H[0, j] = H[0, j - 1] + gap_penalty

    for i in range(1, T + 1):
        for j in range(1, S + 1):
            cell_score = match_score if s[i - 1, j - 1] > match_threshold else mismatch_score
            H[i, j] = max(
                H[i - 1, j - 1] + cell_score,
                H[i - 1, j]     + gap_penalty,
                H[i, j - 1]     + gap_penalty,
            )

    # Traceback
    i, j = T, S
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        cell_score = match_score if s[i - 1, j - 1] > match_threshold else mismatch_score
        diag = H[i - 1, j - 1] + cell_score
        up   = H[i - 1, j]     + gap_penalty
        left = H[i, j - 1]     + gap_penalty
        best = max(diag, up, left)
        if best == diag:
            i -= 1; j -= 1
        elif best == up:
            i -= 1
        else:
            j -= 1

    while i > 0:
        path.append((i - 1, 0)); i -= 1
    while j > 0:
        path.append((0, j - 1)); j -= 1

    return list(reversed(path)), H[1:, 1:]


def dtw_path_classic(sim: torch.Tensor) -> list:
    """
    Standard DTW alignment path using raw cosine similarity.

    No gaps, no match/mismatch thresholds — every element of both
    sequences is consumed.  Allows one-to-many (repetition) mappings.

    sim : [T, S] cosine-similarity matrix (higher = better match)
    Returns: list of (t_idx, s_idx) pairs from (0,0) to (T-1, S-1).
    """
    if isinstance(sim, torch.Tensor):
        s = sim.detach().cpu().numpy().astype(np.float32)
    else:
        s = np.array(sim, dtype=np.float32)

    T, S = s.shape

    # Accumulated similarity matrix (maximise)
    D = np.full((T, S), -np.inf, dtype=np.float32)
    D[0, 0] = s[0, 0]
    for i in range(1, T):
        D[i, 0] = D[i - 1, 0] + s[i, 0]
    for j in range(1, S):
        D[0, j] = D[0, j - 1] + s[0, j]
    for i in range(1, T):
        for j in range(1, S):
            D[i, j] = s[i, j] + max(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])

    # Traceback from (T-1, S-1) to (0, 0)
    i, j = T - 1, S - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = max(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            if best == D[i - 1, j - 1]:
                i -= 1; j -= 1
            elif best == D[i - 1, j]:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    return list(reversed(path))


def soft_dtw_path(sim: torch.Tensor,
                  gamma: float = 1.0) -> tuple:
    """
    Alignment path via Soft-DTW (D3TW topology: match or stay).

    Uses the CPU kernel from soft_dtw_cuda.compute_softdtw on the cosine
    *distance* matrix (D = 1 – sim).  Then traces back through the
    accumulated cost matrix R to produce a hard path.

    D3TW allows only two transitions:
      • match : (i-1, j-1)  – advance both text and image
      • stay  : (i,   j-1)  – repeat current text char (image advances)

    sim   : [T, S] cosine-similarity matrix
    gamma : softness (smaller = closer to hard DTW, default 1.0)

    Returns
    -------
    path : list of (t, s) index pairs from (0, 0) to (T-1, S-1)
    R    : [T, S] float32 array – accumulated SoftDTW cost (for heatmap)
    """
    if isinstance(sim, torch.Tensor):
        s = sim.detach().cpu().numpy().astype(np.float32)
    else:
        s = np.array(sim, dtype=np.float32)

    T, S = s.shape
    D = 1.0 - s                              # cosine distance [T, S]
    D_batch = D[np.newaxis]                  # [1, T, S]

    R_full = compute_softdtw(D_batch, gamma, 0)   # [1, T+2, S+2]
    R = R_full[0, 1:T + 1, 1:S + 1].copy()        # [T, S]  accumulated cost

    # ---- Hard traceback (D3TW: match or stay) ----
    # R_full uses 1-based indexing; trace from (T, S) back to (1, 1)
    i, j = T, S
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if j == 1:
            i -= 1
        elif i == 1:
            j -= 1
        else:
            # match: R_full[i-1, j-1]  vs  stay: R_full[i, j-1]
            if R_full[0, i - 1, j - 1] <= R_full[0, i, j - 1]:
                i -= 1; j -= 1
            else:
                j -= 1
        path.append((i - 1, j - 1))

    return list(reversed(path)), R


def soft_dtw_cost(sim: torch.Tensor, gamma: float = 1.0) -> float:
    """
    SoftDTW alignment cost for a similarity matrix.

    Converts cosine similarity to distance (D = 1 – sim), runs the
    CPU SoftDTW kernel and returns the final accumulated cost normalised
    by (T + S).  A lower value means the pair aligns better.

    sim   : [T, S] cosine-similarity matrix
    gamma : softness (default 1.0)
    """
    if isinstance(sim, torch.Tensor):
        s = sim.detach().cpu().numpy().astype(np.float32)
    else:
        s = np.array(sim, dtype=np.float32)

    T, S = s.shape
    D_batch = (1.0 - s)[np.newaxis]             # [1, T, S] cosine distance
    R_full  = compute_softdtw(D_batch, gamma, 0)  # [1, T+2, S+2]
    return float(R_full[0, -2, -2]) / (T + S)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_test_pairs(data_dir: str, split: str = 'test',
                    seed: int = 42, n_samples: int = None):
    """
    Yield (img1_path, text1, img2_path, text2) tuples from the dataset.

    data_dir : e.g. 'DataSet/Synthetic_Arabic'
    split    : 'train' | 'valid' | 'test'  — uses the same 60/20/20 split
    n_samples: if given, cap the number of pairs returned
    """
    img_dir  = os.path.join(data_dir, 'images')
    txt_dir  = os.path.join(data_dir, 'texts')

    # Collect all indices present on disk
    all_indices = sorted(
        int(os.path.basename(p).replace('img1_', '').replace('.png', ''))
        for p in glob.glob(os.path.join(img_dir, 'img1_*.png'))
    )

    rng = np.random.default_rng(seed)
    idx = np.array(all_indices)
    rng.shuffle(idx)

    n = len(idx)
    train_end = int(0.6 * n)
    valid_end = int(0.8 * n)

    if split == 'train':
        idx = idx[:train_end]
    elif split == 'valid':
        idx = idx[train_end:valid_end]
    else:  # test
        idx = idx[valid_end:]

    if n_samples is not None:
        idx = idx[:n_samples]

    for i in idx:
        img1 = os.path.join(img_dir, f'img1_{i}.png')
        img2 = os.path.join(img_dir, f'img2_{i}.png')
        t1   = os.path.join(txt_dir, f'text1_{i}.txt')
        t2   = os.path.join(txt_dir, f'text2_{i}.txt')

        if not all(os.path.exists(p) for p in [img1, img2, t1, t2]):
            continue

        with open(t1, encoding='utf-8') as f:
            text1 = f.read().strip()
        with open(t2, encoding='utf-8') as f:
            text2 = f.read().strip()

        yield img1, text1, img2, text2

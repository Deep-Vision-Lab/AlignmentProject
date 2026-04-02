import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from saveDATA import *
from Visualization import *
from embeddingModel import EmbeddingModel
from DiffNWAlgo import * # Assuming sliding_window is not here
from newDataLoader import test_dataloader
from pathExtractor import *
from LossFunctionWithHelpers import *

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Retrieval metrics for training-loop evaluation
# ---------------------------------------------------------------------------

from tqdm import tqdm
from soft_dtw_cuda import compute_softdtw
from Parameters import device

def compute_retrieval_metrics(imageEmbedding, textEmbedding,
                              dataloader, n_samples: int = 64,
                              gamma: float = 0.1):
    """
    Compute Recall@1, Recall@5, Recall@10 and MRR on a single batch
    taken from *dataloader*.  Suitable for cheap per-epoch monitoring.

    Protocol
    --------
    - Sample up to n_samples (image, positive_text) pairs from the loader.
    - Build an (N x N) soft-DTW cost matrix: cost[i,j] = dtw(img_i, txt_j).
    - The correct match for image i is text i (diagonal).
    - Rank image i's true text among all N texts; collect rank.

    Returns
    -------
    dict with keys: R1, R5, R10, MRR  (R@k already in %, MRR in [0,1])
    """
    imageEmbedding.eval()
    textEmbedding.eval()

    img_embs, txt_embs = [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Embedding', unit='batch')
        for batch in pbar:
            images, pos_texts, _ = batch
            images = images.to(device)
            pbar.set_postfix(collected=len(img_embs), total=n_samples)

            # Embed images  [B, S_img, D]
            img_out = imageEmbedding(images)
            for b in range(img_out.shape[0]):
                img_embs.append(img_out[b])          # [S_img, D]

            # Embed positive texts  [B, S_txt, D]
            for txt in pos_texts:
                emb = textEmbedding(txt)              # [S_txt, D]
                txt_embs.append(emb.to(device))

            if len(img_embs) >= n_samples:
                break

    img_embs = img_embs[:n_samples]
    txt_embs = txt_embs[:n_samples]
    N = len(img_embs)

    # Build N x N soft-DTW cost matrix
    cost_matrix = torch.full((N, N), float('inf'))

    with torch.no_grad():
        for i in tqdm(range(N), desc='Cost matrix', unit='query',
                      postfix={'N': N}):
            img = img_embs[i]                        # [S_img, D]
            for j in range(N):
                txt = txt_embs[j]                    # [S_txt, D]
                # cosine similarity matrix [S_img, S_txt]
                sim = F.cosine_similarity(
                    img.unsqueeze(1),                # [S_img, 1, D]
                    txt.unsqueeze(0),                # [1, S_txt, D]
                    dim=-1
                )
                # soft-DTW on distance  D = 1 - sim  shape [1, S_img, S_txt]
                D_np = (1.0 - sim).detach().cpu().numpy().astype('float32')[np.newaxis]  # [1, S_img, S_txt]
                # compute_softdtw expects numpy, returns numpy R [1, S_img+2, S_txt+2]
                R = compute_softdtw(D_np, gamma, 0)
                raw_cost = float(R[0, -2, -2])
                norm_cost = raw_cost / (img.shape[0] + txt.shape[0])
                cost_matrix[i, j] = norm_cost

    # Compute rank-based metrics
    ranks = []
    for i in range(N):
        # ascending sort → rank 1 = lowest cost = best match
        order = torch.argsort(cost_matrix[i])
        rank  = int((order == i).nonzero(as_tuple=True)[0].item()) + 1
        ranks.append(rank)

    r1  = sum(r <= 1  for r in ranks) / N * 100
    r5  = sum(r <= 5  for r in ranks) / N * 100
    r10 = sum(r <= 10 for r in ranks) / N * 100
    mrr = sum(1.0 / r for r in ranks) / N

    return dict(R1=r1, R5=r5, R10=r10, MRR=mrr)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from Parameters import window_size, vector_size, use_bilstm, \
        use_positional_encoding, positional_encoding_type, bilstm_layers, \
        model_dropout
    from embeddingModel import EmbeddingModel
    from textEmbedding import TextEmbedding
    from newDataLoader import test_dataloader

    parser = argparse.ArgumentParser(description='Retrieval evaluation (R@1, R@5, R@10, MRR)')
    parser.add_argument('--weights',   type=str, default='model_epoch_80.pth',
                        help='Path to model weights (.pth)')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Number of pairs to build the ranking pool from')
    parser.add_argument('--gamma',     type=float, default=0.1,
                        help='Soft-DTW smoothing parameter')
    args = parser.parse_args()

    # ── load models ───────────────────────────────────────────────────────
    stride = max(1, int(window_size * 0.5))

    img_model = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_bilstm=use_bilstm,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_type=positional_encoding_type,
        bilstm_layers=bilstm_layers,
        dropout=model_dropout,
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        img_model.load_state_dict(checkpoint['state_dict'])
    else:
        img_model.load_state_dict(checkpoint)
    img_model.eval()
    print(f"Loaded weights: {args.weights}")

    txt_model = TextEmbedding(embedding_dim=vector_size).to(device)
    txt_model.eval()

    # ── run evaluation ────────────────────────────────────────────────────
    print(f"Building {args.n_samples}-way ranking pool from test set …")
    metrics = compute_retrieval_metrics(
        img_model, txt_model,
        test_dataloader,
        n_samples=args.n_samples,
        gamma=args.gamma,
    )

    print("\n========================================")
    print("        Retrieval Evaluation Results     ")
    print("========================================")
    print(f"  Pool size  : {args.n_samples}")
    print(f"  Recall@1   : {metrics['R1']:.2f}%")
    print(f"  Recall@5   : {metrics['R5']:.2f}%")
    print(f"  Recall@10  : {metrics['R10']:.2f}%")
    print(f"  MRR        : {metrics['MRR']:.4f}")
    print("========================================\n")
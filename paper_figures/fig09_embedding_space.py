"""
fig09_embedding_space.py
=========================
Visualise the learned image-window embedding space via t-SNE / UMAP / PCA.

Two panels:
  (a) Before training — random model embeddings
  (b) After training  — trained model embeddings

Points are coloured by sample id (or by aligned character if alignment path
is computed — requires the text branch).

Output:
  fig09_embedding_space_before_after.png / .pdf
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from utils.model_loading import load_image_model, load_random_model
from utils.sample_data import make_sample, FIG_STRIDE
from utils.similarity import compute_image_embeddings
from utils.plotting import setup_paper_style, save_figure


# ── Dimensionality reduction helper ──────────────────────────────────────────

def _reduce(embeddings, method, n_components=2, seed=42):
    """
    Reduce [N, D] embeddings to [N, 2].

    method: 'tsne' | 'umap' | 'pca'
    Falls back: umap → tsne → pca
    """
    method = method.lower()

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=seed,
                                n_neighbors=min(15, len(embeddings) - 1))
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("  [fig09] umap-learn not installed, falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(5, len(embeddings) // 5))
            reducer = TSNE(n_components=n_components, random_state=seed,
                           perplexity=perp, max_iter=1000, init="pca")
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("  [fig09] scikit-learn not installed — "
                  "install with: pip install scikit-learn")
            print("  [fig09] falling back to PCA (no dimensionality reduction needed).")
            method = "pca"

    # PCA fallback
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=n_components, random_state=seed)
    return reducer.fit_transform(embeddings)


# ── Embedding collection ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(model, num_samples, device):
    """Return (all_window_vecs [N_total, D], sample_ids [N_total])."""
    all_vecs, all_ids = [], []
    for si in range(num_samples):
        img_tensor, _ = make_sample(si)
        emb = compute_image_embeddings(model, img_tensor, device)
        all_vecs.append(emb.detach().cpu().numpy())
        all_ids.extend([si] * emb.shape[0])
    return np.vstack(all_vecs), np.array(all_ids)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_embedding_space(checkpoint, num_samples, output_dir, device, method):
    setup_paper_style()

    random_model  = load_random_model(device,    stride_override=FIG_STRIDE)
    trained_model = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)

    print(f"  Collecting embeddings (before) …")
    vecs_before, ids_before = collect_embeddings(random_model,  num_samples, device)
    print(f"  Collecting embeddings (after)  …")
    vecs_after,  ids_after  = collect_embeddings(trained_model, num_samples, device)

    print(f"  Running {method.upper()} on {len(vecs_before)} points …")
    proj_before = _reduce(vecs_before, method)
    proj_after  = _reduce(vecs_after,  method)

    cmap   = plt.cm.tab20
    n_samp = max(ids_before.max(), ids_after.max()) + 1
    norm   = plt.Normalize(0, n_samp - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                             gridspec_kw={"wspace": 0.35})

    for ax, proj, ids, title in [
        (axes[0], proj_before, ids_before,
         f"(a) Before Training\n({method.upper()} of random model embeddings)"),
        (axes[1], proj_after,  ids_after,
         f"(b) After Training\n({method.upper()} of trained model embeddings)"),
    ]:
        sc = ax.scatter(proj[:, 0], proj[:, 1],
                        c=ids, cmap=cmap, norm=norm,
                        s=8, alpha=0.55, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.tick_params(labelsize=7)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.015, pad=0.02,
    )
    cbar.set_label("Sample ID", fontsize=9)

    fig.suptitle(
        f"Visual Window Embedding Space — {num_samples} samples\n"
        "(each point = one image window; colour = sample id)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "fig09_embedding_space_before_after")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Embedding space visualisation (fig09)."
    )
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of synthetic sentences (max 10)")
    parser.add_argument("--output_dir",  default="paper_figures/outputs")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method",      default="tsne",
                        choices=["tsne", "umap", "pca"])
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_embedding_space(args.checkpoint, args.num_samples,
                          args.output_dir, args.device, args.method)


if __name__ == "__main__":
    main()

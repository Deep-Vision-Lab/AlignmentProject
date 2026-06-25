"""Embedding-space visualization for raw windows or D3TW-pooled characters."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from NormalizeFuncs import normalize_func
from utils.char_pooling import compute_d3tw_char_pool_for_sample
from utils.model_loading import (
    load_char_bank_if_available,
    load_image_model,
    load_sample,
    load_text_embedder,
)
from utils.sample_data import make_sample, pad_text
from utils.similarity import compute_image_embeddings
from utils.plotting import setup_paper_style, save_figure


def _reduce(embeddings, method, seed=42):
    method = method.lower()
    if len(embeddings) < 3:
        return np.zeros((len(embeddings), 2), dtype=np.float32)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(2, len(embeddings) // 5))
            return TSNE(
                n_components=2,
                random_state=seed,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
            ).fit_transform(embeddings)
        except Exception as exc:
            print(f"  [fig09] t-SNE unavailable/failed ({exc}); falling back to PCA.")
            method = "pca"
    if method == "umap":
        try:
            import umap
            return umap.UMAP(
                n_components=2,
                random_state=seed,
                n_neighbors=min(15, len(embeddings) - 1),
            ).fit_transform(embeddings)
        except Exception as exc:
            print(f"  [fig09] UMAP unavailable/failed ({exc}); falling back to PCA.")
            method = "pca"
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(embeddings)


def _sample(data_dir, idx):
    if data_dir:
        return load_sample(data_dir, idx, transform=True)
    return make_sample(idx, transform=True)


def _text_embeddings(chars, checkpoint, device):
    bank_emb, char_to_idx, idx_to_char = load_char_bank_if_available(checkpoint, device)
    if bank_emb is not None and char_to_idx is not None and all(ch in char_to_idx for ch in chars):
        ids = torch.tensor([char_to_idx[ch] for ch in chars], device=device, dtype=torch.long)
        return bank_emb[ids]
    print("  [fig09] Warning: char bank unavailable/incomplete; falling back to text embedder.")
    embedder = load_text_embedder(device)
    with torch.no_grad():
        return normalize_func(embedder("".join(chars)).to(device))


@torch.no_grad()
def collect_window_embeddings(model, num_samples, device, data_dir=None):
    vectors, labels, metadata = [], [], []
    for sample_idx in range(num_samples):
        image, _ = _sample(data_dir, sample_idx)
        emb = compute_image_embeddings(model, image, device)
        vectors.append(emb.detach().cpu().numpy())
        labels.extend([f"sample {sample_idx}"] * emb.shape[0])
        metadata.extend([
            {"sample_idx": sample_idx, "window_idx": int(i)}
            for i in range(emb.shape[0])
        ])
    return np.vstack(vectors), labels, metadata


@torch.no_grad()
def collect_pooled_char_embeddings(model, checkpoint, num_samples, device, data_dir=None):
    vectors, labels, metadata = [], [], []
    for sample_idx in range(num_samples):
        image, text = _sample(data_dir, sample_idx)
        chars = list(pad_text(text))
        visual_emb = normalize_func(model(image.unsqueeze(0).to(device)).squeeze(0).float())
        text_emb = _text_embeddings(chars, checkpoint, device)
        result = compute_d3tw_char_pool_for_sample(
            visual_emb=visual_emb,
            text_emb=text_emb,
            transcript_chars=chars,
            detach_assignment=True,
        )
        pooled = result["pooled_visual"].detach().cpu().numpy()
        vectors.append(pooled)
        labels.extend(chars)
        metadata.extend([
            {
                "sample_idx": int(sample_idx),
                "char_idx": int(j),
                "char": chars[j],
                "assigned_windows": [int(i) for i in result["groups"][j]],
                "num_windows": int(len(result["groups"][j])),
            }
            for j in range(len(chars))
        ])
    return np.vstack(vectors), labels, metadata


def draw_embedding_space(
    checkpoint,
    num_samples,
    output_dir,
    device,
    method,
    embedding_level="pooled_char",
    data_dir=None,
):
    setup_paper_style()
    device = torch.device(device)
    model = load_image_model(checkpoint, device)

    if embedding_level == "pooled_char":
        vectors, labels, metadata = collect_pooled_char_embeddings(
            model, checkpoint, num_samples, device, data_dir=data_dir
        )
        title = "D3TW-pooled visual character vectors"
        color_label = "true Arabic character"
        stem = "fig09_embedding_space_pooled_char"
    else:
        vectors, labels, metadata = collect_window_embeddings(
            model, num_samples, device, data_dir=data_dir
        )
        title = "Raw visual window embeddings"
        color_label = "sample id"
        stem = "fig09_embedding_space_window"

    print(f"  [fig09] Reducing {len(vectors)} {embedding_level} vectors with {method.upper()}")
    projection = _reduce(vectors, method)

    unique_labels = list(dict.fromkeys(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    color_ids = np.array([label_to_id[label] for label in labels])

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=color_ids,
        cmap="tab20",
        s=14 if embedding_level == "pooled_char" else 8,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_title(f"{title}\n(each point labeled by {color_label})", fontsize=13)
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02, label=color_label)

    if embedding_level == "pooled_char":
        for idx, (x, y) in enumerate(projection[: min(220, len(projection))]):
            ax.text(x, y, labels[idx] if labels[idx] != " " else "sp", fontsize=7, alpha=0.75)

    os.makedirs(output_dir, exist_ok=True)
    save_figure(fig, output_dir, stem)
    plt.close(fig)
    print(f"  [fig09] collected metadata for {len(metadata)} points (not saved; PDF-only output).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--embedding_level", default="pooled_char", choices=["window", "pooled_char"])
    parser.add_argument("--output_dir", default="paper_figures/outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap", "pca"])
    args = parser.parse_args()
    os.chdir(_ROOT)
    draw_embedding_space(**vars(args))


if __name__ == "__main__":
    main()

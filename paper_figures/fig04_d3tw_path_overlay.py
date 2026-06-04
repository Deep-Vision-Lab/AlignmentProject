"""
fig04_d3tw_path_overlay.py
===========================
Shows the hard-DTW alignment path overlaid on the trained similarity heatmap.

Also saves a JSON mapping:  char_index → window_indices

Output:
  fig04_d3tw_path_overlay_sample_{idx}.png / .pdf
  fig04_alignment_mapping_sample_{idx}.json
"""
import argparse
import json
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from utils.model_loading import load_image_model, load_text_embedder
from utils.sample_data import make_sample, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_windows, pad_text
from utils.similarity import compute_image_embeddings, compute_text_embeddings, compute_text_image_similarity
from utils.alignment import hard_dtw_path_restricted, path_to_char_window_mapping
from utils.plotting import setup_paper_style, save_figure, plot_similarity_with_path, attach_window_strip, arabic_label


def _get_sim(img_model, text_embedder, img_tensor, text, device):
    img_emb = compute_image_embeddings(img_model, img_tensor, device)
    txt_emb = compute_text_embeddings(text_embedder, text)
    sim     = compute_text_image_similarity(txt_emb, img_emb)
    S = sim.shape[1]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
    return sim


def draw_path_overlay(checkpoint, sentence_idx, output_dir, device):
    setup_paper_style()

    img_tensor, text = make_sample(sentence_idx)
    text_padded      = pad_text(text)
    text_embedder    = load_text_embedder(device)
    model            = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)

    sim    = _get_sim(model, text_embedder, img_tensor, text_padded, device)
    sim_np = sim.detach().cpu().numpy()

    path    = hard_dtw_path_restricted(sim_np)
    mapping = path_to_char_window_mapping(path, text_padded)
    chars   = list(text_padded)

    T, S    = sim_np.shape
    fig_h   = max(16, T * 0.75 + 5)
    fig, ax = plt.subplots(figsize=(28, fig_h))

    im = plot_similarity_with_path(
        ax, sim_np, path,
        title=(
            f"D3TW Alignment  |  Sentence {sentence_idx}\n"
            f'"{arabic_label(text)}"  —  T={T} chars  ×  S={S} windows'
        ),
        xlabel="Image window index",
        ylabel="Text character",
        yticklabels=chars,
        cmap="viridis",
        path_color="red", path_lw=2,
    )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="Cosine similarity")

    wins = get_fig_windows(sentence_idx)
    plt.tight_layout()
    fig.subplots_adjust(bottom=max(0.08, 0.9 / fig_h))
    attach_window_strip(ax, wins)
    save_figure(fig, output_dir, f"fig04_d3tw_path_overlay_s{sentence_idx}")
    plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"fig04_alignment_mapping_s{sentence_idx}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  Saved {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="D3TW alignment path overlay figure (fig04)."
    )
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--sentence_idx", type=int, default=0,
                        help="Index into the built-in Arabic sentence pool")
    parser.add_argument("--output_dir",   default="paper_figures/outputs")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_path_overlay(args.checkpoint, args.sentence_idx,
                      args.output_dir, args.device)


if __name__ == "__main__":
    main()

"""
fig12_window_feature_concentration.py
======================================
Diagnostic feature-concentration analysis for the trained Arabic image-
alignment model.

For each selected sliding-window patch this script produces Grad-CAM,
saliency and / or occlusion maps that answer:

  • Does the model focus on Arabic letter strokes?
  • Does it capture dots and diacritics?
  • Does it attend to inter-letter connection strokes?
  • Does it wrongly focus on background, borders, or whitespace?

See paper_figures/FEATURE_CONCENTRATION_README.md for interpretation guidance.

Usage examples
--------------
  # Grad-CAM only
  python paper_figures/fig12_window_feature_concentration.py \\
      --checkpoint Weights/JOB_ID/model_latest.pth \\
      --data_dir   DataSet/Synthetic_Arabic_100000 \\
      --sample_idx 1 --method gradcam --target_mode aligned_char

  # Saliency only, explicit windows
  python paper_figures/fig12_window_feature_concentration.py \\
      --checkpoint Weights/JOB_ID/model_latest.pth \\
      --data_dir   DataSet/Synthetic_Arabic_100000 \\
      --sample_idx 1 --window_indices 5 6 7 8 --method saliency

  # All methods
  python paper_figures/fig12_window_feature_concentration.py \\
      --checkpoint Weights/JOB_ID/model_latest.pth \\
      --data_dir   DataSet/Synthetic_Arabic_100000 \\
      --sample_idx 1 --method all --num_windows 4 --target_mode aligned_char

  # Occlusion with custom patch / stride
  python paper_figures/fig12_window_feature_concentration.py \\
      --checkpoint Weights/JOB_ID/model_latest.pth \\
      --data_dir   DataSet/Synthetic_Arabic_100000 \\
      --sample_idx 0 --method occlusion \\
      --occlusion_patch_size 4 --occlusion_stride 2

Output directory structure (default: paper_figures/outputs/feature_concentration/)
---------------------------
  sample_<idx>_window_<w>_original.{png,pdf}
  sample_<idx>_window_<w>_gradcam.{png,pdf}
  sample_<idx>_window_<w>_gradcam_overlay.{png,pdf}
  sample_<idx>_window_<w>_saliency.{png,pdf}
  sample_<idx>_window_<w>_saliency_overlay.{png,pdf}
  sample_<idx>_window_<w>_occlusion.{png,pdf}
  sample_<idx>_window_<w>_occlusion_overlay.{png,pdf}
  sample_<idx>_window_<w>_metadata.json
  sample_<idx>_feature_concentration_summary.json
  fig12_feature_concentration_sample_<idx>.{png,pdf}
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm_mod
from PIL import Image as PILImage

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from Parameters import window_size, stride_ratio, lang, vector_size
from embeddingModel import sliding_window
from utils.model_loading import load_image_model, load_text_embedder, load_sample
from utils.similarity import compute_text_embeddings, compute_text_image_similarity
from utils.alignment import hard_dtw_path_restricted
from utils.plotting import setup_paper_style, arabic_label
from utils.gradcam import (
    GradCAMExtractor, compute_saliency, compute_occlusion,
    normalize_map, resize_map, overlay_heatmap_on_image,
)

# ── Global derived constants ──────────────────────────────────────────────────
_STRIDE    = max(1, int(window_size * stride_ratio))
_IS_ARABIC = lang.lower() == "arabic"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Feature-concentration analysis for the image alignment model (fig12).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--checkpoint",  required=True,
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument("--data_dir",    required=True,
                   help="Dataset root containing images/ and texts/")
    p.add_argument("--sample_idx",  type=int, required=True,
                   help="0-based index of the sample to analyse")
    # Output
    p.add_argument("--output_dir",  default="paper_figures/outputs/feature_concentration")
    p.add_argument("--dpi",         type=int,  default=300)
    # Window selection
    p.add_argument("--window_indices", type=int, nargs="+", default=None,
                   help="Explicit model-order window indices to analyse")
    p.add_argument("--num_windows",    type=int, default=6,
                   help="Number of windows to analyse when --window_indices is omitted")
    # Analysis options
    p.add_argument("--method",       default="gradcam",
                   choices=["gradcam", "saliency", "occlusion", "all"],
                   help="Explanation method(s) to run")
    p.add_argument("--target_mode",  default="aligned_char",
                   choices=["aligned_char", "max_similarity", "embedding_norm"],
                   help="How to compute the target score for back-propagation")
    # Occlusion options
    p.add_argument("--occlusion_patch_size", type=int, default=8)
    p.add_argument("--occlusion_stride",     type=int, default=4)
    p.add_argument("--occlusion_value",      default="mean",
                   choices=["zero", "mean"])
    # Grad-CAM layer (layer3 is recommended for 16px windows; layer4 is too coarse)
    p.add_argument("--gradcam_layer", default="layer3",
                   choices=["layer1", "layer2", "layer3", "layer4"],
                   help="ResNet34 layer to hook for Grad-CAM. "
                        "layer3 gives 2x finer maps than layer4 for 16px windows.")
    # Window selection mode
    p.add_argument("--select_mode", default="evenly_spaced",
                   choices=["evenly_spaced", "foreground", "high_score"],
                   help="How to select windows to analyse. "
                        "'foreground': highest foreground pixel ratio. "
                        "'high_score': highest target similarity score.")
    # Misc
    p.add_argument("--save_metadata", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Window index helpers
# ─────────────────────────────────────────────────────────────────────────────

def window_pixel_cols(w, total_windows, stride, ws):
    """
    Return (col_start, col_end) in the original image pixel space for model
    window index w.

    For Arabic (use_flip=True):
        model w=0 = rightmost patch = LTR index (total_windows - 1)
        → col_start = (total_windows - 1 - w) * stride
    """
    ltr = (total_windows - 1 - w) if _IS_ARABIC else w
    col_start = ltr * stride
    return col_start, col_start + ws


def select_windows(total_windows, num_windows, window_indices=None):
    """
    Return a sorted list of model-order window indices (evenly spaced).
    Use select_windows_by_mode() for foreground / high_score modes.
    """
    if window_indices:
        bad = [w for w in window_indices if not (0 <= w < total_windows)]
        if bad:
            raise ValueError(
                f"Window indices {bad} are out of range [0, {total_windows - 1}]."
            )
        return sorted(set(window_indices))

    margin = max(0, int(total_windows * 0.10))
    lo = margin
    hi = total_windows - margin - 1
    n  = min(num_windows, hi - lo + 1)
    return sorted(set(int(x) for x in np.linspace(lo, hi, n)))


def select_windows_by_mode(windows_flipped, total_windows, num_windows,
                            select_mode, model, img_tensor, subfeat,
                            win_to_char, text_emb, device,
                            fg_threshold=30, window_indices=None):
    """
    Select window indices using different strategies.

    Modes:
        evenly_spaced  — existing behaviour.
        foreground     — highest foreground pixel ratio (brightest windows).
        high_score     — highest cosine similarity to aligned character.
    """
    if window_indices:
        return select_windows(total_windows, num_windows, window_indices)

    if select_mode == "evenly_spaced":
        return select_windows(total_windows, num_windows)

    if select_mode == "foreground":
        scores = []
        for w in range(total_windows):
            patch = windows_flipped[0, w]  # [C, H, W]
            gray  = patch.mean(0).cpu().numpy() * 255
            fg_ratio = float((gray > fg_threshold).mean())
            scores.append((fg_ratio, w))
        scores.sort(reverse=True)
        return sorted(w for _, w in scores[:num_windows])

    if select_mode == "high_score":
        if text_emb is None:
            return select_windows(total_windows, num_windows)
        scores = []
        with torch.no_grad():
            raw    = model(img_tensor)
            img_f  = F.normalize(raw[0], p=2, dim=-1)  # [total_seq, D]
        txt_f = F.normalize(text_emb, p=2, dim=-1)     # [T, D]
        for w in range(total_windows):
            w_start = w * subfeat
            w_end   = (w + 1) * subfeat
            win_emb = img_f[w_start:w_end].mean(0, keepdim=True)  # [1, D]
            win_emb = F.normalize(win_emb, p=2, dim=-1)
            char_idx = win_to_char.get(w)
            if char_idx is not None and char_idx < len(txt_f):
                score = F.cosine_similarity(win_emb, txt_f[char_idx:char_idx+1]).item()
            else:
                score = torch.matmul(win_emb, txt_f.T).max().item()
            scores.append((score, w))
        scores.sort(reverse=True)
        return sorted(w for _, w in scores[:num_windows])

    return select_windows(total_windows, num_windows)


def save_context_image(img_tensor, windows_flipped, selected, total_windows,
                        stride, ws, output_dir, sample_idx, dpi):
    """Save the full line image with coloured rectangles around selected windows."""
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(14, 2.0))
    ax.imshow(img_np, aspect="auto")
    ax.axis("off")

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(selected)))
    H = img_np.shape[0]
    for rank, (w, clr) in enumerate(zip(selected, colors)):
        col_start, col_end = window_pixel_cols(w, total_windows, stride, ws)
        import matplotlib.patches as mpatches
        ax.add_patch(mpatches.Rectangle(
            (col_start, 0), ws, H,
            linewidth=2, edgecolor=clr, facecolor=clr, alpha=0.18,
        ))
        ax.add_patch(mpatches.Rectangle(
            (col_start, 0), ws, H,
            linewidth=2, edgecolor=clr, facecolor="none",
        ))
        ax.text(col_start + ws / 2, H * 0.05, f"w{w}",
                ha="center", va="top", fontsize=7, color="white",
                bbox=dict(boxstyle="square,pad=0.1", fc=clr, alpha=0.7))

    ax.set_title(f"Selected windows  |  sample {sample_idx}", fontsize=9, pad=3)
    base = os.path.join(output_dir, f"sample_{sample_idx}_selected_windows_context")
    _save_arr(img_np, base, dpi)
    plt.tight_layout()
    fig.savefig(base + "_annotated.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + "_annotated.pdf",          bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {base}_annotated.{{png,pdf}}")


# ─────────────────────────────────────────────────────────────────────────────
# Score functions
# ─────────────────────────────────────────────────────────────────────────────

def make_score_fn(target_mode, w_start, w_end,
                   text_emb=None, char_idx=None):
    """
    Return score_fn: Callable(output [1, S, D]) → differentiable scalar.

    The model output already has LayerNorm applied but is not L2-normalised.
    We apply L2 normalisation inside score_fn to be consistent with the rest
    of the codebase (NormalizeFuncs.normalize_func with l2 type).
    """
    if target_mode == "embedding_norm":
        def _fn(output):
            return output[0, w_start:w_end, :].mean(0).norm()
        return _fn

    if target_mode == "max_similarity":
        assert text_emb is not None, "max_similarity requires text_emb"
        txt = text_emb.detach()   # [T, D], already L2-normalised

        def _fn(output):
            win = F.normalize(output[0, w_start:w_end, :].mean(0, keepdim=True),
                              p=2, dim=-1)   # [1, D]
            return torch.matmul(win, txt.T).max()
        return _fn

    if target_mode == "aligned_char":
        assert text_emb is not None and char_idx is not None, \
            "aligned_char requires text_emb and char_idx"
        char_vec = text_emb[char_idx].detach()   # [D], L2-normalised

        def _fn(output):
            win = F.normalize(output[0, w_start:w_end, :].mean(0, keepdim=True),
                              p=2, dim=-1)   # [1, D]
            return F.cosine_similarity(win, char_vec.unsqueeze(0)).squeeze()
        return _fn

    raise ValueError(f"Unknown target_mode: {target_mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Alignment (text ↔ window mapping via D3TW)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment(img_tensor, model, text_padded, text_embedder,
                       total_windows, subfeat, device):
    """
    DTW-based window → character alignment.

    Returns
    -------
    sim_np :     [T, num_windows] float32 numpy array (collapsed similarity).
    path :       list of (char_idx, win_idx) tuples.
    win_to_char: dict {win_idx → char_idx} — first char aligned to each window.
    chars :      list of characters in text_padded (including boundary spaces).
    text_emb :   [T, D] torch.Tensor on device, L2-normalised.
    """
    with torch.no_grad():
        raw = model(img_tensor)               # [1, total_seq, D]
        img_feat = F.normalize(raw[0], p=2, dim=-1)  # [total_seq, D]

    text_emb = compute_text_embeddings(text_embedder, text_padded)   # [T, D]

    # Collapse sub-features per window: [total_seq, D] → [num_windows, D]
    if subfeat > 1:
        img_win = img_feat.reshape(total_windows, subfeat, -1).mean(1)
    else:
        img_win = img_feat
    img_win = F.normalize(img_win, p=2, dim=-1)

    sim    = compute_text_image_similarity(text_emb, img_win)   # [T, W]
    sim_np = sim.detach().cpu().numpy()

    path = hard_dtw_path_restricted(sim_np)

    win_to_char: dict = {}
    for ci, wi in path:
        if wi not in win_to_char:   # keep first char mapped to this window
            win_to_char[wi] = ci

    return sim_np, path, win_to_char, list(text_padded), text_emb


# ─────────────────────────────────────────────────────────────────────────────
# I / O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_uint8(t_or_arr):
    """Convert [C, H, W] tensor or [H, W, C] array to uint8 [H, W, 3]."""
    if isinstance(t_or_arr, torch.Tensor):
        arr = t_or_arr.detach().cpu().float().numpy()
    else:
        arr = np.array(t_or_arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.max() <= 1.0 + 1e-4:
        arr = (arr * 255).clip(0, 255)
    return arr[..., :3].astype(np.uint8)


def _heatmap_rgb(hm, cmap="jet"):
    """Convert [H, W] float heatmap to [H, W, 3] uint8 via colormap."""
    return (cm_mod.get_cmap(cmap)(normalize_map(hm))[:, :, :3] * 255).astype(np.uint8)


def _save_arr(arr, base_path, dpi):
    """Save [H, W, 3] uint8 array as PNG and PDF."""
    pil = PILImage.fromarray(arr)
    pil.save(base_path + ".png")
    # PDF via matplotlib for vector-compatible output
    fig, ax = plt.subplots(
        figsize=(max(2, arr.shape[1] / 50), max(2, arr.shape[0] / 50))
    )
    ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.axis("off")
    fig.savefig(base_path + ".pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Single-window analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_window(w, model, img_tensor, windows_flipped,
                    total_windows, subfeat, stride, ws,
                    target_mode, method,
                    text_emb, win_to_char, chars,
                    occ_patch, occ_stride, occ_fill,
                    device, output_dir, sample_idx, dpi, save_metadata,
                    gradcam_layer="layer3", select_mode="evenly_spaced"):
    """
    Run all requested explanation methods for window w and write outputs.

    Returns
    -------
    meta : dict  — JSON-serialisable metadata for this window.
    patch_arr : [H, W_patch, 3] uint8 — display patch (upsampled to 128 px tall).
    maps : dict  — {method_name: [H, W] float32 heatmap in [0,1]}.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"sample_{sample_idx}_window_{w}")

    col_start, col_end = window_pixel_cols(w, total_windows, stride, ws)
    w_start = w * subfeat
    w_end   = (w + 1) * subfeat

    # ── Resolve aligned character ─────────────────────────────────────────────
    char_idx    = win_to_char.get(w) if win_to_char else None
    target_char = chars[char_idx] if (chars and char_idx is not None) else "?"

    # ── Score function ────────────────────────────────────────────────────────
    score_fn = make_score_fn(
        target_mode, w_start, w_end,
        text_emb=text_emb,
        char_idx=char_idx,
    )

    # ── Extract pixel patch (model-order window) ─────────────────────────────
    raw_patch = windows_flipped[0, w]       # [C, H_patch, W_patch]
    patch_u8  = _tensor_to_uint8(raw_patch)  # [H_patch, W_patch, 3]
    H_win, W_win = patch_u8.shape[:2]

    # Upsample for readable display — minimum 96 px wide, 512 px tall
    disp_h = 512
    disp_w = max(96, int(W_win * disp_h / H_win))
    patch_disp = np.array(
        PILImage.fromarray(patch_u8).resize((disp_w, disp_h), PILImage.NEAREST)
    )

    # ── Baseline score (no grad) ──────────────────────────────────────────────
    with torch.no_grad():
        baseline_out = model(img_tensor)
        target_score = score_fn(baseline_out).item()

    method_meta: dict = {}
    maps: dict        = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Grad-CAM
    # ─────────────────────────────────────────────────────────────────────────
    if method in ("gradcam", "all"):
        print(f"  [fig12] GradCAM  window={w}  layer={gradcam_layer}")
        try:
            extractor = GradCAMExtractor(model, layer_name=gradcam_layer)
            cam_raw   = extractor.compute(img_tensor, w, score_fn)
            cam       = resize_map(cam_raw, disp_h, disp_w)
            overlay   = overlay_heatmap_on_image(patch_disp, cam)
            maps["gradcam"] = cam
            method_meta["gradcam"] = {
                "saved": True,
                "gradcam_layer": gradcam_layer,
                "note": (
                    f"Full-model Grad-CAM: gradients propagate through "
                    f"BiLSTM → CNN → backbone ({gradcam_layer})"
                ),
            }
        except Exception as exc:
            print(f"    [fig12] GradCAM failed for window {w}: {exc}")
            method_meta["gradcam"] = {"saved": False, "error": str(exc)}
        finally:
            model.zero_grad()

    # ─────────────────────────────────────────────────────────────────────────
    # Saliency
    # ─────────────────────────────────────────────────────────────────────────
    if method in ("saliency", "all"):
        print(f"  [fig12] Saliency window={w}")
        try:
            sal     = compute_saliency(model, img_tensor, col_start, col_end, score_fn)
            sal_up  = resize_map(sal, disp_h, disp_w)
            overlay = overlay_heatmap_on_image(patch_disp, sal_up)
            maps["saliency"] = sal_up
            method_meta["saliency"] = {
                "saved": True,
                "note": (
                    "Full-image input-gradient saliency (max|grad| over channels), "
                    "cropped to window region."
                ),
            }
        except Exception as exc:
            print(f"    [fig12] Saliency failed for window {w}: {exc}")
            method_meta["saliency"] = {"saved": False, "error": str(exc)}
        finally:
            model.zero_grad()

    # ─────────────────────────────────────────────────────────────────────────
    # Occlusion
    # ─────────────────────────────────────────────────────────────────────────
    if method in ("occlusion", "all"):
        print(f"  [fig12] Occlusion window={w}  (patch={occ_patch}, stride={occ_stride})")
        try:
            occ, orig_s, max_drop = compute_occlusion(
                model, img_tensor, col_start, col_end,
                score_fn, H_win, W_win,
                patch_size=occ_patch,
                occ_stride=occ_stride,
                fill=occ_fill,
            )
            occ_up  = resize_map(occ, disp_h, disp_w)
            overlay = overlay_heatmap_on_image(patch_disp, occ_up)
            maps["occlusion"] = occ_up
            method_meta["occlusion"] = {
                "saved": True,
                "original_score": round(orig_s, 6),
                "max_score_drop": round(max_drop, 6),
            }
        except Exception as exc:
            print(f"    [fig12] Occlusion failed for window {w}: {exc}")
            method_meta["occlusion"] = {"saved": False, "error": str(exc)}

    # ── Per-window metadata ───────────────────────────────────────────────────
    # Foreground ratio: fraction of pixels brighter than threshold
    gray = patch_u8.mean(-1)   # [H_win, W_win]
    fg_ratio = float((gray > 30).mean())

    meta = {
        "sample_idx":         sample_idx,
        "window_idx":         w,
        "window_order":       "right-to-left" if _IS_ARABIC else "left-to-right",
        "pixel_col_start":    int(col_start),
        "pixel_col_end":      int(col_end),
        "target_mode":        target_mode,
        "target_char":        target_char,
        "target_char_index":  int(char_idx) if char_idx is not None else None,
        "visual_positions":   list(range(w_start, w_end)),
        "subfeat_per_window": int(subfeat),
        "target_score":       round(float(target_score), 6),
        "foreground_ratio":   round(fg_ratio, 4),
        "select_mode":        select_mode,
        "gradcam_layer":      gradcam_layer,
        "method_outputs":     method_meta,
    }

    return meta, patch_disp, maps


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_grid(results, sample_idx, method, output_dir, dpi, ckpt_name):
    """
    Save a single figure with all analysed windows as rows.

    Columns depend on the method:
      gradcam   → Original | Target char | GradCAM (raw) | Overlay
      saliency  → Original | Target char | Saliency (raw) | Overlay
      occlusion → Original | Target char | Occlusion (raw) | Overlay
      all       → Original | Target char | GradCAM | Saliency | Occlusion
    """
    setup_paper_style()

    n_rows = len(results)
    if n_rows == 0:
        return

    if method == "all":
        # Show overlays (not raw maps) for the "all" summary grid
        col_names = ["Original", "Target\nchar",
                     "GradCAM\noverlay", "Saliency\noverlay", "Occlusion\noverlay"]
        cmaps     = {"gradcam": "jet", "saliency": "hot", "occlusion": "plasma"}
    else:
        col_names = ["Original", "Target\nchar", method.capitalize(), "Overlay"]
        cmaps     = {method: {"gradcam": "jet", "saliency": "hot",
                               "occlusion": "plasma"}.get(method, "jet")}

    n_cols = len(col_names)
    cw     = 3.8   # inches per column
    rh     = 3.2   # inches per row
    fig_w  = n_cols * cw + 0.8
    fig_h  = n_rows * rh + 1.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.55, wspace=0.20)

    for row, (meta, patch_disp, maps) in enumerate(results):
        w          = meta["window_idx"]
        tgt_char   = meta["target_char"]
        tgt_score  = meta["target_score"]
        col_start  = meta["pixel_col_start"]
        tgt_ci     = meta["target_char_index"]
        tgt_mode   = meta["target_mode"]

        col = 0

        # ── Original patch ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, col]);  col += 1
        ax.imshow(patch_disp, aspect="auto", interpolation="nearest")
        ax.set_title(
            f"w={w}  col={col_start}\n{tgt_mode[:12]}",
            fontsize=8.5, pad=3
        )
        ax.axis("off")

        # ── Target character panel ───────────────────────────────────────────
        ax = fig.add_subplot(gs[row, col]);  col += 1
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")
        try:
            disp_ch = arabic_label(tgt_char) if tgt_char != " " else "⎵"
        except Exception:
            disp_ch = "⎵" if tgt_char == " " else tgt_char
        ax.text(0.5, 0.58, disp_ch, ha="center", va="center",
                fontsize=42, color="white", transform=ax.transAxes)
        lbl = f"char[{tgt_ci}]\nscore={tgt_score:.3f}"
        ax.text(0.5, 0.10, lbl, ha="center", va="bottom",
                fontsize=7.5, color="#aaaaaa", transform=ax.transAxes)
        ax.set_title("Aligned\nchar", fontsize=8.5, pad=3)

        # ── Method columns ───────────────────────────────────────────────────
        if method == "all":
            # Show overlay for each method (not raw map) in the "all" grid
            for mname, cmap_name in [("gradcam",  "jet"),
                                     ("saliency", "hot"),
                                     ("occlusion","plasma")]:
                ax = fig.add_subplot(gs[row, col]);  col += 1
                if mname in maps:
                    ov = overlay_heatmap_on_image(patch_disp, maps[mname])
                    ax.imshow(ov, aspect="auto", interpolation="bilinear")
                    ax.set_title(f"{mname.capitalize()}\noverlay", fontsize=8.5, pad=3)
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                            fontsize=11, transform=ax.transAxes)
                    ax.set_title(f"{mname.capitalize()}\noverlay", fontsize=8.5, pad=3)
                ax.axis("off")
        else:
            cmap_name = {"gradcam": "jet", "saliency": "hot",
                         "occlusion": "plasma"}.get(method, "jet")
            # Raw heatmap
            ax = fig.add_subplot(gs[row, col]);  col += 1
            if method in maps:
                ax.imshow(_heatmap_rgb(maps[method], cmap_name),
                          aspect="auto", interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=9, transform=ax.transAxes)
            ax.set_title(method.capitalize(), fontsize=8.5, pad=3)
            ax.axis("off")
            # Overlay
            ax = fig.add_subplot(gs[row, col])
            if method in maps:
                ov = overlay_heatmap_on_image(patch_disp, maps[method])
                ax.imshow(ov, aspect="auto", interpolation="bilinear")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=11, transform=ax.transAxes)
            ax.set_title("Overlay", fontsize=8.5, pad=3)
            ax.axis("off")

    fig.suptitle(
        f"Feature Concentration — Sample {sample_idx}  |  "
        f"{method}  |  {ckpt_name}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    base = os.path.join(output_dir,
                        f"fig12_feature_concentration_sample_{sample_idx}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    print(f"  Saved {base}.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.chdir(_PROJ_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n[fig12] Sample idx  : {args.sample_idx}")
    print(f"[fig12] Checkpoint  : {args.checkpoint}")
    print(f"[fig12] data_dir    : {args.data_dir}")
    print(f"[fig12] method      : {args.method}")
    print(f"[fig12] target_mode : {args.target_mode}")
    print(f"[fig12] device      : {device}")
    print()

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print("[fig12] Loading model …")
    model = load_image_model(args.checkpoint, device)
    model.eval()
    model_window_size = int(getattr(model, "window_size", window_size))
    model_stride = int(getattr(model, "stride", _STRIDE))
    print(f"  model window_size={model_window_size} stride={model_stride}")

    # ── 2. Load sample ─────────────────────────────────────────────────────────
    print("[fig12] Loading sample …")
    img_tensor, raw_text = load_sample(args.data_dir, args.sample_idx,
                                        transform=True)
    text_padded = " " + raw_text.strip() + " "
    print(f"  text (padded): {text_padded!r}  ({len(text_padded)} chars)")

    img_tensor = img_tensor.unsqueeze(0).to(device)   # [1, C, H, W]

    # ── 3. Extract windows ──────────────────────────────────────────────────────
    with torch.no_grad():
        windows_raw = sliding_window(img_tensor, model_window_size, model_stride)
    # windows_raw: [1, num_windows_ltr, C, H, window_size]

    if _IS_ARABIC:
        windows_flipped = torch.flip(windows_raw, dims=[1])
    else:
        windows_flipped = windows_raw

    total_windows = windows_flipped.shape[1]
    print(f"  total windows (model order): {total_windows}")

    # ── 4. Determine sub-features per window ──────────────────────────────────
    with torch.no_grad():
        probe = model(img_tensor)   # [1, total_seq, D]
    total_seq = probe.shape[1]

    if total_seq % total_windows == 0:
        subfeat = total_seq // total_windows
    else:
        subfeat = round(total_seq / total_windows)
        print(
            f"  WARNING: total_seq={total_seq} not divisible by "
            f"total_windows={total_windows}. "
            f"Using approximate subfeat={subfeat}."
        )
    print(f"  total_seq={total_seq}  subfeat_per_window={subfeat}")

    # ── 5. Alignment (for aligned_char mode) ───────────────────────────────────
    text_emb   = None
    win_to_char = {}
    chars       = []

    if args.target_mode in ("aligned_char", "max_similarity"):
        print("[fig12] Loading text embedder …")
        text_embedder = load_text_embedder(device)
        print("[fig12] Computing alignment …")
        _, _, win_to_char, chars, text_emb = compute_alignment(
            img_tensor, model, text_padded, text_embedder,
            total_windows, subfeat, device,
        )
        if args.target_mode == "max_similarity":
            win_to_char = {}   # alignment not used for char targeting in max_sim mode
    elif args.target_mode == "embedding_norm":
        chars = list(text_padded)   # still useful for display

    # ── 6. Select windows ──────────────────────────────────────────────────────
    selected = select_windows_by_mode(
        windows_flipped=windows_flipped,
        total_windows=total_windows,
        num_windows=args.num_windows,
        select_mode=args.select_mode,
        model=model,
        img_tensor=img_tensor,
        subfeat=subfeat,
        win_to_char=win_to_char,
        text_emb=text_emb,
        device=device,
        window_indices=args.window_indices,
    )
    print(f"[fig12] Selected windows ({args.select_mode}): {selected}")

    # ── 7. Analyse each window ─────────────────────────────────────────────────
    results = []
    for w in selected:
        print(f"\n[fig12] ── window {w} / {total_windows - 1} ──")
        meta, patch_disp, maps = analyse_window(
            w=w,
            model=model,
            img_tensor=img_tensor,
            windows_flipped=windows_flipped,
            total_windows=total_windows,
            subfeat=subfeat,
            stride=model_stride,
            ws=model_window_size,
            target_mode=args.target_mode,
            method=args.method,
            text_emb=text_emb,
            win_to_char=win_to_char,
            chars=chars,
            occ_patch=args.occlusion_patch_size,
            occ_stride=args.occlusion_stride,
            occ_fill=args.occlusion_value,
            device=device,
            output_dir=args.output_dir,
            sample_idx=args.sample_idx,
            dpi=args.dpi,
            save_metadata=args.save_metadata,
            gradcam_layer=args.gradcam_layer,
            select_mode=args.select_mode,
        )
        results.append((meta, patch_disp, maps))

    # ── 8. Summary grid ────────────────────────────────────────────────────────
    print("\n[fig12] Saving summary grid …")
    save_summary_grid(
        results=results,
        sample_idx=args.sample_idx,
        method=args.method,
        output_dir=args.output_dir,
        dpi=args.dpi,
        ckpt_name=os.path.basename(args.checkpoint),
    )

    # ── 9. Sample-level summary metadata ──────────────────────────────────────
    if False and args.save_metadata:
        summary = {
            "sample_idx":    args.sample_idx,
            "transcript":    text_padded,
            "raw_text":      raw_text,
            "checkpoint":    args.checkpoint,
            "data_dir":      args.data_dir,
            "method":        args.method,
            "target_mode":   args.target_mode,
            "model_config": {
                "window_size":    model_window_size,
                "stride":         model_stride,
                "stride_ratio":   float(model_stride / max(model_window_size, 1)),
                "vector_size":    vector_size,
                "lang":           lang,
                "use_flip":       _IS_ARABIC,
                "total_windows":  total_windows,
                "subfeat_per_window": subfeat,
                "total_seq_len":  total_seq,
            },
            "selected_windows": selected,
            "window_results": [m for m, _, _ in results],
            "generated_files": {
                "summary_png": os.path.join(
                    args.output_dir,
                    f"fig12_feature_concentration_sample_{args.sample_idx}.png"
                ),
                "summary_pdf": os.path.join(
                    args.output_dir,
                    f"fig12_feature_concentration_sample_{args.sample_idx}.pdf"
                ),
            },
        }
        sum_path = os.path.join(
            args.output_dir,
            f"sample_{args.sample_idx}_feature_concentration_summary.json"
        )
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Saved {sum_path}")

    print("\n[fig12] Done.")


if __name__ == "__main__":
    main()

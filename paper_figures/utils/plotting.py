"""Matplotlib helpers for publication-quality paper figures."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# ── Optional Arabic text shaping ─────────────────────────────────────────────
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi_display

    def _arabic_label(text):
        return _bidi_display(arabic_reshaper.reshape(str(text)))

    _HAS_ARABIC = True
except ImportError:
    def _arabic_label(text):
        return str(text)
    _HAS_ARABIC = False


def arabic_label(text):
    """Public wrapper: reshape + bidi-reorder an Arabic string for matplotlib."""
    return _arabic_label(text)


def setup_paper_style():
    """Apply consistent paper-style rcParams."""
    plt.rcParams.update({
        "font.family":        "sans-serif",   # DejaVu Sans supports Arabic glyphs
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    1.5,
        "patch.linewidth":    0.8,
    })


def save_figure(fig, output_dir, name):
    """
    Save *fig* as both high-res PNG and PDF.

    Args:
        fig:        matplotlib Figure.
        output_dir: Directory (created automatically).
        name:       Base filename without extension.
    """
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved {png_path}")
    print(f"  Saved {pdf_path}")


def plot_similarity_heatmap(ax, matrix, title="", xlabel="Image windows",
                             ylabel="Text characters", xticklabels=None,
                             yticklabels=None, vmin=None, vmax=None,
                             cmap="viridis"):
    """
    Plot a [T, S] similarity matrix as a heatmap.

    Returns the AxesImage (use for a shared colorbar).
    """
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.array(matrix, dtype=np.float32)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    rows, cols = matrix.shape

    if yticklabels is not None:
        step = max(1, len(yticklabels) // 30)
        ticks = list(range(0, len(yticklabels), step))
        labels = [("_" if yticklabels[t] == " " else _arabic_label(str(yticklabels[t])))
                  for t in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
    if xticklabels is not None:
        step = max(1, len(xticklabels) // 20)
        ticks = list(range(0, len(xticklabels), step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(xticklabels[t]) for t in ticks],
                           rotation=45, fontsize=7)

    # Cell separator grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Y-axis: small visual breathing room (actual boundary rows come from pad_text)
    ax.set_ylim(rows - 0.5 + 0.2, -0.5 - 0.2)

    # Cell value annotations — skip when matrix is too dense to be legible
    if max(rows, cols) <= 64:
        v_lo = matrix.min() if vmin is None else vmin
        v_hi = matrix.max() if vmax is None else vmax
        span = max(v_hi - v_lo, 1e-8)
        fontsize = max(6, min(11, int(320 / max(rows, cols))))
        for r in range(rows):
            for c in range(cols):
                val   = matrix[r, c]
                norm  = (val - v_lo) / span
                color = "white" if norm < 0.6 else "#111111"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=fontsize, color=color, fontweight="bold")

    return im


def plot_similarity_with_path(ax, matrix, path, title="",
                               xlabel="Image windows",
                               ylabel="Text characters",
                               yticklabels=None, vmin=None, vmax=None,
                               cmap="viridis", path_color="red", path_lw=2):
    """
    Plot similarity heatmap with a DTW alignment path overlaid.

    Args:
        path: list of (i, j) tuples.
    """
    im = plot_similarity_heatmap(
        ax, matrix, title=title, xlabel=xlabel, ylabel=ylabel,
        yticklabels=yticklabels, vmin=vmin, vmax=vmax, cmap=cmap,
    )
    if path:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        ax.plot(cols, rows, color=path_color, linewidth=path_lw,
                alpha=0.85, label="Alignment path")
        ax.legend(loc="upper left", fontsize=8)
    return im


def attach_window_strip(ax, patches):
    """
    Add a row of individually framed window-patch images below the x-axis.

    Each patch is shown as a separate, bordered thumbnail with visible gaps
    between neighbours so they appear as distinct elements.

    Must be called AFTER plt.tight_layout() and plt.subplots_adjust().
    Returns the newly created strip Axes.
    """
    n = len(patches)
    if not n:
        return None

    ax.set_xticks([])
    ax.set_xlabel("")

    fig   = ax.get_figure()
    pos   = ax.get_position()
    fig_h = fig.get_figheight()
    s_h   = 0.8 / fig_h               # 0.8-inch strip height as fig fraction
    gap   = 0.008
    bot   = max(0.0, pos.y0 - gap - s_h)

    strip = fig.add_axes([pos.x0, bot, pos.width, s_h])
    strip.set_xlim(-0.5, n - 0.5)
    strip.set_ylim(0, 1)
    strip.set_facecolor("#f0f0f0")     # light grey background makes gaps visible
    strip.axis("off")

    # Each patch occupies [j-0.42, j+0.42] × [0.06, 0.94] — 16% horizontal gap
    pad_x, pad_y = 0.42, 0.06
    for j, patch in enumerate(patches):
        arr = np.asarray(patch, dtype=np.uint8)
        x0, x1 = j - pad_x, j + pad_x
        strip.imshow(arr, extent=(x0, x1, pad_y, 1 - pad_y),
                     aspect="auto", interpolation="nearest", zorder=2)
        # Thin border around each patch
        strip.add_patch(mpatches.FancyBboxPatch(
            (x0, pad_y), x1 - x0, 1 - 2 * pad_y,
            boxstyle="square,pad=0",
            linewidth=0.5, edgecolor="#555555", facecolor="none", zorder=3,
        ))

    return strip


def plot_line_with_windows(ax, pil_image, window_size, stride,
                            show_n_windows=12, title="Arabic Line Image",
                            highlight_color="lime", use_flip=True):
    """
    Display a PIL image with highlighted sliding-window boxes.

    Args:
        pil_image:       PIL Image.
        window_size:     Width of each patch in pixels.
        stride:          Stride between patches.
        show_n_windows:  Number of windows to highlight (evenly spaced).
        use_flip:        If True, label windows right-to-left (Arabic RTL).
    """
    img_arr = np.array(pil_image)
    ax.imshow(img_arr, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    W = img_arr.shape[1]
    num_windows = max(1, (W - window_size) // stride + 1)
    H = img_arr.shape[0]

    idxs = np.linspace(0, num_windows - 1,
                       min(show_n_windows, num_windows), dtype=int)
    idxs = np.unique(idxs)

    for idx in idxs:
        x = int(idx) * stride
        # Semi-transparent fill
        ax.add_patch(mpatches.Rectangle(
            (x, 0), window_size, H,
            linewidth=0, facecolor=highlight_color, alpha=0.12,
        ))
        # Opaque border
        ax.add_patch(mpatches.Rectangle(
            (x, 0), window_size, H,
            linewidth=1.5, edgecolor=highlight_color, facecolor="none",
        ))
        label = (num_windows - 1 - int(idx)) if use_flip else int(idx)
        ax.text(x + window_size / 2, H * 0.05,
                f"w{label}", ha="center", va="top",
                fontsize=6, color="white",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", alpha=0.55))

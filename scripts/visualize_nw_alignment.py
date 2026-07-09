#!/usr/bin/env python3
"""
Visualize image-to-image alignment between two line images using:
1) the trained image encoder to get window embeddings
2) Needleman-Wunsch global alignment on the window similarity matrix
3) colored masks + arrows for consecutive aligned regions

Example:
python scripts/visualize_nw_alignment.py \
    --line1 path/to/line1.png \
    --line2 path/to/line2.png \
    --weights Weights/my_job/model_latest.pth \
    --output outputs/nw_alignment.png \
    --window-size 32 \
    --stride 16 \
    --height 128 \
    --sim-threshold 0.25 \
    --gap-penalty 0.2 \
    --use-flip
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# Adjust the import if needed according to your repo structure.
from embeddingModel import EmbeddingModel


@dataclass
class AlignedPair:
    i: int
    j: int
    score: float


@dataclass
class AlignedGroup:
    line1_start: int
    line1_end: int
    line2_start: int
    line2_end: int
    scores: List[float]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def load_checkpoint_image_model(
    weights_path: str,
    device: str,
    window_size: int,
    stride: int,
    vector_size: int = 128,
    use_bilstm: bool = True,
    use_flip: bool = False,
):
    checkpoint = torch.load(weights_path, map_location=device)

    model_config = checkpoint.get("model_config", {})
    vector_size = model_config.get("vector_size", vector_size)
    use_bilstm = model_config.get("use_bilstm", use_bilstm)

    model = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
        use_bilstm=use_bilstm,
        bilstm_layers=model_config.get("bilstm_layers", 2),
        bilstm_hidden_dim=model_config.get("bilstm_hidden_dim", vector_size),
    ).to(device)

    state = checkpoint.get("image_model_state_dict")
    if state is None:
        state = checkpoint.get("model_state_dict")
    if state is None:
        state = checkpoint

    model.load_state_dict(state, strict=False)
    model.eval()
    return model, model_config


def preprocess_line_image(image_path: str, target_height: int = 128):
    """
    Loads a line image, converts to RGB, and resizes height to target_height
    while preserving aspect ratio.
    Returns:
        pil_img_resized
        torch_tensor [3, H, W]
    """
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    pil_img = pil_img.resize((new_w, target_height), Image.BILINEAR)

    to_tensor = transforms.ToTensor()
    tensor = to_tensor(pil_img)  # [3, H, W]
    return pil_img, tensor


@torch.no_grad()
def extract_window_embeddings(model, image_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    image_tensor: [3, H, W]
    returns: [num_windows, D]
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    emb = model(image_tensor)  # [1, num_windows, D]
    emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.squeeze(0).cpu()


def compute_similarity_matrix(emb1: torch.Tensor, emb2: torch.Tensor) -> np.ndarray:
    """
    emb1: [N, D]
    emb2: [M, D]
    returns cosine similarity matrix [N, M]
    """
    sim = emb1 @ emb2.T
    return sim.numpy()


def needleman_wunsch(score_matrix: np.ndarray, gap_penalty: float = 0.2):
    """
    Global alignment maximizing:
        alignment_score = sum(match_scores) - gap_penalty * num_gaps

    score_matrix[i, j] = similarity score between window i of line1 and window j of line2

    Returns:
        path: list of tuples:
            ("diag", i, j) for matched windows
            ("up", i, None) for gap in line2
            ("left", None, j) for gap in line1
        dp: DP matrix
    """
    n, m = score_matrix.shape
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    ptr = np.zeros((n + 1, m + 1), dtype=np.int32)
    # ptr codes:
    # 0 = diag, 1 = up, 2 = left

    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] - gap_penalty
        ptr[i, 0] = 1

    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] - gap_penalty
        ptr[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1, j - 1] + score_matrix[i - 1, j - 1]
            up = dp[i - 1, j] - gap_penalty
            left = dp[i, j - 1] - gap_penalty

            best = max(diag, up, left)
            dp[i, j] = best
            if best == diag:
                ptr[i, j] = 0
            elif best == up:
                ptr[i, j] = 1
            else:
                ptr[i, j] = 2

    # Backtrace
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ptr[i, j] == 0:
            path.append(("diag", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or ptr[i, j] == 1):
            path.append(("up", i - 1, None))
            i -= 1
        else:
            path.append(("left", None, j - 1))
            j -= 1

    path.reverse()
    return path, dp


def extract_aligned_pairs(
    path: Sequence[Tuple[str, Optional[int], Optional[int]]],
    score_matrix: np.ndarray,
    sim_threshold: float = 0.25,
) -> List[AlignedPair]:
    """
    Keep only diagonal alignments whose similarity is high enough.
    """
    pairs = []
    for op, i, j in path:
        if op != "diag":
            continue
        score = float(score_matrix[i, j])
        if score >= sim_threshold:
            pairs.append(AlignedPair(i=i, j=j, score=score))
    return pairs


def group_consecutive_pairs(pairs: Sequence[AlignedPair]) -> List[AlignedGroup]:
    """
    Merge consecutive diagonal matches:
      (i, j), (i+1, j+1), (i+2, j+2), ...
    into one aligned group with one color.
    """
    if not pairs:
        return []

    groups = []
    current = AlignedGroup(
        line1_start=pairs[0].i,
        line1_end=pairs[0].i,
        line2_start=pairs[0].j,
        line2_end=pairs[0].j,
        scores=[pairs[0].score],
    )

    for prev, cur in zip(pairs[:-1], pairs[1:]):
        if cur.i == prev.i + 1 and cur.j == prev.j + 1:
            current.line1_end = cur.i
            current.line2_end = cur.j
            current.scores.append(cur.score)
        else:
            groups.append(current)
            current = AlignedGroup(
                line1_start=cur.i,
                line1_end=cur.i,
                line2_start=cur.j,
                line2_end=cur.j,
                scores=[cur.score],
            )

    groups.append(current)
    return groups


def window_idx_to_pixel_span(
    start_idx: int,
    end_idx: int,
    num_windows: int,
    window_size: int,
    stride: int,
    image_width: int,
    use_flip: bool = False,
) -> Tuple[int, int]:
    """
    Converts a range of window indices [start_idx, end_idx] to pixel x-range [x0, x1].
    If use_flip=True, the model processed windows in reversed order, so map back to
    displayed image coordinates.
    """
    if use_flip:
        disp_start = num_windows - 1 - end_idx
        disp_end = num_windows - 1 - start_idx
    else:
        disp_start = start_idx
        disp_end = end_idx

    x0 = disp_start * stride
    x1 = disp_end * stride + window_size

    x0 = max(0, min(x0, image_width - 1))
    x1 = max(x0 + 1, min(x1, image_width))
    return x0, x1


def draw_alignment_figure(
    img1: Image.Image,
    img2: Image.Image,
    groups: Sequence[AlignedGroup],
    num_windows1: int,
    num_windows2: int,
    window_size: int,
    stride: int,
    output_path: str,
    use_flip: bool = False,
    title: Optional[str] = None,
):
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    h1, w1 = arr1.shape[:2]
    h2, w2 = arr2.shape[:2]

    canvas_w = max(w1, w2)
    gap_y = 80
    top_margin = 20
    bottom_margin = 20

    y1_top = top_margin
    y1_bottom = y1_top + h1

    y2_top = y1_bottom + gap_y
    y2_bottom = y2_top + h2

    fig_h = (y2_bottom + bottom_margin) / 100.0
    fig_w = canvas_w / 100.0

    fig, ax = plt.subplots(figsize=(max(fig_w, 10), max(fig_h, 4)))
    ax.imshow(arr1, extent=(0, w1, y1_bottom, y1_top))
    ax.imshow(arr2, extent=(0, w2, y2_bottom, y2_top))

    ax.text(0, y1_top - 6, "Line 1", fontsize=12, weight="bold")
    ax.text(0, y2_top - 6, "Line 2", fontsize=12, weight="bold")

    cmap = plt.get_cmap("tab20")

    for idx, group in enumerate(groups):
        color = cmap(idx % 20)

        x1a, x1b = window_idx_to_pixel_span(
            group.line1_start, group.line1_end,
            num_windows1, window_size, stride, w1, use_flip=use_flip
        )
        x2a, x2b = window_idx_to_pixel_span(
            group.line2_start, group.line2_end,
            num_windows2, window_size, stride, w2, use_flip=use_flip
        )

        # colored masks
        rect1 = Rectangle(
            (x1a, y1_top),
            x1b - x1a,
            h1,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            alpha=0.28,
        )
        rect2 = Rectangle(
            (x2a, y2_top),
            x2b - x2a,
            h2,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            alpha=0.28,
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        # arrow from line 1 segment to line 2 segment
        cx1 = 0.5 * (x1a + x1b)
        cx2 = 0.5 * (x2a + x2b)

        arrow = FancyArrowPatch(
            (cx1, y1_bottom + 2),
            (cx2, y2_top - 2),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2,
            color=color,
            alpha=0.95,
        )
        ax.add_patch(arrow)

        # optional segment label
        ax.text(cx1, y1_top - 10, f"{idx}", color=color, fontsize=10, ha="center")
        ax.text(cx2, y2_top - 10, f"{idx}", color=color, fontsize=10, ha="center")

    if title:
        ax.set_title(title, fontsize=14)

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(y2_bottom + bottom_margin, 0)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line1", required=True, help="Path to first line image")
    parser.add_argument("--line2", required=True, help="Path to second line image")
    parser.add_argument("--weights", required=True, help="Path to trained weights .pth")
    parser.add_argument("--output", required=True, help="Path to output figure")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--gap-penalty", type=float, default=0.2)
    parser.add_argument("--sim-threshold", type=float, default=0.25)
    parser.add_argument("--use-bilstm", action="store_true")
    parser.add_argument("--use-flip", action="store_true")
    parser.add_argument("--title", default="Needleman–Wunsch Alignment Visualization")
    args = parser.parse_args()

    model, model_config = load_checkpoint_image_model(
        weights_path=args.weights,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
        vector_size=args.vector_size,
        use_bilstm=args.use_bilstm,
        use_flip=args.use_flip,
    )

    img1_pil, img1_tensor = preprocess_line_image(args.line1, target_height=args.height)
    img2_pil, img2_tensor = preprocess_line_image(args.line2, target_height=args.height)

    emb1 = extract_window_embeddings(model, img1_tensor, device=args.device)
    emb2 = extract_window_embeddings(model, img2_tensor, device=args.device)

    sim = compute_similarity_matrix(emb1, emb2)
    path, dp = needleman_wunsch(sim, gap_penalty=args.gap_penalty)
    pairs = extract_aligned_pairs(path, sim, sim_threshold=args.sim_threshold)
    groups = group_consecutive_pairs(pairs)

    print(f"line1 windows: {emb1.shape[0]}")
    print(f"line2 windows: {emb2.shape[0]}")
    print(f"num diagonal aligned pairs above threshold: {len(pairs)}")
    print(f"num consecutive aligned groups: {len(groups)}")

    draw_alignment_figure(
        img1=img1_pil,
        img2=img2_pil,
        groups=groups,
        num_windows1=emb1.shape[0],
        num_windows2=emb2.shape[0],
        window_size=args.window_size,
        stride=args.stride,
        output_path=args.output,
        use_flip=args.use_flip,
        title=args.title,
    )

    print(f"saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
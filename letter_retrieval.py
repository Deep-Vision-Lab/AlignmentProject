"""
letter_retrieval.py

Generates n*m Arabic letter images (n random letters × m augmented variants),
embeds them with the trained EmbeddingModel, then retrieves the top-k gallery
images closest to a query image.

Usage
-----
python letter_retrieval.py \
    --n_letters 10 \
    --m_images  20 \
    --top_k     10 \
    --query     "ب"                  # a single Arabic letter  OR  path to a PNG
    --weights   Weights/localRun/model_latest.pth

All other model hyper-parameters are read from Parameters.py.
"""

import argparse
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from Parameters import (
    window_size, stride_ratio, vector_size, device,
    use_bilstm, bilstm_layers, bilstm_hidden_dim, model_dropout,
)
from embeddingModel import EmbeddingModel

# ─── Constants ────────────────────────────────────────────────────────────────

IMG_W, IMG_H = 1024, 128          # must match training resolution
FONT_DIR      = "Fonts"
# Amiri first — it has the most complete Arabic Unicode coverage including
# all isolated letter presentation forms (FE8x–FExx range).
FONTS         = [
    "Amiri-Regular.ttf",
    "Arslan_Wessam_B.ttf",
    "Arslan_Wessam.ttf",
    "Alexandria.ttf",
    "Rakkas-Regular.ttf",
    "Sahel.ttf",
    "Vibes-Regular.ttf",
]

# Isolated Arabic letters (Unicode order, no diacritics)
ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

# ─── Image rendering ──────────────────────────────────────────────────────────

def _shape_text(text: str) -> str:
    reshaper = arabic_reshaper.ArabicReshaper(configuration={
        "delete_harakat": True,
        "support_zwj": False,
    })
    return get_display(reshaper.reshape(text))


def _render_letter(letter: str, font_path: str, font_size: int = 90) -> Image.Image:
    """Render a single Arabic letter and stretch it to fill 1024×128.

    Draws the letter on a natural-sized canvas, tight-crops to the glyph
    (+ small padding), then resizes to IMG_W×IMG_H.  This matches the way
    generateDataArabic.py renders full sentences — the letter fills the frame
    so the model sees an image distribution close to training.
    """
    shaped  = _shape_text(letter)
    PADDING = 10  # px of black border around the glyph

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Measure glyph on scratch canvas
    scratch = Image.new("RGB", (2000, 500), "black")
    draw    = ImageDraw.Draw(scratch)
    bb      = draw.textbbox((PADDING, PADDING), shaped, font=font)
    tw, th  = bb[2] - bb[0], bb[3] - bb[1]

    # Draw on a canvas just big enough for the glyph
    cw = tw + PADDING * 2
    ch = th + PADDING * 2
    canvas = Image.new("RGB", (max(cw, 1), max(ch, 1)), "black")
    ImageDraw.Draw(canvas).text((PADDING, PADDING), shaped, fill="white", font=font)

    # Resize to training dimensions (stretches letter to fill width)
    return canvas.resize((IMG_W, IMG_H), Image.LANCZOS)


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply random visual augmentation to a PIL image."""
    # 1. Random font size already chosen by caller — nothing needed here.

    # 2. Slight rotation (±6°)
    angle = rng.uniform(-6, 6)
    img   = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    # 3. Brightness / contrast jitter
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))

    # 4. Occasional blur
    if rng.random() < 0.3:
        radius = rng.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 5. Additive Gaussian noise
    arr   = np.array(img, dtype=np.float32)
    noise = rng.gauss(0, 1) * np.random.normal(0, rng.uniform(5, 20), arr.shape)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    # 6. Random horizontal crop/shift (±5 % of width)
    shift = int(rng.uniform(-0.05, 0.05) * IMG_W)
    if shift != 0:
        img = img.transform(
            (IMG_W, IMG_H), Image.AFFINE,
            (1, 0, shift, 0, 1, 0),
            resample=Image.BILINEAR, fillcolor=(0, 0, 0),
        )
    return img


def generate_gallery(
    n_letters: int,
    m_images: int,
    rng: random.Random,
) -> tuple[list[Image.Image], list[str]]:
    """
    Returns:
        images  – list of n*m PIL Images
        labels  – list of n*m str (one letter label per image)
    """
    letters = rng.sample(ARABIC_LETTERS, min(n_letters, len(ARABIC_LETTERS)))
    print(f"Sampled {len(letters)} letters: {'  '.join(letters)}")

    available_fonts = [
        os.path.join(FONT_DIR, f) for f in FONTS
        if os.path.exists(os.path.join(FONT_DIR, f))
    ]
    if not available_fonts:
        raise FileNotFoundError(f"No fonts found in {FONT_DIR}/")

    images, labels = [], []
    for letter in letters:
        for j in range(m_images):
            font_path = available_fonts[j % len(available_fonts)]
            font_size = rng.randint(70, 110)
            img = _render_letter(letter, font_path, font_size)
            img = _augment(img, rng)
            images.append(img)
            labels.append(letter)

    print(f"Generated {len(images)} gallery images  ({n_letters} letters × {m_images} variants)")
    return images, labels


# ─── Model helpers ────────────────────────────────────────────────────────────

_to_tensor = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])


def _load_model(weights_path: str) -> EmbeddingModel:
    stride = max(1, int(window_size * stride_ratio))
    model  = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=True,
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        dropout=model_dropout,
    ).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded weights from  {weights_path}")
    return model


@torch.no_grad()
def _embed(model: EmbeddingModel, pil_images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
    """
    Returns L2-normalised mean-pooled embeddings: [N, vector_size].
    Mean-pooling over the spatial sequence gives one vector per image.
    """
    all_vecs = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        t = torch.stack([_to_tensor(img.convert("RGB")) for img in batch]).to(device)
        feat = model(t)                          # [B, seq_len, vector_size]
        vec  = feat.mean(dim=1)                  # [B, vector_size]  — mean-pool spatial dim
        vec  = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_vecs.append(vec.cpu())
    return torch.cat(all_vecs, dim=0)            # [N, vector_size]


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_vec: torch.Tensor,         # [vector_size]
    gallery_vecs: torch.Tensor,      # [N, vector_size]
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity retrieval; returns (top_k_scores, top_k_indices)."""
    sims    = (gallery_vecs @ query_vec).squeeze()   # [N]
    top_k   = min(k, len(sims))
    scores, idx = sims.topk(top_k)
    return scores, idx


# ─── Visualisation ────────────────────────────────────────────────────────────

def _show_image(ax, img: Image.Image, title: str, border_color=None):
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")
    if border_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)


def save_results(
    query_img: Image.Image,
    query_label: str,
    gallery_imgs: list[Image.Image],
    gallery_labels: list[str],
    scores: torch.Tensor,
    indices: torch.Tensor,
    output_path: str = "retrieval_results.png",
):
    k      = len(indices)
    n_cols = min(k, 5)
    n_res_rows = (k + n_cols - 1) // n_cols   # rows needed for results
    total_rows = n_res_rows + 1               # +1 for query row

    fig, axes = plt.subplots(
        total_rows, n_cols,
        figsize=(n_cols * 3.2, total_rows * 1.8),
        gridspec_kw={"hspace": 0.6, "wspace": 0.3},
    )
    # Normalise to 2-D array regardless of shape
    if total_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Top-{k} Nearest Neighbours  |  Query: '{query_label}'",
        fontsize=13, fontweight="bold",
    )

    # ── Row 0: query centred, rest hidden ─────────────────────────────────────
    for c in range(n_cols):
        axes[0, c].axis("off")
    mid = n_cols // 2
    _show_image(axes[0, mid], query_img, f"QUERY  '{query_label}'", border_color="#e74c3c")

    # ── Rows 1+: top-k results ────────────────────────────────────────────────
    for rank, (score, idx) in enumerate(zip(scores, indices)):
        r = rank // n_cols + 1
        c = rank % n_cols
        lbl = gallery_labels[idx.item()]
        _show_image(axes[r, c], gallery_imgs[idx.item()],
                    f"#{rank+1}  '{lbl}'  {score:.3f}")

    # Hide any unused cells in the last row
    for rank_unused in range(k, n_res_rows * n_cols):
        r = rank_unused // n_cols + 1
        c = rank_unused % n_cols
        axes[r, c].axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Arabic letter image retrieval demo")
    parser.add_argument("--n_letters", type=int, default=10,
                        help="How many distinct Arabic letters to sample (max 28)")
    parser.add_argument("--m_images",  type=int, default=20,
                        help="Augmented variants per letter")
    parser.add_argument("--top_k",     type=int, default=10,
                        help="Number of nearest neighbours to show")
    parser.add_argument("--query",     type=str, default="ب",
                        help="Single Arabic letter  OR  path to a PNG query image")
    parser.add_argument("--weights",   type=str,
                        default="Weights/localRun/model_latest.pth",
                        help="Path to model weights (.pth)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    type=str, default="retrieval_results.png",
                        help="Where to save the results image")
    parser.add_argument("--batch_size",type=int, default=32,
                        help="Embedding batch size (reduce if OOM)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 1. Generate gallery ───────────────────────────────────────────────────
    gallery_imgs, gallery_labels = generate_gallery(args.n_letters, args.m_images, rng)

    # ── 2. Prepare query image ────────────────────────────────────────────────
    if os.path.isfile(args.query):
        query_img   = Image.open(args.query).convert("RGB").resize((IMG_W, IMG_H))
        query_label = os.path.splitext(os.path.basename(args.query))[0]
        print(f"Query: image file  {args.query}")
    else:
        # Treat as a single letter
        available_fonts = [
            os.path.join(FONT_DIR, f) for f in FONTS
            if os.path.exists(os.path.join(FONT_DIR, f))
        ]
        query_img   = _render_letter(args.query, available_fonts[0], font_size=90)
        query_label = args.query
        print(f"Query: letter  '{args.query}'  (no augmentation)")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    model = _load_model(args.weights)

    # ── 4. Embed everything ───────────────────────────────────────────────────
    print("Embedding gallery …")
    gallery_vecs = _embed(model, gallery_imgs, batch_size=args.batch_size)   # [N, D]

    print("Embedding query …")
    query_vec    = _embed(model, [query_img], batch_size=1).squeeze(0)       # [D]

    # ── 5. Retrieve top-k ─────────────────────────────────────────────────────
    scores, indices = retrieve_top_k(query_vec, gallery_vecs, k=args.top_k)

    print(f"\n{'Rank':<6} {'Letter':<8} {'Score':>7}")
    print("─" * 24)
    for rank, (s, idx) in enumerate(zip(scores, indices), 1):
        print(f"  {rank:<4} {gallery_labels[idx.item()]:<8} {s.item():>7.4f}")

    # ── 6. Save visualisation ─────────────────────────────────────────────────
    save_results(
        query_img, query_label,
        gallery_imgs, gallery_labels,
        scores, indices,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

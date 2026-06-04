"""
sample_data.py
==============
Synthetic Arabic line-image generator for paper figures.
No dataset required — produces short, legible (image, text) pairs.

Public API
----------
make_sample(sentence_idx, transform=True) -> (tensor_or_PIL, text_str)
    Mirrors load_sample() so figure scripts swap one import for the other.

ARABIC_SENTENCES : list[str]
    Pool of 10 short sentences (10–22 chars) used as figure examples.

FIG_WINDOW_SIZE : int   = 16  (width of each display window in pixels)
FIG_STRIDE : int        = 8   (50% overlap stride for figures, matches training)
FIG_NUM_PATCHES : int   = (708-16)//8 + 1   (= 87 overlapping windows)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PROJ_ROOT)

try:
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi
    _HAS_ARABIC = True
except ImportError:
    _HAS_ARABIC = False

# ── Sentence pool ─────────────────────────────────────────────────────────────
# Short Arabic phrases (10–22 printable chars) — clearly legible in heatmaps.
ARABIC_SENTENCES = [
    "العلم نور والجهل ظلام",
    "الوقت أثمن من الذهب",
    "القراءة تنير العقل",
    "العمل يشرف صاحبه",
    "الصبر مفتاح الفرج",
    "النجاح بالجهد والإرادة",
    "الصدق طريق النجاح",
    "الكتاب خير صديق",
    "بالعلم ترتفع الأمم",
    "العزم يقهر الصعاب",
]

# ── Sentence pairs with shared content (for fig10 image-image alignment) ─────
# Each pair (A, B) shares visible Arabic word clusters so the NW alignment
# path in the image-image similarity matrix shows meaningful correspondence.
FIG10_SENTENCE_PAIRS = [
    # Pair 0 — share "العلم" and "نور العمل" but in reversed order
    ("العلم نور والعمل طريق",
     "نور العلم أساس العمل"),
    # Pair 1 — share "خير صديق" and "الكتاب"
    ("الكتاب خير صديق للعاقل",
     "خير صديق هو الكتاب"),
    # Pair 2 — share "الصبر" and "النجاح"
    ("الصبر مفتاح النجاح دائما",
     "النجاح يحتاج الصبر والعزم"),
]

# ── Non-overlapping stride for figure display ─────────────────────────────────
FIG_WINDOW_SIZE = 16                                     # display window width (px)
FIG_STRIDE      = 8                                      # 50 % overlap — matches training
_IMG_W          = 708
_IMG_H          = 128
FIG_NUM_PATCHES = (_IMG_W - FIG_WINDOW_SIZE) // FIG_STRIDE + 1  # = 127

_PIL_SIZE  = (_IMG_W, _IMG_H)     # PIL Image.new / PIL resize: (width, height)
_TV_SIZE   = (_IMG_H, _IMG_W)     # torchvision Resize: (height, width)

# ── Font discovery ─────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    os.path.join(os.path.expanduser("~"),
                 ".conda/envs/manucripts_align/fonts/DejaVuSans.ttf"),
]

def _get_font(size):
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _measure(draw, text, font):
    """Return (width, height) of rendered text."""
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1], bb[1]
    except AttributeError:
        w, h = draw.textsize(text, font=font)
        return w, h, 0


def _render_arabic(text, img_size=_PIL_SIZE, bg=(0, 0, 0), fg=(255, 255, 255)):
    """Render an Arabic string as a PIL RGB line image."""
    display_text = (
        _bidi(arabic_reshaper.reshape(text)) if _HAS_ARABIC else text
    )

    # Auto-fit: shrink font until text fits within 90 % of image width
    font_size = 72
    img  = Image.new("RGB", img_size, color=bg)
    draw = ImageDraw.Draw(img)

    while font_size >= 16:
        font = _get_font(font_size)
        tw, th, ty_off = _measure(draw, display_text, font)
        if tw <= img_size[0] * 0.90 and th <= img_size[1] * 0.85:
            break
        font_size -= 4

    # Center in image
    x = max(0, (img_size[0] - tw) // 2)
    y = max(0, (img_size[1] - th) // 2) - ty_off
    draw.text((x, y), display_text, fill=fg, font=font)
    return img


# ── Public API ────────────────────────────────────────────────────────────────

def pad_text(text):
    """Prepend and append a space token so the model sees alignment boundaries."""
    return " " + text + " "


def get_fig_windows(sentence_idx=0):
    """
    Return the FIG_NUM_PATCHES window patches aligned to heatmap columns.
    Reversed so index 0 = rightmost patch, matching use_flip=True (Arabic RTL).
    """
    pil, _ = make_sample(sentence_idx, transform=False)
    pil    = pil.resize((_IMG_W, _IMG_H))
    arr    = np.array(pil)  # [128, 1024, 3]
    patches = [arr[:, j * FIG_STRIDE : j * FIG_STRIDE + FIG_WINDOW_SIZE, :]
               for j in range(FIG_NUM_PATCHES)]
    return list(reversed(patches))   # RTL: col 0 = rightmost (first Arabic char)


def get_fig_pair_windows(pair_idx=0):
    """
    Return (windows_a, windows_b) for a FIG10_SENTENCE_PAIRS entry.
    Both lists are reversed to match use_flip=True.
    """
    _, _, _, _, pil_a, pil_b = make_fig10_pair(pair_idx)
    pil_a = pil_a.resize((_IMG_W, _IMG_H))
    pil_b = pil_b.resize((_IMG_W, _IMG_H))
    arr_a, arr_b = np.array(pil_a), np.array(pil_b)
    wins_a = list(reversed([arr_a[:, j*FIG_STRIDE:j*FIG_STRIDE+FIG_WINDOW_SIZE, :] for j in range(FIG_NUM_PATCHES)]))
    wins_b = list(reversed([arr_b[:, j*FIG_STRIDE:j*FIG_STRIDE+FIG_WINDOW_SIZE, :] for j in range(FIG_NUM_PATCHES)]))
    return wins_a, wins_b



_transform = transforms.Compose([
    transforms.Resize(_TV_SIZE),      # torchvision uses (height, width) = (128, 1024)
    transforms.ToTensor(),
])


def make_fig10_pair(pair_idx=0, transform=True):
    """
    Return (img_a, text_a, img_b, text_b, pil_a, pil_b) for a shared-content pair.
    img_* = tensor [3,H,W] if transform=True, else pil_* is the PIL image.
    pil_* always returned as resized (1024×128) PIL images for display.
    """
    text_a, text_b = FIG10_SENTENCE_PAIRS[pair_idx % len(FIG10_SENTENCE_PAIRS)]
    pil_a_raw = _render_arabic(text_a)
    pil_b_raw = _render_arabic(text_b)
    pil_a = pil_a_raw.resize((_IMG_W, _IMG_H))
    pil_b = pil_b_raw.resize((_IMG_W, _IMG_H))
    if transform:
        return _transform(pil_a_raw), text_a, _transform(pil_b_raw), text_b, pil_a, pil_b
    return pil_a, text_a, pil_b, text_b, pil_a, pil_b


def make_sample(sentence_idx=0, transform=True):
    """
    Generate a synthetic (image, text) pair from ARABIC_SENTENCES.

    Mirrors the interface of load_sample() so figure scripts can swap
    one import for the other without any other changes.

    Args:
        sentence_idx : 0-based index (wraps around len(ARABIC_SENTENCES)).
        transform    : If True  → returns float32 Tensor [3, H, W].
                       If False → returns PIL Image.

    Returns:
        (image, text_str)
    """
    text    = ARABIC_SENTENCES[sentence_idx % len(ARABIC_SENTENCES)]
    pil_img = _render_arabic(text)
    if transform:
        return _transform(pil_img), text
    return pil_img, text

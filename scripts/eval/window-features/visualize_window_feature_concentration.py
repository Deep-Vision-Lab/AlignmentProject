#!/usr/bin/env python3
"""Visualize local Arabic window evidence without confusing it with region context.

Each output reports two labels:
- local/core: the best visible core span for this individual window;
- region/core: the core span selected globally by hard Span-DTW for the region.

Grad-CAM is always targeted with the local core embedding. Context-only look-ahead
characters and spaces are never presented as if they were visible in the crop.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arabic_span_text_encoder import ArabicSpanTextEncoder  # noqa: E402
from arabic_token_text_encoder import ArabicTokenTextEncoder  # noqa: E402
from embeddingModel import EmbeddingModel  # noqa: E402
from span_alignment_loss import hard_span_dtw_path  # noqa: E402
from textEmbedding import TextEmbedding  # noqa: E402

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def display_text(text: str) -> str:
    text = "" if text is None else str(text)
    if arabic_reshaper is not None and get_display is not None:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass
    return text


def safe_token(text: str, max_len: int = 24) -> str:
    text = str(text or "empty").strip() or "empty"
    text = text.replace("<BLANK>", "BLANK").replace("<SPACE>", "SPACE")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\u0600-\u06FF]+", "", text)
    return text[:max_len] or "token"


def read_text(path: Optional[str], text: Optional[str]) -> str:
    if text is not None:
        return text
    if path is None:
        raise ValueError(
            "Provide --text/--text-path with --line, or use --data-dir + --index."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def infer_paths(args) -> Tuple[str, str]:
    if args.line:
        if args.text_path is None and args.text is None:
            raise ValueError(
                "When using --line, also provide --text or --text-path."
            )
        return args.line, args.text_path
    if args.data_dir is None or args.index is None:
        raise ValueError(
            "Provide --line + --text/--text-path, or --data-dir + --index."
        )
    image_path = os.path.join(
        args.data_dir, "images", f"img{args.which_line}_{args.index}.png"
    )
    text_path = os.path.join(
        args.data_dir, "texts", f"text{args.which_line}_{args.index}.txt"
    )
    return image_path, text_path


def load_image(path: str, height: int, width: int) -> Tuple[Image.Image, torch.Tensor]:
    pil = Image.open(path).convert("RGB").resize(
        (int(width), int(height)), Image.BILINEAR
    )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pil, transform(pil).unsqueeze(0)


def checkpoint_config(loaded) -> dict:
    return dict(loaded.get("model_config") or {}) if isinstance(loaded, dict) else {}


def image_state(loaded):
    if isinstance(loaded, dict):
        return (
            loaded.get("image_model_state_dict")
            or loaded.get("model_state_dict")
            or loaded
        )
    return loaded


def build_image_model(args, cfg: dict, device: str) -> EmbeddingModel:
    cfg_flip = bool(cfg.get("use_flip", str(cfg.get("lang", "")).lower() == "arabic"))
    requested_flip = cfg_flip if args.use_flip is None else bool(args.use_flip)
    model = EmbeddingModel(
        window_size=int(args.window_size or cfg.get("window_size", 32)),
        stride=int(args.stride or cfg.get("stride", 16)),
        vector_size=int(args.vector_size or cfg.get("vector_size", 128)),
        device=device,
        use_flip=requested_flip,
        use_bilstm=bool(cfg.get("use_bilstm", True)) and not args.no_bilstm,
        bilstm_layers=int(cfg.get("bilstm_layers", 1)),
        bilstm_hidden_dim=int(
            cfg.get(
                "bilstm_hidden_dim",
                args.vector_size or cfg.get("vector_size", 128),
            )
        ),
        use_local_grouping=bool(cfg.get("use_local_window_grouping", True)),
        local_group_size=int(cfg.get("local_group_size", 3)),
    ).to(device)
    return model


def build_text_encoder(args, cfg: dict, loaded, device: str):
    encoder_type = str(
        args.text_encoder_type or cfg.get("text_encoder_type", "arabic_span")
    ).lower()
    dim = int(args.vector_size or cfg.get("vector_size", 128))
    model_name = args.arabic_text_model_name or cfg.get(
        "arabic_text_model_name", "aubmindlab/bert-base-arabertv02"
    )
    if encoder_type == "arabic_span":
        encoder = ArabicSpanTextEncoder(
            model_name=model_name,
            output_dim=dim,
            max_span_chars=int(
                args.max_span_chars or cfg.get("max_text_span_chars", 3)
            ),
            freeze_backbone=True,
            device=device,
            strip_text_edges=bool(cfg.get("strip_span_text_edges", True)),
            cache_size=int(cfg.get("span_feature_cache_size", 2048)),
            cache_dtype=str(cfg.get("span_feature_cache_dtype", "float16")),
            boundary_context_chars=_env_int("SPAN_BOUNDARY_CONTEXT_CHARS", 1),
            include_space_context=_env_flag("SPAN_INCLUDE_SPACE_CONTEXT", False),
            boundary_context_max_core_chars=_env_int(
                "SPAN_BOUNDARY_CONTEXT_MAX_CORE_CHARS", 1
            ),
            allow_character_space_surfaces=_env_flag(
                "SPAN_ALLOW_CHARACTER_SPACE_SURFACES", False
            ),
        )
    elif encoder_type == "arabic_token":
        encoder = ArabicTokenTextEncoder(
            model_name=model_name,
            output_dim=dim,
            max_token_chars=int(
                args.max_token_chars or cfg.get("max_text_token_chars", 3)
            ),
            freeze_backbone=True,
            device=device,
        )
    elif encoder_type == "char":
        encoder = TextEmbedding(embedding_dim=dim).to(device)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError(f"Unknown text encoder type: {encoder_type}")

    if isinstance(loaded, dict):
        state = loaded.get("text_encoder_state_dict") or loaded.get(
            "text_embedder_state_dict"
        )
        if state:
            encoder.load_state_dict(state, strict=False)

    # Evaluation defaults are intentionally strict: no implicit character+space
    # s²È="24€€€¤(€€€€€€€€¤(€€€€€€€•¹Ñ•È€ô€À¸Ô€¨€¡àÀ€¬àÄ¤(€€€€€€€…á¥Ì¹Ñ•áÐ (€€€€€€€€€€€•¹Ñ•È°(€€€€€€€€€€€¡•¥¡Ð€¬€Ü°(€€€€€€€€€€€‘¥ÍÁ±…å}Ñ•áÐ¡±½…±}±…‰•°¤°(€€€€€€€€€€€¡„ô‰•¹Ñ•Èˆ°(€€€€€€€€€€€Ù„ô‰Ñ½Àˆ°(€€€€€€€€€€€™½¹ÑÍ¥é”õ…ÉÌ¹Ñ½­•¹}™½¹ÑÍ¥é”°(€€€€€€€€€€€É½Ñ…Ñ¥½¸ôäÀ¥˜…ÉÌ¹É½Ñ…Ñ•}Ñ½­•¹Ì•±Í”€À°(€€€€€€€€¤(€€€€€€€…á¥Ì¹Ñ•áÐ (€€€€€€€€€€€•¹Ñ•È°(€€€€€€€€€€€¡•¥¡Ð€¬€ÌÀ°(€€€€€€€€€€€‘¥ÍÁ±…å}Ñ•áÐ¡É•¥½¹}±…‰•±Ím¥¹‘•át¤°(€€€€€€€€€€€¡„ô‰•¹Ñ•Èˆ°(€€€€€€€€€€€Ù„ô‰Ñ½Àˆ°(€€€€€€€€€€€™½¹ÑÍ¥é”õµ…à Ð¸À°…ÉÌ¹Ñ½­•¹}™½¹ÑÍ¥é”€´€Ä¸À¤°(€€€€€€€€€€€É½Ñ…Ñ¥½¸ôäÀ¥˜…ÉÌ¹É½Ñ…Ñ•}Ñ½­•¹Ì•±Í”€À°(€€€€€€€€€€€…±Á¡„ôÀ¸Ô°(€€€€€€€€¤(€€€€€€€¥˜™±½…Ð¡¥¹­}É…Ñ¥½m¥¹‘•át¤€ð™±½…Ð¡…ÉÌ¹‰±…¹­}¥¹­}Ñ¡É•Í¡½±¤è(€€€€€€€€€€€…á¥Ì¹…‘‘}Á…Ñ  (€€€€€€€€€€€€€€€Á±Ð¹I•Ñ…¹±” (€€€€€€€€€€€€€€€€€€€€¡àÀ°€À¤°(€€€€€€€€€€€€€€€€€€€µ…à Ä°àÄ€´àÀ¤°(€€€€€€€€€€€€€€€€€€€¡•¥¡Ð°(€€€€€€€€€€€€€€€€€€€™¥±°õQÉÕ”°(€€€€€€€€€€€€€€€€€€€…±Á¡„ôÀ¸Àà°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€™¥ÕÉ”¹Í…Ù•™¥œ¡½ÕÑ}Á…Ñ °‘Á¤õ…ÉÌ¹‘Á¤°‰‰½á}¥¹¡•Ìô‰Ñ¥¡Ðˆ¤(€€€Á±Ð¹±½Í”¡™¥ÕÉ”¤(()‘•˜Á…ÉÍ•}…ÉÌ ¤è(€€€Á…ÉÍ•È€ô…ÉÁ…ÉÍ”¹ÉÕµ•¹ÑA…ÉÍ•È (€€€€€€€‘•ÍÉ¥ÁÑ¥½¸ô‰É•…Ñ”ÑÉÕÑ¡™Õ°Á•ÈµÝ¥¹‘½ÜÉ…µ4…¹½É”µÍÁ…¸±…‰•±Ì¸ˆ(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ•¥¡ÑÌˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ±¥¹”ˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÑ•áÐˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÑ•áÐµÁ…Ñ ˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ‘…Ñ„µ‘¥Èˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¥¹‘•àˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ¡¥ µ±¥¹”ˆ°ÑåÁ”õ¥¹Ð°¡½¥•ÌõlÄ°€Ét°‘•™…Õ±ÐôÄ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½ÕÑÁÕÐˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¹¼µÍÕµµ…Éäµ¥µ…”ˆ°…Ñ¥½¸ô‰ÍÑ½É•}ÑÉÕ”ˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍ…Ù”µÁ•ÈµÝ¥¹‘½Üµ¥µ…•Ìˆ°…Ñ¥½¸ô‰ÍÑ½É•}ÑÉÕ”ˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ•ÈµÝ¥¹‘½Üµ½ÕÑÁÕÐµ‘¥Èˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ•…É±äµ™ÕÍ¥½¸µ…±Á¡„ˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÀ¸ÔÔ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ•ÈµÝ¥¹‘½Üµ¥µ…”µÝ¥‘Ñ ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÈÔØ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ•ÈµÝ¥¹‘½Üµ¥µ…”µ¡•¥¡Ðˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÈÔØ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ•ÈµÝ¥¹‘½ÜµÍ…±”ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðôà¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ•ÈµÝ¥¹‘½Üµ‘Á¤ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÄØÀ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð (€€€€€€€€ˆ´µ‘•Ù¥”ˆ°‘•™…Õ±Ðô‰Õ‘„ˆ¥˜Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤•±Í”€‰ÁÔˆ(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¡•¥¡Ðˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÄÈà¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ¥‘Ñ ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÄÀÈÐ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ¥¹‘½ÜµÍ¥é”ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍÑÉ¥‘”ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÙ•Ñ½ÈµÍ¥é”ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð (€€€€€€€€ˆ´µÕÍ”µ™±¥Àˆ°…Ñ¥½¸õ…ÉÁ…ÉÍ”¹	½½±•…¹=ÁÑ¥½¹…±Ñ¥½¸°‘•™…Õ±Ðõ9½¹”(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¹¼µ‰¥±ÍÑ´ˆ°…Ñ¥½¸ô‰ÍÑ½É•}ÑÉÕ”ˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð (€€€€€€€€ˆ´µ™•…ÑÕÉ”µÍÁ…”ˆ°(€€€€€€€¡½¥•Ìõl‰±½…°ˆ°€‰É½ÕÁ•ˆ°€‰½¹Ñ•áÑÕ…°‰t°(€€€€€€€‘•™…Õ±Ðô‰±½…°ˆ°(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð (€€€€€€€€ˆ´µ…±¥¹µ•¹ÐµÍÁ…”ˆ°(€€€€€€€¡½¥•Ìõl‰±½…°ˆ°€‰É½ÕÁ•ˆ°€‰½¹Ñ•áÑÕ…°‰t°(€€€€€€€‘•™…Õ±Ðô‰½¹Ñ•áÑÕ…°ˆ°(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð (€€€€€€€€ˆ´µÑ•áÐµ•¹½‘•ÈµÑåÁ”ˆ°(€€€€€€€¡½¥•Ìõl‰…É…‰¥}ÍÁ…¸ˆ°€‰…É…‰¥}Ñ½­•¸ˆ°€‰¡…È‰t°(€€€€€€€‘•™…Õ±Ðõ9½¹”°(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ…É…‰¥ŒµÑ•áÐµµ½‘•°µ¹…µ”ˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ…àµÍÁ…¸µ¡…ÉÌˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ…àµÑ½­•¸µ¡…ÉÌˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ…àµÝ¥¹‘½ÝÌµÁ•ÈµÍÁ…¸ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÐ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÑ•µÁ•É…ÑÕÉ”ˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÀ¸ÀÜ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ¥¹‘½Üµ½Õ¹ÐµÁ•¹…±Ñäˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÀ¸ÀÔ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ±½…°µ±…‰•°µµ…àµ¡…ÉÌˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÈ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ±½…°µ±…‰•°µ±•¹Ñ µÁ•¹…±Ñäˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÀ¸ÀÔ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ‰±…¹¬µ¥¹¬µÑ¡É•Í¡½±ˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôÀ¸ÀÀÔ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÉ…‘…´µ±…å•Èˆ°‘•™…Õ±Ðô‰¹¹}•¹½‘•È¹‰…­‰½¹”¸Üˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð (€€€€€€€€ˆ´µÉ…‘…´µÑ…É•Ðˆ°(€€€€€€€¡½¥•Ìõl‰Ñ½­•¸ˆ°€‰•¹•Éäˆ°€‰Ñ½Á}™•…ÑÕÉ”‰t°(€€€€€€€‘•™…Õ±Ðô‰Ñ½­•¸ˆ°(€€€€¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ…Àˆ°‘•™…Õ±Ðô‰Ù¥É¥‘¥Ìˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÑ½­•¸µ™½¹ÑÍ¥é”ˆ°ÑåÁ”õ™±½…Ð°‘•™…Õ±ÐôØ¸À¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÉ½Ñ…Ñ”µÑ½­•¹Ìˆ°…Ñ¥½¸ô‰ÍÑ½É•}ÑÉÕ”ˆ°‘•™…Õ±ÐõQÉÕ”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ‘Á¤ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÄàÀ¤(€€€€Œ½µÁ…Ñ¥‰¥±¥Ñä…ÉÕµ•¹ÑÌÉ•Ñ…¥¹•™½È•á¥ÍÑ¥¹œ±…Õ¹¡•ÉÌ¸(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍØµ½ÕÑÁÕÐˆ°‘•™…Õ±Ðõ9½¹”¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¹¼µÍØˆ°…Ñ¥½¸ô‰ÍÑ½É•}ÑÉÕ”ˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½¹•¹ÑÉ…Ñ¥½¸µµ•ÑÉ¥Œˆ°‘•™…Õ±Ðô‰Ñ½Á­}µ…ÍÌˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½¹•¹ÑÉ…Ñ¥½¸µÑ½Àµ¬ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðôà¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µ™•…ÑÕÉ”µ¹½Éµ…±¥é”ˆ°‘•™…Õ±Ðô‰Á•É}Ý¥¹‘½Üˆ¤(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍØµÑ½Àµ™•…ÑÕÉ•Ìˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±Ðôà¤(€€€É•ÑÕÉ¸Á…ÉÍ•È¹Á…ÉÍ•}…ÉÌ ¤(()‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ôÁ…ÉÍ•}…ÉÌ ¤(€€€¥µ…•}Á…Ñ °Ñ•áÑ}Á…Ñ €ô¥¹™•É}Á…Ñ¡Ì¡…ÉÌ¤(€€€Ñ•áÐ€ôÉ•…‘}Ñ•áÐ¡Ñ•áÑ}Á…Ñ °…ÉÌ¹Ñ•áÐ¤(€€€±½…‘•€ôÑ½É ¹±½…¡…ÉÌ¹Ý•¥¡ÑÌ°µ…Á}±½…Ñ¥½¸õ…ÉÌ¹‘•Ù¥”¤(€€€½¹™¥œ€ô¡•­Á½¥¹Ñ}½¹™¥œ¡±½…‘•¤(€€€…ÉÌ¹Ý¥¹‘½Ý}Í¥é”€ô¥¹Ð¡…ÉÌ¹Ý¥¹‘½Ý}Í¥é”½È½¹™¥œ¹•Ð ‰Ý¥¹‘½Ý}Í¥é”ˆ°€ÌÈ¤¤(€€€…ÉÌ¹ÍÑÉ¥‘”€ô¥¹Ð (€€€€€€€…ÉÌ¹ÍÑÉ¥‘”½È½¹™¥œ¹•Ð ‰ÍÑÉ¥‘”ˆ°µ…à Ä°…ÉÌ¹Ý¥¹‘½Ý}Í¥é”€¼¼€È¤¤(€€€€¤(€€€…ÉÌ¹Ù•Ñ½É}Í¥é”€ô¥¹Ð¡…ÉÌ¹Ù•Ñ½É}Í¥é”½È½¹™¥œ¹•Ð ‰Ù•Ñ½É}Í¥é”ˆ°€ÄÈà¤¤(€€€…ÉÌ¹µ…á}ÍÁ…¹}¡…ÉÌ€ô¥¹Ð (€€€€€€€…ÉÌ¹µ…á}ÍÁ…¹}¡…ÉÌ½È½¹™¥œ¹•Ð ‰µ…á}Ñ•áÑ}ÍÁ…¹}¡…ÉÌˆ°€Ì¤(€€€€¤(€€€…ÉÌ¹µ…á}Ñ½­•¹}¡…ÉÌ€ô¥¹Ð (€€€€€€€…ÉÌ¹µ…á}Ñ½­•¹}¡…ÉÌ½È½¹™¥œ¹•Ð ‰µ…á}Ñ•áÑ}Ñ½­•¹}¡…ÉÌˆ°€Ì¤(€€€€¤(€€€¥˜€‰½¹ÑÉ…ÍÑ¥Ù•}Ñ•µÁ•É…ÑÕÉ”ˆ¥¸½¹™¥œ…¹…ÉÌ¹Ñ•µÁ•É…ÑÕÉ”€ôô€À¸ÀÜè(€€€€€€€…ÉÌ¹Ñ•µÁ•É…ÑÕÉ”€ô™±½…Ð¡½¹™¥œ¹•Ð ‰½¹ÑÉ…ÍÑ¥Ù•}Ñ•µÁ•É…ÑÕÉ”ˆ°€À¸ÀÜ¤¤((€€€µ½‘•°€ô‰Õ¥±‘}¥µ…•}µ½‘•°¡…ÉÌ°½¹™¥œ°…ÉÌ¹‘•Ù¥”¤(€€€µ½‘•°¹±½…‘}ÍÑ…Ñ•}‘¥Ð¡¥µ…•}ÍÑ…Ñ”¡±½…‘•¤°ÍÑÉ¥Ðõ…±Í”¤(€€€µ½‘•°¹•Ù…° ¤(€€€…ÉÌ¹É•Í½±Ù•‘}ÕÍ•}™±¥À€ô‰½½°¡µ½‘•°¹ÕÍ•}™±¥À¤(€€€Ñ•áÑ}•¹½‘•È€ô‰Õ¥±‘}Ñ•áÑ}•¹½‘•È¡…ÉÌ°½¹™¥œ°±½…‘•°…ÉÌ¹‘•Ù¥”¤((€€€Á¥±}¥µ…”°¥µ…•}Ñ•¹Í½È€ô±½…‘}¥µ…” (€€€€€€€¥µ…•}Á…Ñ °…ÉÌ¹¡•¥¡Ð°…ÉÌ¹Ý¥‘Ñ (€€€€¤(€€€½¹Ñ•áÑÕ…°°±½…°°É½ÕÁ•°¥¹­}É…Ñ¥¼€ô•Ñ}±¥¹•}•µ‰•‘‘¥¹Ì (€€€€€€€µ½‘•°°¥µ…•}Ñ•¹Í½È°…ÉÌ¹‘•Ù¥”(€€€€¤(€€€™•…ÑÕÉ•}ÍÁ…•Ì€ôì(€€€€€€€€‰±½…°ˆè±½…°°(€€€€€€€€‰É½ÕÁ•ˆèÉ½ÕÁ•°(€€€€€€€€‰½¹Ñ•áÑÕ…°ˆè½¹Ñ•áÑÕ…°°(€€€ô(€€€…±¥¹µ•¹Ñ}™•…ÑÕÉ•Ì€ô™•…ÑÕÉ•}ÍÁ…•Ím…ÉÌ¹…±¥¹µ•¹Ñ}ÍÁ…•t(€€€±½…±}™•…ÑÕÉ•Ì€ô™•…ÑÕÉ•}ÍÁ…•Ím…ÉÌ¹™•…ÑÕÉ•}ÍÁ…•t(€€€É•¥½¹}±…‰•±Ì°}É•¥½¹}¥¹‘¥•Ì°Á…Ñ °•¹½‘¥¹œ€ô…±¥¹•‘}É•¥½¹Ì (€€€€€€€Ñ•áÑ}•¹½‘•È°Ñ•áÐ°…±¥¹µ•¹Ñ}™•…ÑÕÉ•Ì°…ÉÌ(€€€€¤(€€€±½…±}±…‰•±Ì°±½…±}¥¹‘¥•Ì°±½…±}Í½É•Ì€ô±½…±}½É•}±…‰•±Ì (€€€€€€€•¹½‘¥¹œ°(€€€€€€€Á…Ñ °(€€€€€€€±½…±}™•…ÑÕÉ•Ì°(€€€€€€€¥¹­}É…Ñ¥¼°(€€€€€€€…ÉÌ°(€€€€¤(€€€Á…Ñ¡•Ì€ôµ½‘•±}Á…Ñ¡•Ì¡¥µ…•}Ñ•¹Í½È°…ÉÌ°…ÉÌ¹‘•Ù¥”¤((€€€¥˜¹½Ð…ÉÌ¹¹½}ÍÕµµ…Éå}¥µ…”è(€€€€€€€Í…Ù•}ÍÕµµ…Éä (€€€€€€€€€€€Á¥±}¥µ…”°(€€€€€€€€€€€±½…±}±…‰•±Ì°(€€€€€€€€€€€É•¥½¹}±…‰•±Ì°(€€€€€€€€€€€¥¹­}É…Ñ¥¼°(€€€€€€€€€€€…ÉÌ°(€€€€€€€€€€€…ÉÌ¹½ÕÑÁÕÐ°(€€€€€€€€¤(€€€€€€€ÁÉ¥¹Ð¡˜‰Í…Ù•ÍÕµµ…Éä™¥ÕÉ”èí…ÉÌ¹½ÕÑÁÕÑôˆ¤((€€€¥˜…ÉÌ¹Í…Ù•}Á•É}Ý¥¹‘½Ý}¥µ…•Ìè(€€€€€€€½ÕÑÁÕÑ}‘¥È€ô…ÉÌ¹Á•É}Ý¥¹‘½Ý}½ÕÑÁÕÑ}‘¥È½È€ (€€€€€€€€€€€ÍÑÈ¡A…Ñ ¡…ÉÌ¹½ÕÑÁÕÐ¤¹Ý¥Ñ¡}ÍÕ™™¥à ˆˆ¤¤€¬€‰}É…‘…µ}Ý¥¹‘½ÝÌˆ(€€€€€€€€¤(€€€€€€€Í…Ù•}Á•É}Ý¥¹‘½Ý}É…‘…µÌ (€€€€€€€€€€€Á¥±}¥µ…”°(€€€€€€€€€€€±½…±}±…‰•±Ì°(€€€€€€€€€€€±½…±}¥¹‘¥•Ì°(€€€€€€€€€€€±½…±}Í½É•Ì°(€€€€€€€€€€€É•¥½¹}±…‰•±Ì°(€€€€€€€€€€€•¹½‘¥¹œ°(€€€€€€€€€€€Á…Ñ¡•Ì°(€€€€€€€€€€€µ½‘•°°(€€€€€€€€€€€¥¹­}É…Ñ¥¼°(€€€€€€€€€€€…ÉÌ°(€€€€€€€€€€€½ÕÑÁÕÑ}‘¥È°(€€€€€€€€¤(€€€€€€€ÁÉ¥¹Ð¡˜‰Í…Ù•Á•ÈµÝ¥¹‘½ÜÉ…µ4¥µ…•Ìèí½ÕÑÁÕÑ}‘¥Éôˆ¤(()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(
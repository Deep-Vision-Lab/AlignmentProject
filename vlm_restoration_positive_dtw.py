"""Arabic window encoder with ResNet-18 local features, ViT-Tiny context and DTW.

Pipeline:
  crop outer margins -> proportional resize -> overlapping real RGB windows ->
  shared ResNet-18 window encoder -> ViT-Tiny sequence context ->
  local/context fusion -> positive letter-DTW.

There is intentionally no restoration decoder and no reconstruction loss on
this revision. The frozen character codebook supervises only the fused visual
representation during training; final evaluation remains image-only.
"""
from __future__ import annotations

from types import MethodType
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn import all_reduce as differentiable_all_reduce

from restoration_recommended_components import (
    LocalContextFusion,
    contrastive_margin_from_costs,
    line_padding_masks,
)

DEFAULT_ARABIC_LETTERS = "ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويىة"


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def apply_branch_config(P):
    """Configure ResNet-18 local windows + canonical ViT-Tiny context + DTW."""
    P.experiment_name = "resnet18_tinyvit_positive_dtw"

    P.epochs = _env_int("RESTORATION_EPOCHS", int(P.epochs))
    P.finetune_epochs = P.epochs
    P.learning_rate = _env_float(
        "RESTORATION_LEARNING_RATE", float(P.learning_rate)
    )
    P.finetune_learning_rate = P.learning_rate
    P.num_samples = _env_int("RESTORATION_NUM_SAMPLES", int(P.num_samples))

    # Restoration pretraining no longer exists.
    requested_stage = os.environ.get(
        "RESTORATION_TRAINING_STAGE", "align"
    ).strip().lower()
    if requested_stage not in {"", "align"}:
        raise ValueError(
            "The restoration decoder was removed. RESTORATION_TRAINING_STAGE "
            "must be 'align' (Stage-A pretraining no longer exists)."
        )
    P.restoration_training_stage = "align"

    # Keep the current line geometry. Synthetic training stays on the original
    # RGB appearance. Real-data fine-tuning can opt into a synthetic-style
    # binary appearance (white handwriting on black background) while keeping
    # the same foreground crop + aspect-preserving 1024x128 geometry.
    synthetic_style_real = _env_flag("REAL_SYNTHETIC_STYLE", False)
    P.window_size = 32
    P.stride_ratio = _env_float("RESTORATION_DTW_STRIDE_RATIO", 0.50)
    P.vit_input_height = 128
    P.vit_binarize_input = False
    P.real_binarize = bool(synthetic_style_real)
    P.real_binarize_autocontrast = bool(synthetic_style_real)
    P.real_augment = False
    P.real_train_samples_per_epoch = _env_int(
        "REAL_TRAIN_SAMPLES_PER_EPOCH",
        int(getattr(P, "real_train_samples_per_epoch", 0)),
    )
    P.zero_shot_preprocess = True
    P.zero_shot_preserve_aspect = True
    P.zero_shot_foreground_crop = True
    P.zero_shot_source_geometry = False
    os.environ["SYNTHETIC_BINARIZE"] = "0"
    os.environ["REAL_BINARIZE"] = "1" if synthetic_style_real else "0"
    os.environ["REAL_BINARIZE_AUTO_INVERT"] = "1"
    os.environ["REAL_BINARIZE_AUTOCONTRAST"] = "1" if synthetic_style_real else "0"
    # Respect explicit launcher choices.  In particular, the XML bbox pipeline
    # already performs deterministic four-sided text cropping and therefore
    # disables the old heuristic foreground crop.
    os.environ.setdefault("ZERO_SHOT_FOREGROUND_CROP", "1")
    os.environ.setdefault("ZERO_SHOT_PRESERVE_ASPECT", "1")
    os.environ.setdefault("ZERO_SHOT_SOURCE_GEOMETRY", "0")
    os.environ["LINE_GEOMETRY_MODE"] = (
        "xml-bbox-gray-aspect-preserving"
        if _env_flag("VISUAL_GRAYSCALE", False)
        else "crop-aspect-preserving-rgb"
    )

    # Canonical ViT-Tiny dimensions. ResNet-18 creates one token per physical
    # window; the transformer contextualizes that window-token sequence.
    P.vector_size = 192
    P.visual_input_channels = _env_int("VISUAL_INPUT_CHANNELS", 1)
    if P.visual_input_channels not in {1, 3}:
        raise ValueError("VISUAL_INPUT_CHANNELS must be 1 or 3")
    P.visual_grayscale = bool(P.visual_input_channels == 1)
    P.vit_layers = 12
    P.vit_heads = 3
    P.vit_mlp_dim = 768
    P.vit_dropout = _env_float("TINY_VIT_DROPOUT", 0.0)
    P.vit_max_tokens = max(int(getattr(P, "vit_max_tokens", 256)), 256)
    P.restoration_context_layers = P.vit_layers
    P.restoration_fusion = "concat_projection_norm"
    P.restoration_local_encoder = "resnet18"
    P.resnet18_pretrained = _env_flag("RESNET18_PRETRAINED", True)
    P.tiny_vit_pretrained = _env_flag("TINY_VIT_PRETRAINED", True)
    P.tiny_vit_pretrained_model = os.environ.get(
        "TINY_VIT_PRETRAINED_MODEL", "facebook/deit-tiny-patch16-224"
    ).strip()
    P.pretrained_local_only = _env_flag("PRETRAINED_LOCAL_ONLY", True)
    P.restoration_semantic_adapter = "identity"
    P.span_dtw_backend = "torch"

    # Frozen character identity codebook.
    P.text_encoder_type = "char"
    P.max_text_span_chars = 1
    P.max_text_token_chars = 1
    P.letter_codebook_seed = _env_int("LETTER_CODEBOOK_SEED", 1234)
    P.letter_codebook_vocab_size = _env_int("LETTER_CODEBOOK_VOCAB_SIZE", 4096)
    P.letter_inventory = "unicode-arabic-letters-after-nfkc"

    P.keep_paired_lines_for_independent_training = True
    P.image_text_loss_on_both_lines = True

    # Current objective: positive transcript DTW only.
    # Negative transcript DTW is optional and, when explicitly enabled, is
    # limited to at most four negatives per positive sample.
    requested_negatives = max(0, _env_int("RESTORATION_NUM_NEGATIVES", 0))
    P.num_negatives = min(requested_negatives, 4)
    P.restoration_contrastive_weight = _env_float(
        "RESTORATION_CONTRASTIVE_WEIGHT", 0.0
    )
    P.restoration_contrastive_margin = _env_float(
        "RESTORATION_CONTRASTIVE_MARGIN", 0.20
    )
    P.use_local_hard_negatives = False
    P.local_hard_negative_weight = 0.0
    P.use_image_pair_contrastive = False
    P.image_pair_loss_weight = 0.0
    P.sequence_consistency_loss_weight = 0.0
    P.image_variance_loss_weight = 0.0
    P.real_filter_infeasible_span_dtw = False

    P.positive_letter_dtw_gamma_start = _env_float(
        "POSITIVE_LETTER_DTW_GAMMA_START", 0.50
    )
    P.positive_letter_dtw_gamma_end = _env_float(
        "POSITIVE_LETTER_DTW_GAMMA_END", 0.05
    )
    P.positive_letter_dtw_anneal_epochs = _env_int(
        "POSITIVE_LETTER_DTW_ANNEAL_EPOCHS", 10
    )
    P.positive_letter_dtw_gamma = P.positive_letter_dtw_gamma_start
    P.positive_letter_dtw_vertical_penalty = _env_float(
        "POSITIVE_LETTER_DTW_VERTICAL_PENALTY", 0.05
    )
    P.positive_letter_dtw_horizontal_penalty = _env_float(
        "POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY", 0.30
    )
    P.positive_letter_dtw_step_penalty = P.positive_letter_dtw_vertical_penalty
    P.positive_letter_dtw_position_prior = _env_float(
        "POSITIVE_LETTER_DTW_POSITION_PRIOR", 0.15
    )
    P.positive_letter_dtw_disable_horizontal_when_feasible = _env_flag(
        "POSITIVE_LETTER_DTW_DISABLE_HORIZONTAL_WHEN_FEASIBLE", True
    )
    P.positive_letter_dtw_competition_temperature = _env_float(
        "POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE", 0.10
    )
    P.positive_letter_dtw_cost_mode = os.environ.get(
        "POSITIVE_LETTER_DTW_COST_MODE", "full_alphabet_nll"
    ).strip().lower()
    if P.positive_letter_dtw_cost_mode not in {"cosine", "full_alphabet_nll"}:
        raise ValueError(
            "POSITIVE_LETTER_DTW_COST_MODE must be cosine or full_alphabet_nll"
        )
    P.positive_letter_dtw_min_ink = 0.0
    P.positive_letter_dtw_weight = _env_float(
        "POSITIVE_LETTER_DTW_WEIGHT", 1.0
    )

    # Strong SIGReg (LeJEPA-style ECF matching) regularizes the fused visual
    # representation before its final L2 normalization. Keep it opt-in at the
    # branch level; the real-data launcher enables it for the controlled
    # continuation run.
    P.sigreg_weight = _env_float("SIGREG_LAMBDA", 0.0)
    P.sigreg_sketch_dim = max(1, _env_int("SIGREG_SKETCH_DIM", 1024))
    P.sigreg_num_knots = max(3, _env_int("SIGREG_NUM_KNOTS", 17))
    P.sigreg_t_min = _env_float("SIGREG_T_MIN", 0.0)
    P.sigreg_t_max = _env_float("SIGREG_T_MAX", 3.0)
    P.sigreg_min_samples = max(2, _env_int("SIGREG_MIN_SAMPLES", 32))
    P.sigreg_slice_chunk = max(1, _env_int("SIGREG_SLICE_CHUNK", 128))

    # Legacy compatibility attribute: reconstruction is intentionally disabled.
    P.restoration_weight = 0.0
    # Diagnostic mode: one backward/optimizer update per batch; no accumulation.
    P.gradient_accumulation_steps = 1
    P.use_wandb = False

def _is_arabic_letter(character: str) -> bool:
    codepoint = ord(character)
    in_arabic_block = (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )
    return in_arabic_block and unicodedata.category(character).startswith("L")


def _clean_letters(text: str) -> list[str]:
    """Keep Arabic letters while removing spaces, marks, punctuation and tatweel.

    NFKC folds Arabic presentation forms/ligatures into their ordinary Unicode
    letter sequence. This avoids silently dropping valid manuscript/Quranic
    letters such as alef-wasla simply because they were absent from a hand-made
    alphabet list.
    """
    letters = []
    for character in unicodedata.normalize("NFKC", str(text)):
        if character.isspace() or character == "ـ":
            continue
        if _is_arabic_letter(character):
            letters.append(character)
    return letters


def _softmin(values: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = max(float(gamma), 1e-5)
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def _soft_dtw_cost_matrix(
    costs: torch.Tensor,
    *,
    gamma: float,
    vertical_penalty: float,
    horizontal_penalty: float,
    position_prior_weight: float = 0.0,
    disable_horizontal_when_feasible: bool = False,
) -> torch.Tensor:
    """Differentiable monotonic DTW over a precomputed [T,L] cost matrix.

    A vertical move advances to the next image window while staying on the same
    letter. A horizontal move advances the transcript without advancing the
    image window, so it is deliberately much more expensive by default.
    """
    if costs.ndim != 2:
        raise ValueError("DTW cost matrix must be [T,L]")
    T, L = int(costs.shape[0]), int(costs.shape[1])
    if T <= 0 or L <= 0:
        return costs.sum() * 0.0

    if float(position_prior_weight) > 0.0 and T > 1 and L > 1:
        image_position = torch.linspace(
            0.0, 1.0, T, device=costs.device, dtype=costs.dtype
        )
        text_position = torch.linspace(
            0.0, 1.0, L, device=costs.device, dtype=costs.dtype
        )
        costs = costs + float(position_prior_weight) * (
            image_position[:, None] - text_position[None, :]
        ).abs()

    large_value = 1e4
    gamma = max(float(gamma), 1e-5)
    v_penalty = float(vertical_penalty)
    h_penalty = float(horizontal_penalty)
    # When there are at least as many image windows as transcript letters,
    # horizontal moves are unnecessary: diagonal + vertical transitions can
    # reach the endpoint while assigning >=1 window to every letter. Forbid
    # horizontal transitions in that case so one image window cannot absorb an
    # arbitrary run of transcript characters.
    if bool(disable_horizontal_when_feasible) and T >= L:
        h_penalty = large_value
    previous = None
    previous_start = 0
    previous2 = None
    previous2_start = 0

    for diagonal_index in range(T + L - 1):
        start_i = max(0, diagonal_index - (L - 1))
        end_i = min(T - 1, diagonal_index)
        i = torch.arange(start_i, end_i + 1, device=costs.device)
        j = diagonal_index - i
        count = int(i.numel())

        vertical_valid = i > 0
        horizontal_valid = j > 0
        diagonal_valid = vertical_valid & horizontal_valid

        vertical = costs.new_full((count,), large_value)
        horizontal = costs.new_full((count,), large_value)
        diagonal = costs.new_full((count,), large_value)

        if previous is not None:
            vertical_index = (i - 1 - previous_start).clamp(
                0, max(0, int(previous.numel()) - 1)
            )
            horizontal_index = (i - previous_start).clamp(
                0, max(0, int(previous.numel()) - 1)
            )
            vertical_values = previous.index_select(0, vertical_index)
            horizontal_values = previous.index_select(0, horizontal_index)
            vertical = torch.where(vertical_valid, vertical_values, vertical)
            horizontal = torch.where(horizontal_valid, horizontal_values, horizontal)

        if previous2 is not None:
            diagonal_index_in_previous = (i - 1 - previous2_start).clamp(
                0, max(0, int(previous2.numel()) - 1)
            )
            diagonal_values = previous2.index_select(
                0, diagonal_index_in_previous
            )
            diagonal = torch.where(diagonal_valid, diagonal_values, diagonal)

        origin = (i == 0) & (j == 0)
        diagonal = torch.where(origin, torch.zeros_like(diagonal), diagonal)

        predecessors = torch.stack(
            [
                diagonal,
                vertical + v_penalty,
                horizontal + h_penalty,
            ],
            dim=0,
        )
        current = costs[i, j] - gamma * torch.logsumexp(
            -predecessors / gamma,
            dim=0,
        )
        previous2, previous2_start = previous, previous_start
        previous, previous_start = current, start_i

    return previous[0] / float(max(1, T + L))


def positive_monotonic_letter_dtw_cost(
    visual_tokens: torch.Tensor,
    target_prototypes: torch.Tensor,
    *,
    gamma: float,
    step_penalty: float,
    horizontal_penalty: float | None = None,
    position_prior_weight: float = 0.0,
) -> torch.Tensor:
    """Positive differentiable DTW from cosine costs.

    This public helper keeps the historical API for tests/ablations while using
    the new asymmetric transition penalties internally.
    """
    if visual_tokens.ndim != 2 or target_prototypes.ndim != 2:
        raise ValueError("DTW expects [T,D] visual and [L,D] letter tensors")
    if visual_tokens.shape[0] <= 0 or target_prototypes.shape[0] <= 0:
        return visual_tokens.sum() * 0.0

    visual = F.normalize(visual_tokens.float(), p=2, dim=-1)
    target = F.normalize(target_prototypes.float(), p=2, dim=-1)
    costs = 1.0 - torch.matmul(visual, target.T)
    return _soft_dtw_cost_matrix(
        costs,
        gamma=gamma,
        vertical_penalty=float(step_penalty),
        horizontal_penalty=(
            float(step_penalty)
            if horizontal_penalty is None
            else float(horizontal_penalty)
        ),
        position_prior_weight=float(position_prior_weight),
    )


class ResidualSemanticAdapter(nn.Module):
    """Map primitive visual tokens into the fixed letter identity space."""

    def __init__(self, dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, primitive: torch.Tensor) -> torch.Tensor:
        return self.norm(primitive + self.adapter(primitive))


class IdentitySemanticAdapter(nn.Module):
    """Pass primitive features directly to DTW with no trainable projection."""

    def forward(self, primitive: torch.Tensor) -> torch.Tensor:
        return primitive


def _soft_stroke_target_from_normalized_patches(patches: torch.Tensor, contrast_scale: float):
    """Convert normalized RGB patches into soft foreground/stroke maps in [0,1]."""
    if patches.ndim != 5 or patches.shape[2] != 3:
        raise ValueError("Expected [B,T,3,H,W] normalized RGB patches")
    mean = patches.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = patches.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    rgb = (patches.float() * std + mean).clamp(0.0, 1.0)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]

    height, width = int(gray.shape[-2]), int(gray.shape[-1])
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.08)))
    border = torch.cat(
        [
            gray[..., :border_h, :].flatten(start_dim=2),
            gray[..., -border_h:, :].flatten(start_dim=2),
            gray[..., :, :border_w].flatten(start_dim=2),
            gray[..., :, -border_w:].flatten(start_dim=2),
        ],
        dim=-1,
    )
    background = border.median(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
    contrast = (gray - background).abs()
    scale = max(float(contrast_scale), 1e-3)
    target = (contrast / scale).clamp(0.0, 1.0)
    return target.unsqueeze(2)


def attach_restoration_dtw_stages(model, P):
    """Install ResNet-18 local windows, ViT-Tiny context and fused DTW features."""
    if getattr(model, "_restoration_positive_dtw_installed", False):
        return model
    if not hasattr(model, "vit_encoder"):
        raise TypeError("This branch expects the shared ViT window container")

    from resnet18_window_encoder import ResNet18WindowEncoder
    from pretrained_tiny_vit import initialize_tiny_vit_from_pretrained

    vit = model.vit_encoder
    dim = int(vit.embed_dim)
    if dim != 192:
        raise ValueError(
            f"ViT-Tiny requires a 192-D token space, got embed_dim={dim}. "
            "Build this branch with VECTOR_SIZE=192."
        )

    reference_parameter = next(vit.patch_embedding.parameters())
    device = reference_parameter.device
    dtype = reference_parameter.dtype

    vit.patch_embedding = ResNet18WindowEncoder(
        input_height=int(vit.input_height),
        window_size=int(vit.window_size),
        stride=int(vit.stride),
        embed_dim=dim,
        pretrained=bool(getattr(P, "resnet18_pretrained", True)),
        local_files_only=bool(getattr(P, "pretrained_local_only", True)),
        input_channels=int(getattr(P, "visual_input_channels", 1)),
    ).to(device=device, dtype=dtype)
    vit.restoration_local_encoder = "resnet18"
    vit.local_encoder_type = "resnet18"

    # The shared sequence transformer is required to match ViT-Tiny.
    if len(vit.encoder.layers) != 12:
        raise ValueError(
            f"Expected 12 ViT-Tiny transformer layers, got {len(vit.encoder.layers)}"
        )
    first_layer = vit.encoder.layers[0]
    if int(first_layer.self_attn.num_heads) != 3:
        raise ValueError(
            f"Expected 3 ViT-Tiny attention heads, got {first_layer.self_attn.num_heads}"
        )
    if int(first_layer.linear1.out_features) != 768:
        raise ValueError(
            f"Expected ViT-Tiny MLP width 768, got {first_layer.linear1.out_features}"
        )
    vit.vit_variant = "vit_tiny_192d_12l_3h"

    if bool(getattr(P, "tiny_vit_pretrained", False)):
        initialize_tiny_vit_from_pretrained(
            vit,
            model_name=str(
                getattr(
                    P,
                    "tiny_vit_pretrained_model",
                    "facebook/deit-tiny-patch16-224",
                )
            ),
            local_files_only=bool(getattr(P, "pretrained_local_only", True)),
        )
    else:
        vit.context_pretrained = False
        vit.context_pretrained_model = ""

    # Kept only for old branch metadata/API compatibility; not used in forward.
    vit.semantic_adapter = IdentitySemanticAdapter().to(device=device)
    vit.restoration_semantic_adapter = "identity"

    vit.fusion_head = LocalContextFusion(dim).to(device=device, dtype=dtype)

    def encode_restoration_sequence(self, image, *, use_flip):
        expected_channels = int(getattr(P, "visual_input_channels", 1))
        if image.ndim != 4 or int(image.shape[1]) != expected_channels:
            raise ValueError(
                f"Expected image [B,{expected_channels},H,W], got {tuple(image.shape)}"
            )
        if int(image.shape[2]) != self.input_height:
            raise ValueError("Unexpected input height")
        if int(image.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than the window width")

        model_input = image

        if _env_flag("PACK_VALID_WINDOWS", False):
            physical_valid, _pixel_valid = line_padding_masks(
                model_input,
                window_size=self.window_size,
                stride=self.stride,
                use_flip=False,
            )
            if not hasattr(self.patch_embedding, "forward_packed"):
                raise TypeError(
                    "PACK_VALID_WINDOWS requires a window encoder with "
                    "forward_packed()"
                )
            tokens, token_valid = self.patch_embedding.forward_packed(
                model_input,
                physical_valid,
                use_flip=use_flip,
            )
            if tokens.shape[2] != 1:
                raise RuntimeError("Window encoder must produce one token row")
            local = tokens.squeeze(2).transpose(1, 2).contiguous()
        else:
            tokens = self.patch_embedding(model_input)
            if tokens.shape[2] != 1:
                raise RuntimeError("Window encoder must produce one token row")
            local = tokens.squeeze(2).transpose(1, 2).contiguous()
            if use_flip:
                local = torch.flip(local, dims=[1])
            token_valid, _pixel_valid = line_padding_masks(
                model_input,
                window_size=self.window_size,
                stride=self.stride,
                use_flip=use_flip,
            )

        local = self.local_norm(local)

        positional = local + self._position_tokens(local.shape[1]).to(
            dtype=local.dtype, device=local.device
        )
        contextual = self.encoder(
            self.input_dropout(positional),
            src_key_padding_mask=~token_valid,
        )
        fused = self.fusion_head(local, contextual)
        return fused, local, contextual, model_input, token_valid

    def minimal_window_forward(self, image, *, use_flip, return_model_input=False):
        fused, local, _contextual, model_input, _token_valid = (
            self.encode_restoration_sequence(image, use_flip=use_flip)
        )
        if return_model_input:
            return fused, local, model_input
        return fused, local

    vit.encode_restoration_sequence = MethodType(encode_restoration_sequence, vit)
    vit.forward = MethodType(minimal_window_forward, vit)

    def model_forward(
        self,
        image,
        show_dims=False,
        return_local=False,
        return_ink=False,
        return_grouped=False,
        return_training_bundle=False,
    ):
        fused, local, contextual, model_input, token_valid = (
            self.vit_encoder.encode_restoration_sequence(
                image, use_flip=self.use_flip
            )
        )

        fused_pre_l2 = self.vision_norm(fused).float()
        fused_out = F.normalize(
            fused_pre_l2, p=2, dim=-1
        ).to(dtype=fused.dtype)
        local_out = self.vision_norm(local)

        if not return_training_bundle:
            if show_dims:
                print(
                    "image embeddings: ResNet18 + ViT-Tiny "
                    f"fused={tuple(fused_out.shape)} local={tuple(local_out.shape)} "
                    f"context={tuple(contextual.shape)}",
                    flush=True,
                )
            outputs = [fused_out]
            if return_local:
                outputs.append(local_out)
            if return_grouped:
                outputs.append(local_out)
            if return_ink:
                outputs.append(token_valid.float())
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        contextual_out = self.vision_norm(contextual)

        # Retain intermediate activation gradients for the per-batch diagnostic.
        # The training loop clears _gradient_probe_records before every batch.
        if self.training and torch.is_grad_enabled():
            local.retain_grad()
            contextual.retain_grad()
            fused.retain_grad()
            fused_pre_l2.retain_grad()
            fused_out.retain_grad()
            if not hasattr(self, "_gradient_probe_records"):
                self._gradient_probe_records = []
            self._gradient_probe_records.append(
                {
                    "after_resnet18": local,
                    "after_vit_tiny": contextual,
                    "after_fusion": fused,
                    "pre_l2_fused": fused_pre_l2,
                    "final_fused": fused_out,
                    "token_valid": token_valid,
                }
            )

        return {
            "semantic": fused_out,
            "fused": fused_out,
            "fused_pre_l2": fused_pre_l2,
            "primitive": local_out,
            "primitive_raw": local,
            "contextual": contextual_out,
            "ink": token_valid.float(),
            "token_valid": token_valid,
            "model_input": model_input,
        }

    model.forward = MethodType(model_forward, model)
    model._restoration_positive_dtw_installed = True
    return model


def letter_dtw_cost_matrix(P, text_encoder, visual: torch.Tensor, letters: list[str]):
    """Return the actual training cost matrix [windows, transcript letters].

    In full_alphabet_nll mode every window competes against the complete Arabic
    character inventory. The transcript columns then select the negative
    log-probability of the corresponding true character.
    """
    if visual.ndim != 2:
        raise ValueError("visual must be [T,D]")
    if not letters:
        return visual.new_empty((visual.shape[0], 0))

    visual_norm = F.normalize(visual.float(), p=2, dim=-1)
    with torch.no_grad():
        target = text_encoder("".join(letters)).detach().to(visual.device)
        target = F.normalize(target.float(), p=2, dim=-1)

    if str(P.positive_letter_dtw_cost_mode) == "full_alphabet_nll":
        inventory = list(dict.fromkeys(DEFAULT_ARABIC_LETTERS + "".join(letters)))
        with torch.no_grad():
            inventory_vectors = text_encoder("".join(inventory)).detach().to(
                visual.device
            )
            inventory_vectors = F.normalize(
                inventory_vectors.float(), p=2, dim=-1
            )
        temperature = max(
            1e-4, float(P.positive_letter_dtw_competition_temperature)
        )
        logits = torch.matmul(visual_norm, inventory_vectors.T) / temperature
        nll = -F.log_softmax(logits, dim=-1)
        lookup = {character: index for index, character in enumerate(inventory)}
        target_indices = torch.tensor(
            [lookup[character] for character in letters],
            dtype=torch.long,
            device=visual.device,
        )
        return nll.index_select(1, target_indices)

    return 1.0 - torch.matmul(visual_norm, target.T)


def positive_letter_dtw_loss(P, text_encoder, semantic_tokens, ink_ratios, positive_texts):
    losses = []
    windows_used = []
    letters_used = []
    path_cost_means = []

    for sample_index, text in enumerate(positive_texts):
        letters = _clean_letters(text)
        if not letters:
            continue

        visual = semantic_tokens[sample_index]
        if ink_ratios is not None:
            mask = ink_ratios[sample_index].to(visual.device).bool()
            if bool(mask.any()):
                visual = visual[mask]
        if visual.shape[0] == 0:
            continue

        costs = letter_dtw_cost_matrix(
            P, text_encoder, visual, letters
        )

        cost = _soft_dtw_cost_matrix(
            costs,
            gamma=float(P.positive_letter_dtw_gamma),
            vertical_penalty=float(P.positive_letter_dtw_vertical_penalty),
            horizontal_penalty=float(P.positive_letter_dtw_horizontal_penalty),
            position_prior_weight=float(P.positive_letter_dtw_position_prior),
            disable_horizontal_when_feasible=bool(
                P.positive_letter_dtw_disable_horizontal_when_feasible
            ),
        )
        losses.append(cost)
        windows_used.append(float(visual.shape[0]))
        letters_used.append(float(len(letters)))
        path_cost_means.append(float(cost.detach().item()))

    if not losses:
        zero = semantic_tokens.sum() * 0.0
        return zero, {
            "positive_letter_dtw": 0.0,
            "dtw_windows": 0.0,
            "dtw_letters": 0.0,
            "dtw_gamma": float(P.positive_letter_dtw_gamma),
            "dtw_vertical_penalty": float(P.positive_letter_dtw_vertical_penalty),
            "dtw_horizontal_penalty": float(P.positive_letter_dtw_horizontal_penalty),
        }

    loss = torch.stack(losses).mean()
    return loss, {
        "positive_letter_dtw": float(loss.detach().item()),
        "dtw_windows": sum(windows_used) / len(windows_used),
        "dtw_letters": sum(letters_used) / len(letters_used),
        "dtw_gamma": float(P.positive_letter_dtw_gamma),
        "dtw_vertical_penalty": float(P.positive_letter_dtw_vertical_penalty),
        "dtw_horizontal_penalty": float(P.positive_letter_dtw_horizontal_penalty),
    }


def negative_letter_dtw_margin_loss(
    P,
    text_encoder,
    semantic_tokens,
    token_valid,
    positive_texts,
    negative_texts,
):
    """Contrast positive transcript DTW against per-sample negative transcripts."""
    if not negative_texts or float(P.restoration_contrastive_weight) <= 0.0:
        return semantic_tokens.sum() * 0.0, {
            "negative_letter_dtw": 0.0,
            "contrastive_margin_loss": 0.0,
        }

    sample_losses = []
    negative_cost_values = []
    for sample_index, positive_text in enumerate(positive_texts):
        positives = _clean_letters(positive_text)
        if not positives:
            continue
        visual = semantic_tokens[sample_index]
        if token_valid is not None:
            valid = token_valid[sample_index].to(visual.device).bool()
            if bool(valid.any()):
                visual = visual[valid]
        if visual.shape[0] == 0:
            continue

        pos_costs = letter_dtw_cost_matrix(P, text_encoder, visual, positives)
        positive_cost = _soft_dtw_cost_matrix(
            pos_costs,
            gamma=float(P.positive_letter_dtw_gamma),
            vertical_penalty=float(P.positive_letter_dtw_vertical_penalty),
            horizontal_penalty=float(P.positive_letter_dtw_horizontal_penalty),
            position_prior_weight=float(P.positive_letter_dtw_position_prior),
            disable_horizontal_when_feasible=bool(
                P.positive_letter_dtw_disable_horizontal_when_feasible
            ),
        )
        sample_negative_costs = []
        for negative_text in negative_texts[sample_index]:
            negative_letters = _clean_letters(negative_text)
            if not negative_letters:
                continue
            neg_costs = letter_dtw_cost_matrix(
                P, text_encoder, visual, negative_letters
            )
            negative_cost = _soft_dtw_cost_matrix(
                neg_costs,
                gamma=float(P.positive_letter_dtw_gamma),
                vertical_penalty=float(P.positive_letter_dtw_vertical_penalty),
                horizontal_penalty=float(P.positive_letter_dtw_horizontal_penalty),
                position_prior_weight=float(P.positive_letter_dtw_position_prior),
                disable_horizontal_when_feasible=bool(
                    P.positive_letter_dtw_disable_horizontal_when_feasible
                ),
            )
            sample_negative_costs.append(negative_cost)
            negative_cost_values.append(float(negative_cost.detach().item()))
        if sample_negative_costs:
            sample_losses.append(
                contrastive_margin_from_costs(
                    positive_cost,
                    sample_negative_costs,
                    float(P.restoration_contrastive_margin),
                )
            )

    if not sample_losses:
        zero = semantic_tokens.sum() * 0.0
        return zero, {
            "negative_letter_dtw": 0.0,
            "contrastive_margin_loss": 0.0,
        }
    loss = torch.stack(sample_losses).mean()
    return loss, {
        "negative_letter_dtw": (
            sum(negative_cost_values) / max(1, len(negative_cost_values))
        ),
        "contrastive_margin_loss": float(loss.detach().item()),
    }


def strong_sigreg_loss(
    embeddings: torch.Tensor,
    token_valid: torch.Tensor | None = None,
    *,
    sketch_dim: int = 1024,
    num_knots: int = 17,
    t_min: float = 0.0,
    t_max: float = 3.0,
    min_samples: int = 32,
    slice_chunk: int = 128,
):
    """DDP-correct sliced Epps-Pulley SIGReg on valid pre-L2 fused tokens.

    The implementation follows the LeJEPA reference structure:
      * random unit directions shared by every DDP rank;
      * Epps-Pulley characteristic-function discrepancy on t in [0, 3];
      * trapezoid weights doubled to represent the symmetric integral;
      * the empirical characteristic function is computed over the GLOBAL
        valid-token population across all DDP ranks, not independently per GPU.

    Slices are processed in chunks to keep the temporary [N,S,K] trigonometric
    tensor small. SIGReg itself is float32 under AMP.
    """
    if embeddings.ndim != 3:
        raise ValueError("SIGReg expects fused embeddings [B,T,D]")

    values = embeddings.float()
    if token_valid is not None:
        valid = token_valid.to(device=values.device, dtype=torch.bool)
        if valid.shape != values.shape[:2]:
            raise ValueError(
                "SIGReg token_valid must match the [B,T] embedding prefix"
            )
        values = values[valid]
    else:
        values = values.reshape(-1, values.shape[-1])

    local_n = int(values.shape[0])
    d = int(values.shape[-1])
    distributed = dist.is_available() and dist.is_initialized()

    count = torch.tensor(float(local_n), device=values.device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_n = int(round(float(count.item())))

    if global_n < int(min_samples):
        zero = embeddings.float().sum() * 0.0
        return zero, {
            "sigreg_samples": float(global_n),
            "sigreg_dim_std_mean": 0.0,
            "sigreg_dim_std_min": 0.0,
            "sigreg_embedding_mean_norm": 0.0,
        }

    # Reference Epps-Pulley quadrature: use t >= 0 and double the trapezoid
    # weights so this represents the symmetric integral over negative t too.
    num_knots = int(num_knots)
    t_min = max(0.0, float(t_min))
    t_max = max(t_min + 1e-6, float(t_max))
    t = torch.linspace(
        t_min,
        t_max,
        num_knots,
        device=values.device,
        dtype=torch.float32,
    )
    if num_knots > 1:
        dt = (t_max - t_min) / float(num_knots - 1)
    else:
        dt = 1.0
    weights = torch.full(
        (num_knots,),
        2.0 * dt,
        device=values.device,
        dtype=torch.float32,
    )
    weights[0] = dt
    weights[-1] = dt
    gaussian_cf = torch.exp(-0.5 * t.square())
    weights = weights * gaussian_cf

    total_stat = values.new_zeros(())
    total_slices = max(1, int(sketch_dim))
    chunk_size = max(1, int(slice_chunk))

    for slice_start in range(0, total_slices, chunk_size):
        current = min(chunk_size, total_slices - slice_start)

        # Every rank must test the same slices. Generate on rank 0 and broadcast.
        directions = torch.empty(
            d,
            current,
            device=values.device,
            dtype=torch.float32,
        )
        if (not distributed) or dist.get_rank() == 0:
            directions.normal_()
            directions.div_(
                directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
            )
        if distributed:
            dist.broadcast(directions, src=0)

        projected = values @ directions
        args = projected.unsqueeze(-1) * t.view(1, 1, -1)
        local_moments = torch.stack(
            [
                torch.cos(args).sum(dim=0),
                torch.sin(args).sum(dim=0),
            ],
            dim=0,
        )

        if distributed:
            global_moments = differentiable_all_reduce(
                local_moments,
                op=dist.ReduceOp.SUM,
            )
        else:
            global_moments = local_moments

        empirical_real = global_moments[0] / float(global_n)
        empirical_imag = global_moments[1] / float(global_n)
        error = (
            (empirical_real - gaussian_cf.unsqueeze(0)).square()
            + empirical_imag.square()
        )
        per_slice = (error * weights.unsqueeze(0)).sum(dim=-1) * float(global_n)
        total_stat = total_stat + per_slice.sum()

    loss = total_stat / float(total_slices)

    # Global diagnostics only; they do not participate in the loss graph.
    with torch.no_grad():
        local_sum = values.sum(dim=0)
        local_sq = values.square().sum(dim=0)
        if distributed:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        mean = local_sum / float(global_n)
        var = (local_sq / float(global_n) - mean.square()).clamp_min(0.0)
        dim_std = var.sqrt()
        mean_norm = mean.norm(p=2)

    return loss, {
        "sigreg_samples": float(global_n),
        "sigreg_dim_std_mean": float(dim_std.mean().item()),
        "sigreg_dim_std_min": float(dim_std.min().item()),
        "sigreg_embedding_mean_norm": float(mean_norm.item()),
    }

def freeze_text_encoder(text_encoder):
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_encoder.eval()
    return text_encoder


def install_training_objective(train_module):
    """Install positive-DTW training with optional negative-margin DTW."""
    from textEmbedding import OrthogonalCharEmbedding

    def build_text_encoder():
        encoder = OrthogonalCharEmbedding(
            embedding_dim=int(train_module.P.vector_size),
            vocab_size=int(train_module.P.letter_codebook_vocab_size),
            seed=int(train_module.P.letter_codebook_seed),
        ).to(train_module.P.device)
        return freeze_text_encoder(encoder)

    train_module.build_text_encoder = build_text_encoder

    if hasattr(train_module, "_load_initial_states"):
        def load_initial_states(args, model, text_encoder):
            if args.resume:
                raise ValueError("Resume from an old objective is unsupported; start a new run")
            if not args.pretrained_weights:
                return None
            loaded = torch.load(args.pretrained_weights, map_location=train_module.P.device)
            loaded_config = loaded.get("model_config", {}) if isinstance(loaded, dict) else {}
            loaded_family = str(loaded_config.get("architecture_family", ""))
            loaded_backbone = str(
                loaded_config.get(
                    "local_encoder_type",
                    loaded_config.get("restoration_local_encoder", ""),
                )
            ).lower()
            if loaded_family == "restoration-positive-dtw-window-encoder" and loaded_backbone != "resnet18":
                raise ValueError(
                    "This checkpoint belongs to the old restoration/CNN revision. "
                    "The new branch uses ResNet-18 + ViT-Tiny and has no decoder; "
                    "start a fresh run or use a checkpoint produced by this revision."
                )

            state = train_module.extract_model_state(loaded)
            state = dict(state)
            target_state = model.state_dict()
            conv1_key = "vit_encoder.patch_embedding.backbone.conv1.weight"
            if conv1_key in state and conv1_key in target_state:
                source_conv = state[conv1_key]
                target_conv = target_state[conv1_key]
                if (
                    source_conv.ndim == 4
                    and target_conv.ndim == 4
                    and int(source_conv.shape[1]) == 3
                    and int(target_conv.shape[1]) == 1
                ):
                    state[conv1_key] = source_conv.mean(dim=1, keepdim=True)
                    if getattr(train_module, "CTX", None) is None or train_module.CTX.is_main:
                        print(
                            "Checkpoint migration: ResNet18 conv1 RGB -> grayscale "
                            f"{tuple(source_conv.shape)} -> {tuple(state[conv1_key].shape)}",
                            flush=True,
                        )
            incompatible = model.load_state_dict(state, strict=False)
            if loaded_family == "restoration-positive-dtw-window-encoder":
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "ResNet18/ViT-Tiny checkpoint did not load exactly: "
                        f"missing={incompatible.missing_keys[:10]} "
                        f"unexpected={incompatible.unexpected_keys[:10]}"
                    )
            freeze_text_encoder(text_encoder)
            return None

        train_module._load_initial_states = load_initial_states

    def loss_from_bundle(bundle, text_encoder, texts, negative_texts=None):
        dtw, dtw_stats = positive_letter_dtw_loss(
            train_module.P,
            text_encoder,
            bundle["semantic"],
            bundle["ink"],
            texts,
        )
        contrastive_weight = float(
            train_module.P.restoration_contrastive_weight
        )
        negatives_enabled = (
            contrastive_weight > 0.0
            and int(train_module.P.num_negatives) > 0
            and negative_texts is not None
        )
        if negatives_enabled:
            contrastive, contrastive_stats = negative_letter_dtw_margin_loss(
                train_module.P,
                text_encoder,
                bundle["semantic"],
                bundle["token_valid"],
                texts,
                negative_texts,
            )
        else:
            contrastive = dtw.new_zeros(())
            contrastive_stats = {
                "negative_letter_dtw": 0.0,
                "contrastive_margin_loss": 0.0,
            }

        sigreg_weight = float(train_module.P.sigreg_weight)
        if sigreg_weight > 0.0:
            sigreg, sigreg_stats = strong_sigreg_loss(
                bundle["fused_pre_l2"],
                bundle["token_valid"],
                sketch_dim=int(train_module.P.sigreg_sketch_dim),
                num_knots=int(train_module.P.sigreg_num_knots),
                t_min=float(train_module.P.sigreg_t_min),
                t_max=float(train_module.P.sigreg_t_max),
                min_samples=int(train_module.P.sigreg_min_samples),
                slice_chunk=int(train_module.P.sigreg_slice_chunk),
            )
        else:
            sigreg = dtw.new_zeros(())
            sigreg_stats = {
                "sigreg_samples": 0.0,
                "sigreg_dim_std_mean": 0.0,
                "sigreg_dim_std_min": 0.0,
                "sigreg_embedding_mean_norm": 0.0,
            }

        total = (
            float(train_module.P.positive_letter_dtw_weight) * dtw
            + contrastive_weight * contrastive
            + sigreg_weight * sigreg
        )
        stats = {
            **dtw_stats,
            **contrastive_stats,
            "norm_pos": float(dtw.detach().item()),
            "norm_neg": float(contrastive_stats["negative_letter_dtw"]),
            "cost_pos": float(dtw.detach().item()),
            "cost_neg": float(contrastive_stats["negative_letter_dtw"]),
            "gap": float(
                contrastive_stats["negative_letter_dtw"] - float(dtw.detach().item())
            ),
            "local_hard_neg": 0.0,
            "image_pair_loss": 0.0,
            "order_loss": 0.0,
            "pair_terms": 0.0,
            "img_var_loss": 0.0,
            "sigreg_loss": float(sigreg.detach().item()),
            "sigreg_weight": sigreg_weight,
            "sigreg_weighted": float((sigreg_weight * sigreg).detach().item()),
            **sigreg_stats,
            "total": float(total.detach().item()),
        }
        return total, stats

    def single_line_loss(
        image_embedder, text_encoder, images, texts, negative_texts=None
    ):
        with train_module.autocast(
            dtype=train_module.AMP_DTYPE,
            enabled=train_module.USE_AMP,
        ):
            bundle = image_embedder(images, return_training_bundle=True)
        return loss_from_bundle(bundle, text_encoder, texts, negative_texts)

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        del criterion
        if isinstance(batch, dict):
            images1 = batch["images1"].to(train_module.P.device, non_blocking=True)
            images2 = batch["images2"].to(train_module.P.device, non_blocking=True)
            if images1.shape[1:] != images2.shape[1:]:
                raise ValueError(
                    "Paired lines must have identical post-transform geometry for "
                    "the single-forward DDP path, got "
                    f"{tuple(images1.shape)} and {tuple(images2.shape)}"
                )

            pair_batch = int(images1.shape[0])
            combined_images = torch.cat([images1, images2], dim=0)
            with train_module.autocast(
                dtype=train_module.AMP_DTYPE,
                enabled=train_module.USE_AMP,
            ):
                combined = image_embedder(
                    combined_images, return_training_bundle=True
                )

            bundle1 = {
                "semantic": combined["semantic"][:pair_batch],
                "fused_pre_l2": combined["fused_pre_l2"][:pair_batch],
                "ink": combined["ink"][:pair_batch],
                "token_valid": combined["token_valid"][:pair_batch],
            }
            bundle2 = {
                "semantic": combined["semantic"][pair_batch:],
                "fused_pre_l2": combined["fused_pre_l2"][pair_batch:],
                "ink": combined["ink"][pair_batch:],
                "token_valid": combined["token_valid"][pair_batch:],
            }
            loss1, stats1 = loss_from_bundle(
                bundle1,
                text_encoder,
                batch["texts1"],
                batch.get("neg_texts1"),
            )
            loss2, stats2 = loss_from_bundle(
                bundle2,
                text_encoder,
                batch["texts2"],
                batch.get("neg_texts2"),
            )
            loss = 0.5 * (loss1 + loss2)
            stats = train_module.average_stats([stats1, stats2])
            stats["independent_lines_per_pair"] = 2.0
            stats["model_forwards_per_batch"] = 1.0
            stats["total"] = float(loss.detach().item())
            return loss, stats

        images, texts, negative_texts = batch
        images = images.to(train_module.P.device, non_blocking=True)
        loss, stats = single_line_loss(
            image_embedder, text_encoder, images, texts, negative_texts
        )
        stats["independent_lines_per_pair"] = 1.0
        stats["model_forwards_per_batch"] = 1.0
        return loss, stats

    train_module.compute_batch_loss = compute_batch_loss

    def epoch_start_hook(*, epoch, total_epochs):
        del total_epochs
        start = max(1e-5, float(train_module.P.positive_letter_dtw_gamma_start))
        end = max(1e-5, float(train_module.P.positive_letter_dtw_gamma_end))
        anneal_epochs = max(1, int(train_module.P.positive_letter_dtw_anneal_epochs))
        progress = min(1.0, max(0.0, (int(epoch) - 1) / max(1, anneal_epochs - 1)))
        gamma = start * ((end / start) ** progress)
        train_module.P.positive_letter_dtw_gamma = float(gamma)

    train_module.epoch_start_hook = epoch_start_hook

    def epoch_diagnostic_hook(
        *,
        model,
        text_encoder,
        valid_loader,
        epoch,
        job_id,
        config,
        device,
    ):
        del model, text_encoder, valid_loader, device
        if not _env_flag("TRAIN_SAMPLE_EVAL", False):
            return

        requested = {
            int(value.strip())
            for value in os.environ.get(
                "TRAIN_SAMPLE_EVAL_EPOCHS", "1,5,10,15,20,25,30"
            ).split(",")
            if value.strip()
        }
        epoch = int(epoch)
        if epoch not in requested:
            return

        weights = (
            Path(train_module.weights_dir(job_id))
            / f"model_epoch_{epoch:03d}.pth"
        )
        if not weights.is_file():
            raise FileNotFoundError(
                f"Epoch diagnostic checkpoint not found: {weights}"
            )

        dataset = Path(str(config.get("data_dir", ""))).expanduser().resolve()
        if not dataset.exists():
            raise FileNotFoundError(
                f"Epoch diagnostic dataset not found: {dataset}"
            )

        start_index = max(
            1, _env_int("TRAIN_SAMPLE_EVAL_START_INDEX", 1)
        )
        eval_device = os.environ.get(
            "TRAIN_SAMPLE_EVAL_DEVICE", "cpu"
        ).strip() or "cpu"
        threshold = _env_float("TRAIN_SAMPLE_EVAL_THRESHOLD", 0.10)
        output_root = (
            Path("Results")
            / "Evaluation"
            / "Point2"
            / "train_epoch_sample"
            / str(job_id)
            / f"epoch_{epoch:03d}"
        )
        if output_root.exists() and any(output_root.iterdir()):
            output_root = output_root / f"rerun_{time.time_ns()}_{os.getpid()}"
        nw_output = output_root / "nw"
        dtw_output = output_root / "dtw"

        env = os.environ.copy()
        # Evaluation must be deterministic/clean and must match the exact
        # deterministic preprocessing used by this real training run:
        # XML four-sided bbox crop -> grayscale -> aspect-preserving 1024x128.
        # Only stochastic scan corruption is disabled.
        env.update(
            {
                "DATASET_TYPE": "real",
                "LINE_HEIGHT": "128",
                "LINE_WIDTH": "1024",
                "PACK_VALID_WINDOWS": "0",
                "REAL_SYNTHETIC_STYLE": "0",
                "REAL_BINARIZE": "0",
                "REAL_BINARIZE_AUTO_INVERT": "0",
                "REAL_BINARIZE_AUTOCONTRAST": "0",
                "REAL_AUGMENT": "0",
                "REAL_SCAN_AUGMENT": "0",
                "VISUAL_INPUT_CHANNELS": "1",
                "VISUAL_GRAYSCALE": "1",
                "REAL_GRAYSCALE": "1",
                "REAL_BBOX_CROP": "1",
                "REAL_BBOX_CROP_STRICT": "1",
                "REAL_BBOX_MARGIN_RATIO": os.environ.get(
                    "REAL_BBOX_MARGIN_RATIO", "0.05"
                ),
                "REAL_BBOX_MIN_MARGIN_PX": os.environ.get(
                    "REAL_BBOX_MIN_MARGIN_PX", "2"
                ),
                "ZERO_SHOT_PREPROCESS": "1",
                "ZERO_SHOT_FOREGROUND_CROP": "0",
                "ZERO_SHOT_PRESERVE_ASPECT": "1",
                "ZERO_SHOT_SOURCE_GEOMETRY": "0",
                "ZERO_SHOT_TARGET_INK_HEIGHT_RATIO": os.environ.get(
                    "ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72"
                ),
                "LINE_GEOMETRY_MODE": "xml-bbox-gray-aspect-preserving",
                "EVAL_COMPACT_OUTPUT": "1",
            }
        )
        env["PYTHONPATH"] = (
            str(Path(__file__).resolve().parent)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        common = [
            "--dataset", str(dataset),
            "--split", "test",
            "--training-samples", "6000",
            "--split-seed", "42",
            "--n-samples", "1",
            "--start-index", str(start_index),
            "--device", eval_device,
            "--image-preprocessing", "training",
        ]

        print(
            f"EPOCH_SAMPLE_EVAL epoch={epoch} sample={start_index} "
            f"checkpoint={weights} device={eval_device}",
            flush=True,
        )

        nw_cmd = [
            sys.executable,
            "-u",
            "-m",
            "Evaluation.eval_point2",
            "--point2-representation", "fused",
            "--weights", str(weights),
            "--branch", "restoration",
            "--alignment-unit", "window",
            "--word-support-floor", "0.0",
            "--min-aligned-windows", "5",
            "--score-mode", "raw",
            "--threshold", str(threshold),
            "--gap", "-0.30",
            "--output-dir", str(nw_output),
            *common,
        ]
        nw_env = dict(env)
        nw_env["EVAL_COSINE_TRACE"] = "nw"
        nw_env["EVAL_PRIMARY_ALIGNMENT"] = "nw"
        diagnostic_failures = []

        try:
            subprocess.run(nw_cmd, check=True, env=nw_env)
        except subprocess.CalledProcessError as exc:
            diagnostic_failures.append(f"NW exit={exc.returncode}")
            print(
                f"EPOCH_SAMPLE_EVAL_WARNING epoch={epoch} method=NW "
                f"exit={exc.returncode}; training will continue",
                flush=True,
            )

        dtw_cmd = [
            sys.executable,
            "-u",
            "-m",
            "Evaluation.eval_point3_hard_paths",
            "--weights", str(weights),
            "--output-dir", str(dtw_output),
            *common,
        ]
        try:
            subprocess.run(dtw_cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            diagnostic_failures.append(f"DTW exit={exc.returncode}")
            print(
                f"EPOCH_SAMPLE_EVAL_WARNING epoch={epoch} method=DTW "
                f"exit={exc.returncode}; training will continue",
                flush=True,
            )

        print(
            f"EPOCH_SAMPLE_EVAL_DONE epoch={epoch} "
            f"nw={nw_output} dtw={dtw_output} "
            f"status={'ok' if not diagnostic_failures else '; '.join(diagnostic_failures)}",
            flush=True,
        )

    train_module.epoch_diagnostic_hook = epoch_diagnostic_hook
    train_module.save_d3tw_visualization = lambda *args, **kwargs: None

    if hasattr(train_module, "wandb_log_epoch_metrics"):
        original_logger = train_module.wandb_log_epoch_metrics

        def wandb_log_epoch_metrics(run, epoch, train_loss, val_loss, train_stats):
            if run is not None and getattr(train_module, "wandb", None) is not None:
                train_module.wandb.log(
                    {
                        "minimal/positive_letter_dtw": float(
                            train_stats.get("positive_letter_dtw", 0.0)
                        ),
                        "minimal/contrastive_margin_dtw": float(
                            train_stats.get("contrastive_margin_loss", 0.0)
                        ),
                        "minimal/negative_letter_dtw": float(
                            train_stats.get("negative_letter_dtw", 0.0)
                        ),
                        "minimal/dtw_windows": float(train_stats.get("dtw_windows", 0.0)),
                        "minimal/dtw_letters": float(train_stats.get("dtw_letters", 0.0)),
                        "minimal/dtw_gamma": float(
                            train_stats.get(
                                "dtw_gamma",
                                train_module.P.positive_letter_dtw_gamma,
                            )
                        ),
                        "minimal/sigreg_loss": float(
                            train_stats.get("sigreg_loss", 0.0)
                        ),
                        "minimal/sigreg_weighted": float(
                            train_stats.get("sigreg_weighted", 0.0)
                        ),
                        "minimal/sigreg_dim_std_mean": float(
                            train_stats.get("sigreg_dim_std_mean", 0.0)
                        ),
                        "minimal/sigreg_dim_std_min": float(
                            train_stats.get("sigreg_dim_std_min", 0.0)
                        ),
                    },
                    step=int(epoch),
                    commit=False,
                )
            return original_logger(run, epoch, train_loss, val_loss, train_stats)

        train_module.wandb_log_epoch_metrics = wandb_log_epoch_metrics


def model_config(P):
    return {
        "architecture_family": "restoration-positive-dtw-window-encoder",
        "architecture_revision": "pretrained-resnet18-deit-tiny-no-restoration",
        "training_stage": "align",
        "training_supervision": (
            (
                "positive+negative letter-dtw + strong sigreg"
                if float(P.restoration_contrastive_weight) > 0.0
                and int(P.num_negatives) > 0
                else "positive letter-dtw + strong sigreg"
            )
            if float(P.sigreg_weight) > 0.0
            else (
                "positive+negative letter-dtw"
                if float(P.restoration_contrastive_weight) > 0.0
                and int(P.num_negatives) > 0
                else "positive letter-dtw only"
            )
        ),
        "local_encoder_type": "resnet18",
        "visual_input_channels": int(P.visual_input_channels),
        "visual_grayscale": bool(P.visual_grayscale),
        "real_bbox_crop": _env_flag("REAL_BBOX_CROP", False),
        "real_bbox_margin_ratio": _env_float("REAL_BBOX_MARGIN_RATIO", 0.05),
        "real_bbox_min_margin_px": _env_int("REAL_BBOX_MIN_MARGIN_PX", 2),
        "real_scan_augment": _env_flag("REAL_SCAN_AUGMENT", False),
        "real_scan_augment_probability": _env_float(
            "REAL_SCAN_AUGMENT_PROB", 0.95
        ),
        "real_scan_blur_probability": _env_float("REAL_SCAN_BLUR_PROB", 0.55),
        "real_scan_blur_radius_min": _env_float(
            "REAL_SCAN_BLUR_RADIUS_MIN", 0.35
        ),
        "real_scan_blur_radius_max": _env_float(
            "REAL_SCAN_BLUR_RADIUS_MAX", 1.40
        ),
        "real_scan_gaussian_noise_probability": _env_float(
            "REAL_SCAN_GAUSSIAN_NOISE_PROB", 0.75
        ),
        "real_scan_gaussian_noise_std_min": _env_float(
            "REAL_SCAN_GAUSSIAN_NOISE_STD_MIN", 5.0
        ),
        "real_scan_gaussian_noise_std_max": _env_float(
            "REAL_SCAN_GAUSSIAN_NOISE_STD_MAX", 16.0
        ),
        "restoration_local_encoder": "resnet18",
        "resnet18_pretrained": bool(P.resnet18_pretrained),
        "resnet18_pretrained_source": "torchvision/ResNet18_Weights.DEFAULT",
        "resnet18_feature_dim": 512,
        "visual_input_channels": int(P.visual_input_channels),
        "visual_grayscale": bool(P.visual_grayscale),
        "resnet18_conv1_input_channels": int(P.visual_input_channels),
        "tiny_vit_pretrained": bool(P.tiny_vit_pretrained),
        "tiny_vit_pretrained_model": str(P.tiny_vit_pretrained_model),
        "pretrained_local_only": bool(P.pretrained_local_only),
        "vit_variant": "vit_tiny",
        "vit_embed_dim": 192,
        "vit_layers": 12,
        "vit_heads": 3,
        "vit_mlp_dim": 768,
        "decoder_present": False,
        "restoration_loss_active": False,
        "primary_representation": "normalized-local-context-fusion",
        "local_representation": "resnet18-window-token",
        "restoration_semantic_adapter": "identity",
        "semantic_projection_trainable": False,
        "dtw_representation": "normalized-local-context-fusion",
        "context_transformer_active": True,
        "context_transformer_layers": 12,
        "fusion_mode": str(P.restoration_fusion),
        "negative_transcripts": int(P.num_negatives),
        "contrastive_dtw_weight": float(P.restoration_contrastive_weight),
        "contrastive_dtw_margin": float(P.restoration_contrastive_margin),
        "image_pair_supervision": False,
        "text_encoder_type": "char",
        "letter_codebook": "frozen-orthogonal-character-identities",
        "letter_codebook_seed": int(P.letter_codebook_seed),
        "letter_codebook_vocab_size": int(P.letter_codebook_vocab_size),
        "letter_inventory": str(P.letter_inventory),
        "positive_letter_dtw_weight": float(P.positive_letter_dtw_weight),
        "sigreg_active": bool(float(P.sigreg_weight) > 0.0),
        "sigreg_weight": float(P.sigreg_weight),
        "sigreg_target": "pre-l2 fused valid-window representation",
        "sigreg_variant": "strong-ecf-epps-pulley",
        "sigreg_sketch_dim": int(P.sigreg_sketch_dim),
        "sigreg_num_knots": int(P.sigreg_num_knots),
        "sigreg_t_min": float(P.sigreg_t_min),
        "sigreg_t_max": float(P.sigreg_t_max),
        "sigreg_min_samples": int(P.sigreg_min_samples),
        "sigreg_slice_chunk": int(P.sigreg_slice_chunk),
        "sigreg_ddp_distribution": "global-valid-token-population",
        "sigreg_shared_slices_across_ranks": True,
        "positive_letter_dtw_gamma": float(P.positive_letter_dtw_gamma),
        "positive_letter_dtw_gamma_start": float(P.positive_letter_dtw_gamma_start),
        "positive_letter_dtw_gamma_end": float(P.positive_letter_dtw_gamma_end),
        "positive_letter_dtw_anneal_epochs": int(P.positive_letter_dtw_anneal_epochs),
        "positive_letter_dtw_step_penalty": float(P.positive_letter_dtw_step_penalty),
        "positive_letter_dtw_vertical_penalty": float(
            P.positive_letter_dtw_vertical_penalty
        ),
        "positive_letter_dtw_horizontal_penalty": float(
            P.positive_letter_dtw_horizontal_penalty
        ),
        "positive_letter_dtw_position_prior": float(
            P.positive_letter_dtw_position_prior
        ),
        "positive_letter_dtw_disable_horizontal_when_feasible": bool(
            P.positive_letter_dtw_disable_horizontal_when_feasible
        ),
        "positive_letter_dtw_competition_temperature": float(
            P.positive_letter_dtw_competition_temperature
        ),
        "positive_letter_dtw_cost_mode": str(P.positive_letter_dtw_cost_mode),
        "positive_letter_dtw_min_ink": float(P.positive_letter_dtw_min_ink),
        "real_binarize": bool(P.real_binarize),
        "real_synthetic_style": _env_flag("REAL_SYNTHETIC_STYLE", False),
        "real_independent_lines": _env_flag("REAL_INDEPENDENT_LINES", False),
        "real_all_page_lines": _env_flag("REAL_ALL_PAGE_LINES", False),
        "real_manifest_name": os.environ.get(
            "REAL_MANIFEST_NAME", "dataset_manifest.jsonl"
        ),
        "real_pair_labels_used_for_training": not (
            _env_flag("REAL_INDEPENDENT_LINES", False)
            or _env_flag("REAL_ALL_PAGE_LINES", False)
        ),
        "real_positive_partner_required": False,
        "pack_valid_windows": _env_flag("PACK_VALID_WINDOWS", False),
        "real_output_polarity": (
            "white_ink_on_black"
            if _env_flag("REAL_SYNTHETIC_STYLE", False)
            else (
                "source_grayscale"
                if bool(P.visual_grayscale)
                else "source_rgb"
            )
        ),
        "synthetic_binarize": False,
        "zero_shot_preprocess": True,
        "zero_shot_preserve_aspect": _env_flag(
            "ZERO_SHOT_PRESERVE_ASPECT", True
        ),
        "zero_shot_foreground_crop": _env_flag(
            "ZERO_SHOT_FOREGROUND_CROP", True
        ),
        "target_ink_height_ratio": _env_float(
            "ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72
        ),
        "line_geometry_mode": os.environ.get(
            "LINE_GEOMETRY_MODE",
            (
                "xml-bbox-gray-aspect-preserving"
                if bool(P.visual_grayscale)
                and _env_flag("REAL_BBOX_CROP", False)
                else "crop-aspect-preserving-rgb"
            ),
        ),
        "keep_paired_lines_for_independent_training": bool(
            P.keep_paired_lines_for_independent_training
        ),
    }

"""Arabic window encoder with ResNet-18 local features, ViT-Tiny context and DTW.

Pipeline:
  crop outer margins -> proportional resize -> overlapping real RGB windows ->
  shared ResNet-18 window encoder -> ViT-Tiny sequence context ->
  local/context fusion -> positive/negative letter-DTW.

There is intentionally no restoration decoder and no reconstruction loss on
this revision. The frozen character codebook supervises only the fused visual
representation during training; final evaluation remains image-only.
"""
from __future__ import annotations

from types import MethodType
import os
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # Keep the current line geometry and original RGB input.
    P.window_size = 32
    P.stride_ratio = _env_float("RESTORATION_DTW_STRIDE_RATIO", 0.50)
    P.vit_input_height = 128
    P.vit_binarize_input = False
    P.real_binarize = False
    P.real_binarize_autocontrast = False
    P.real_augment = False
    P.zero_shot_preprocess = True
    P.zero_shot_preserve_aspect = True
    P.zero_shot_foreground_crop = True
    P.zero_shot_source_geometry = False
    os.environ["SYNTHETIC_BINARIZE"] = "0"
    os.environ["REAL_BINARIZE"] = "0"
    os.environ["REAL_BINARIZE_AUTOCONTRAST"] = "0"
    os.environ["ZERO_SHOT_FOREGROUND_CROP"] = "1"
    os.environ["ZERO_SHOT_PRESERVE_ASPECT"] = "1"
    os.environ["ZERO_SHOT_SOURCE_GEOMETRY"] = "0"
    os.environ["LINE_GEOMETRY_MODE"] = "crop-aspect-preserving-rgb"

    # Canonical ViT-Tiny dimensions. ResNet-18 creates one token per physical
    # window; the transformer contextualizes that window-token sequence.
    P.vector_size = 192
    P.vit_layers = 12
    P.vit_heads = 3
    P.vit_mlp_dim = 768
    P.vit_dropout = _env_float("TINY_VIT_DROPOUT", 0.0)
    P.vit_max_tokens = max(int(getattr(P, "vit_max_tokens", 256)), 256)
    P.restoration_context_layers = P.vit_layers
    P.restoration_fusion = "concat_projection_norm"
    P.restoration_local_encoder = "resnet18"
    P.resnet18_pretrained = _env_flag("RESNET18_PRETRAINED", False)
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

    # Positive transcript DTW + negative transcript margin DTW.
    P.num_negatives = _env_int("RESTORATION_NUM_NEGATIVES", 10)
    P.restoration_contrastive_weight = _env_float(
        "RESTORATION_CONTRASTIVE_WEIGHT", 0.50
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

    # Legacy compatibility attribute: reconstruction is intentionally disabled.
    P.restoration_weight = 0.0
    # Diagnostic mode: one backward/optimizer update per batch; no accumulation.\n    P.gradient_accumulation_steps = 1\n

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
        pretrained=bool(getattr(P, "resnet18_pretrained", False)),
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

    # Kept only for old branch metadata/API compatibility; not used in forward.
    vit.semantic_adapter = IdentitySemanticAdapter().to(device=device)
    vit.restoration_semantic_adapter = "identity"

    vit.fusion_head = LocalContextFusion(dim).to(device=device, dtype=dtype)

    def encode_restoration_sequence(self, image, *, use_flip):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected image [B,3,H,W]")
        if int(image.shape[2]) != self.input_height:
            raise ValueError("Unexpected input height")
        if int(image.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than the window width")

        model_input = image
        tokens = self.patch_embedding(model_input)
        if tokens.shape[2] != 1:
            raise RuntimeError("Window encoder must produce one token row")
        local = tokens.squeeze(2).transpose(1, 2).contiguous()
        if use_flip:
            local = torch.flip(local, dims=[1])
        local = self.local_norm(local)

        token_valid, _pixel_valid = line_padding_masks(
            model_input,
            window_size=self.window_size,
            stride=self.stride,
            use_flip=use_flip,
        )
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

        fused_out = F.normalize(
            self.vision_norm(fused).float(), p=2, dim=-1
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
            fused_out.retain_grad()
            if not hasattr(self, "_gradient_probe_records"):
                self._gradient_probe_records = []
            self._gradient_probe_records.append(
                {
                    "after_resnet18": local,
                    "after_vit_tiny": contextual,
                    "after_fusion": fused,
                    "final_fused": fused_out,
                }
            )

        return {
            "semantic": fused_out,
            "fused": fused_out,
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


def freeze_text_encoder(text_encoder):
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_encoder.eval()
    return text_encoder


def install_training_objective(train_module):
    """Install the positive-DTW + negative-margin-DTW objective."""
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
            incompatible = model.load_state_dict(state, strict=False)
            if loaded_family == "restoration-positive-dtw-window-encoder":
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "ResNet18/ViT-Tiny checkpoint did not load exactly: "
                        f"missing={incompatible.missing_keys[:10]} "
                        f"unexpected={incompatible.unexpected_keys[:10]}"
                    )
            if train_module.CTX.is_main:
                print(
                    "Loaded visual initialization: "
                    f"missing={len(incompatible.missing_keys)} "
                    f"unexpected={len(incompatible.unexpected_keys)}",
                    flush=True,
                )
            freeze_text_encoder(text_encoder)
            return None

        train_module._load_initial_states = load_initial_states

    def single_line_loss(
        image_embedder, text_encoder, images, texts, negative_texts=None
    ):
        with train_module.autocast(
            dtype=train_module.AMP_DTYPE,
            enabled=train_module.USE_AMP,
        ):
            bundle = image_embedder(images, return_training_bundle=True)

        dtw, dtw_stats = positive_letter_dtw_loss(
            train_module.P,
            text_encoder,
            bundle["semantic"],
            bundle["ink"],
            texts,
        )
        contrastive, contrastive_stats = negative_letter_dtw_margin_loss(
            train_module.P,
            text_encoder,
            bundle["semantic"],
            bundle["token_valid"],
            texts,
            negative_texts,
        )
        total = (
            float(train_module.P.positive_letter_dtw_weight) * dtw
            + float(train_module.P.restoration_contrastive_weight) * contrastive
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
            "total": float(total.detach().item()),
        }
        return total, stats

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        del criterion
        if isinstance(batch, dict):
            images1 = batch["images1"].to(train_module.P.device, non_blocking=True)
            images2 = batch["images2"].to(train_module.P.device, non_blocking=True)
            loss1, stats1 = single_line_loss(
                image_embedder,
                text_encoder,
                images1,
                batch["texts1"],
                batch.get("neg_texts1"),
            )
            loss2, stats2 = single_line_loss(
                image_embedder,
                text_encoder,
                images2,
                batch["texts2"],
                batch.get("neg_texts2"),
            )
            loss = 0.5 * (loss1 + loss2)
            stats = train_module.average_stats([stats1, stats2])
            stats["independent_lines_per_pair"] = 2.0
            stats["total"] = float(loss.detach().item())
            return loss, stats

        images, texts, negative_texts = batch
        images = images.to(train_module.P.device, non_blocking=True)
        loss, stats = single_line_loss(
            image_embedder, text_encoder, images, texts, negative_texts
        )
        stats["independent_lines_per_pair"] = 1.0
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
        if train_module.CTX.is_main:
            print(
                "[letter-dtw-curriculum] "
                f"epoch={epoch} gamma={gamma:.5f} "
                f"v_penalty={train_module.P.positive_letter_dtw_vertical_penalty:.3f} "
                f"h_penalty={train_module.P.positive_letter_dtw_horizontal_penalty:.3f} "
                f"position_prior={train_module.P.positive_letter_dtw_position_prior:.3f}",
                flush=True,
            )

    train_module.epoch_start_hook = epoch_start_hook
    train_module.epoch_diagnostic_hook = None
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
                    },
                    step=int(epoch),
                    commit=False,
                )
            return original_logger(run, epoch, train_loss, val_loss, train_stats)

        train_module.wandb_log_epoch_metrics = wandb_log_epoch_metrics


def model_config(P):
    return {
        "architecture_family": "restoration-positive-dtw-window-encoder",
        "architecture_revision": "resnet18-vit-tiny-no-restoration",
        "training_stage": "align",
        "training_supervision": "positive+negative letter-dtw only",
        "local_encoder_type": "resnet18",
        "restoration_local_encoder": "resnet18",
        "resnet18_pretrained": bool(P.resnet18_pretrained),
        "resnet18_feature_dim": 512,
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
        "synthetic_binarize": False,
        "zero_shot_preprocess": True,
        "zero_shot_preserve_aspect": True,
        "zero_shot_foreground_crop": True,
        "keep_paired_lines_for_independent_training": bool(
            P.keep_paired_lines_for_independent_training
        ),
    }


"""Single-line Arabic window encoder with stroke restoration and positive-only DTW.

Active supervision is intentionally minimal:
  1) stroke restoration from primitive window tokens;
  2) positive monotonic letter DTW against a fixed frozen character codebook.

There are no negative transcripts, no image-image contrastive loss, no pair
cross-attention, no contextual Span-DTW, no dense alphabet classifier, and no
variance/context-consistency auxiliary losses in this experiment.
"""
from __future__ import annotations

from types import MethodType
import os
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def apply_branch_config(P):
    """Configure the minimal restoration + positive-DTW experiment."""
    P.experiment_name = "vit_restoration_positive_dtw"

    # Branch-local experiment controls make short diagnostic runs possible
    # without editing the shared Parameters.py.
    P.epochs = _env_int("RESTORATION_EPOCHS", int(P.epochs))
    P.finetune_epochs = P.epochs
    P.learning_rate = _env_float(
        "RESTORATION_LEARNING_RATE", float(P.learning_rate)
    )
    P.finetune_learning_rate = P.learning_rate
    P.num_samples = _env_int("RESTORATION_NUM_SAMPLES", int(P.num_samples))
    P.restoration_training_stage = os.environ.get(
        "RESTORATION_TRAINING_STAGE", "align"
    ).strip().lower()
    if P.restoration_training_stage not in {"pretrain", "align"}:
        raise ValueError("RESTORATION_TRAINING_STAGE must be pretrain or align")

    # Visual geometry. The Transformer is not part of the active architecture;
    # one frozen layer remains only because the shared ViT container requires it.
    P.window_size = 32
    P.stride_ratio = _env_float("RESTORATION_DTW_STRIDE_RATIO", 0.50)
    P.vit_layers = 1
    P.vit_binarize_input = False
    P.real_binarize = False
    # Existing real augmentation is designed around binary ink. Keep the first
    # minimal experiment on the original resized line rather than mixing in a
    # different preprocessing objective.
    P.real_augment = False
    P.vit_max_tokens = max(int(getattr(P, "vit_max_tokens", 256)), 256)
    # No Span-DTW/JAX objective is active on this branch.
    P.span_dtw_backend = "torch"

    # The target space is a fixed character identity codebook, not AraBERT spans.
    P.text_encoder_type = "char"
    # These span values are inactive, but shared optimization validation still
    # inspects them during startup. Keep them in a harmless valid range.
    P.max_text_span_chars = 1
    P.max_text_token_chars = 1
    P.letter_codebook_seed = _env_int("LETTER_CODEBOOK_SEED", 1234)
    P.letter_codebook_vocab_size = _env_int("LETTER_CODEBOOK_VOCAB_SIZE", 4096)
    # Informational label: actual targets accept Unicode Arabic letters after
    # NFKC normalization rather than filtering through a closed alphabet list.
    P.letter_inventory = "unicode-arabic-letters-after-nfkc"

    # New runs use a restoration-pretrained local CNN over each 128x32 window.
    # Historical checkpoints used one full-height Conv2d projection.
    P.restoration_local_encoder = os.environ.get(
        "RESTORATION_LOCAL_ENCODER", "cnn_seq2seq"
    ).strip().lower()
    if P.restoration_local_encoder not in {"cnn_seq2seq", "fullheight_conv"}:
        raise ValueError(
            "RESTORATION_LOCAL_ENCODER must be cnn_seq2seq or fullheight_conv"
        )

    # New diagnostic-first architecture: supervise the primitive window encoder
    # directly with positive letter-DTW. "identity" removes the residual
    # 128->256->128 MLP entirely. "residual_mlp" is retained only so historical
    # checkpoints from this branch remain loadable/evaluable.
    P.restoration_semantic_adapter = os.environ.get(
        "RESTORATION_SEMANTIC_ADAPTER", "identity"
    ).strip().lower()
    if P.restoration_semantic_adapter not in {"identity", "residual_mlp"}:
        raise ValueError(
            "RESTORATION_SEMANTIC_ADAPTER must be identity or residual_mlp"
        )

    # Keep both sides of an available pair only as two independent training lines.
    P.keep_paired_lines_for_independent_training = True
    P.image_text_loss_on_both_lines = True

    # Remove all negative/pair/contextual auxiliary supervision.
    P.num_negatives = 0
    P.use_local_hard_negatives = False
    P.local_hard_negative_weight = 0.0
    P.use_image_pair_contrastive = False
    P.image_pair_loss_weight = 0.0
    P.sequence_consistency_loss_weight = 0.0
    P.image_variance_loss_weight = 0.0
    P.real_filter_infeasible_span_dtw = False

    # Positive-only weak letter grounding. Alignment training uses a curriculum:
    # high gamma first (many plausible paths receive gradient), then lower gamma.
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
    # A horizontal transition advances letters without advancing image windows.
    # The old 0.02 symmetric penalty allowed one window to absorb many letters.
    P.positive_letter_dtw_horizontal_penalty = _env_float(
        "POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY", 0.30
    )
    P.positive_letter_dtw_step_penalty = P.positive_letter_dtw_vertical_penalty
    P.positive_letter_dtw_position_prior = _env_float(
        "POSITIVE_LETTER_DTW_POSITION_PRIOR", 0.15
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
    P.positive_letter_dtw_min_ink = _env_float("POSITIVE_LETTER_DTW_MIN_INK", 0.01)

    if P.restoration_training_stage == "pretrain":
        P.positive_letter_dtw_weight = 0.0
        default_restoration_weight = 1.0
    else:
        P.positive_letter_dtw_weight = _env_float(
            "POSITIVE_LETTER_DTW_WEIGHT", 1.0
        )
        default_restoration_weight = 0.05

    # Stroke restoration is pretraining supervision, then a small anti-forgetting
    # regularizer during DTW alignment.
    P.restoration_weight = _env_float(
        "RESTORATION_WEIGHT", default_restoration_weight
    )
    P.restoration_pixel_weight = _env_float("RESTORATION_PIXEL_WEIGHT", 1.0)
    P.restoration_edge_weight = _env_float("RESTORATION_EDGE_WEIGHT", 0.50)
    P.restoration_foreground_weight = _env_float(
        "RESTORATION_FOREGROUND_WEIGHT", 2.0
    )
    P.restoration_contrast_scale = _env_float("RESTORATION_CONTRAST_SCALE", 0.15)
    P.restoration_decoder_channels = _env_int("RESTORATION_DECODER_CHANNELS", 128)


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


class StrokeRestorationDecoder(nn.Module):
    """Decode one primitive window token into a 1x128x32 soft stroke map."""

    def __init__(self, dim: int, output_height: int, output_width: int, channels: int = 64):
        super().__init__()
        if output_height % 8 or output_width % 8:
            raise ValueError("Restoration decoder expects H and W divisible by 8")
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.seed_height = self.output_height // 8
        self.seed_width = self.output_width // 8
        channels = max(16, int(channels))
        self.channels = channels

        self.project = nn.Linear(int(dim), channels * self.seed_height * self.seed_width)
        c2 = max(16, channels // 2)
        c3 = max(8, channels // 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, c2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c3, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("Restoration decoder expects [B,T,D] tokens")
        batch, count, _ = tokens.shape
        x = self.project(tokens)
        x = x.reshape(batch * count, self.channels, self.seed_height, self.seed_width)
        x = self.decoder(x)
        x = x.reshape(batch, count, 1, self.output_height, self.output_width)
        return x


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


def stroke_restoration_loss(P, prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.float()
    target = target.float()
    foreground_weight = max(float(P.restoration_foreground_weight), 0.0)
    weights = 1.0 + foreground_weight * target
    pixel = (weights * (prediction - target).abs()).sum() / weights.sum().clamp_min(1.0)

    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    edge = 0.5 * (
        (pred_dx - target_dx).abs().mean()
        + (pred_dy - target_dy).abs().mean()
    )

    total = float(P.restoration_pixel_weight) * pixel + float(P.restoration_edge_weight) * edge
    return total, pixel, edge


def attach_restoration_dtw_stages(model, P):
    """Install primitive->semantic and primitive->restoration branches."""
    if getattr(model, "_restoration_positive_dtw_installed", False):
        return model
    if not hasattr(model, "vit_encoder"):
        raise TypeError("Restoration-DTW branch expects the shared ViT window container")

    from embeddingModel import (
        binarize_three_channel_input,
        sliding_window,
        window_ink_ratio_from_patches,
    )

    vit = model.vit_encoder
    dim = int(vit.embed_dim)
    device = vit.patch_embedding.weight.device
    dtype = vit.patch_embedding.weight.dtype

    semantic_mode = str(
        getattr(P, "restoration_semantic_adapter", "residual_mlp")
    ).strip().lower()
    if semantic_mode == "identity":
        vit.semantic_adapter = IdentitySemanticAdapter().to(device=device)
    elif semantic_mode == "residual_mlp":
        vit.semantic_adapter = ResidualSemanticAdapter(dim).to(
            device=device, dtype=dtype
        )
    else:
        raise ValueError(
            f"Unknown restoration semantic adapter mode: {semantic_mode!r}"
        )
    vit.restoration_semantic_adapter = semantic_mode

    vit.stroke_decoder = StrokeRestorationDecoder(
        dim,
        output_height=int(vit.input_height),
        output_width=int(vit.window_size),
        channels=int(P.restoration_decoder_channels),
    ).to(device=device, dtype=dtype)

    # The global Transformer and position embeddings are deliberately absent from
    # this experiment's computation graph. Freeze them so DDP never expects grads.
    for parameter in vit.encoder.parameters():
        parameter.requires_grad_(False)
    vit.position_embedding.requires_grad_(False)

    def minimal_window_forward(self, image, *, use_flip, return_model_input=False):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected image [B,3,H,W]")
        if int(image.shape[2]) != self.input_height:
            raise ValueError("Unexpected input height")
        if int(image.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than the window width")

        model_input = binarize_three_channel_input(image) if self.binarize_input else image
        tokens = self.patch_embedding(model_input)
        if tokens.shape[2] != 1:
            raise RuntimeError("Full-height patch projection must produce one token row")
        primitive = tokens.squeeze(2).transpose(1, 2).contiguous()
        if use_flip:
            primitive = torch.flip(primitive, dims=[1])
        primitive = self.local_norm(primitive)
        semantic = self.semantic_adapter(primitive)
        if return_model_input:
            return semantic, primitive, model_input
        return semantic, primitive

    vit.forward = MethodType(minimal_window_forward, vit)

    original_forward = model.forward

    def model_forward(
        self,
        image,
        show_dims=False,
        return_local=False,
        return_ink=False,
        return_grouped=False,
        return_training_bundle=False,
    ):
        if not return_training_bundle:
            return original_forward(
                image,
                show_dims=show_dims,
                return_local=return_local,
                return_ink=return_ink,
                return_grouped=return_grouped,
            )

        semantic, primitive, model_input = self.vit_encoder(
            image,
            use_flip=self.use_flip,
            return_model_input=True,
        )
        patches = sliding_window(model_input, self.window_size, self.stride)
        if self.use_flip:
            patches = torch.flip(patches, dims=[1])
        ink = window_ink_ratio_from_patches(patches)
        if int(ink.shape[1]) != int(primitive.shape[1]):
            raise RuntimeError("Token/window count mismatch in restoration branch")

        restoration = self.vit_encoder.stroke_decoder(primitive)
        target = _soft_stroke_target_from_normalized_patches(
            patches, float(P.restoration_contrast_scale)
        ).to(device=restoration.device)

        return {
            "semantic": self.vision_norm(semantic),
            "primitive": self.vision_norm(primitive),
            "primitive_raw": primitive,
            "ink": ink,
            "restoration": restoration,
            "restoration_target": target,
        }

    model.forward = MethodType(model_forward, model)
    model._restoration_positive_dtw_installed = True
    return model


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
            mask = ink_ratios[sample_index].to(visual.device).float() >= float(
                P.positive_letter_dtw_min_ink
            )
            if bool(mask.any()):
                visual = visual[mask]
        if visual.shape[0] == 0:
            continue

        visual_norm = F.normalize(visual.float(), p=2, dim=-1)
        with torch.no_grad():
            target = text_encoder("".join(letters)).detach().to(visual.device)
            target = F.normalize(target.float(), p=2, dim=-1)

        if str(P.positive_letter_dtw_cost_mode) == "full_alphabet_nll":
            # No negative transcripts are introduced. Instead, every window must
            # compete against all Arabic character identities before DTW selects
            # a monotonic route through the true transcript.
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
            costs = nll.index_select(1, target_indices)
        else:
            costs = 1.0 - torch.matmul(visual_norm, target.T)

        cost = _soft_dtw_cost_matrix(
            costs,
            gamma=float(P.positive_letter_dtw_gamma),
            vertical_penalty=float(P.positive_letter_dtw_vertical_penalty),
            horizontal_penalty=float(P.positive_letter_dtw_horizontal_penalty),
            position_prior_weight=float(P.positive_letter_dtw_position_prior),
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


def freeze_text_encoder(text_encoder):
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_encoder.eval()
    return text_encoder


def install_training_objective(train_module):
    """Replace the shared contrastive objective with the two-loss experiment."""
    from textEmbedding import OrthogonalCharEmbedding

    def build_text_encoder():
        encoder = OrthogonalCharEmbedding(
            embedding_dim=int(train_module.P.vector_size),
            vocab_size=int(train_module.P.letter_codebook_vocab_size),
            seed=int(train_module.P.letter_codebook_seed),
        ).to(train_module.P.device)
        return freeze_text_encoder(encoder)

    train_module.build_text_encoder = build_text_encoder

    # Old checkpoints may contain a different text model. Do not load it into the
    # fixed codebook; only visual weights are relevant for this branch.
    if hasattr(train_module, "_load_initial_states"):
        def load_initial_states(args, model, text_encoder):
            if args.resume:
                raise ValueError("Resume from old objective is unsupported; start a new run")
            if not args.pretrained_weights:
                return None
            loaded = torch.load(args.pretrained_weights, map_location=train_module.P.device)
            state = train_module.extract_model_state(loaded)
            incompatible = model.load_state_dict(state, strict=False)
            if train_module.CTX.is_main:
                print(
                    "Loaded visual initialization with strict=False: "
                    f"missing={len(incompatible.missing_keys)} "
                    f"unexpected={len(incompatible.unexpected_keys)}",
                    flush=True,
                )
            freeze_text_encoder(text_encoder)
            return None

        train_module._load_initial_states = load_initial_states

    def single_line_loss(image_embedder, text_encoder, images, texts):
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
        restoration, pixel, edge = stroke_restoration_loss(
            train_module.P,
            bundle["restoration"],
            bundle["restoration_target"],
        )
        total = (
            float(train_module.P.positive_letter_dtw_weight) * dtw
            + float(train_module.P.restoration_weight) * restoration
        )
        stats = {
            **dtw_stats,
            "restoration_loss": float(restoration.detach().item()),
            "restoration_pixel": float(pixel.detach().item()),
            "restoration_edge": float(edge.detach().item()),
            "norm_pos": float(dtw.detach().item()),
            "norm_neg": float("nan"),
            "cost_pos": float(dtw.detach().item()),
            "cost_neg": float("nan"),
            "gap": float("nan"),
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
                image_embedder, text_encoder, images1, batch["texts1"]
            )
            loss2, stats2 = single_line_loss(
                image_embedder, text_encoder, images2, batch["texts2"]
            )
            loss = 0.5 * (loss1 + loss2)
            stats = train_module.average_stats([stats1, stats2])
            stats["independent_lines_per_pair"] = 2.0
            stats["total"] = float(loss.detach().item())
            return loss, stats

        images, texts, _ignored_negatives = batch
        images = images.to(train_module.P.device, non_blocking=True)
        loss, stats = single_line_loss(image_embedder, text_encoder, images, texts)
        stats["independent_lines_per_pair"] = 1.0
        return loss, stats

    train_module.compute_batch_loss = compute_batch_loss

    # Fixed validation-line diagnostic after every epoch. This is analysis-only:
    # it never participates in the optimizer update.
    probe_state = {
        "patch_weight": None,
        "matrix": None,
        "path": None,
    }

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
        enabled = os.environ.get("RESTORATION_EPOCH_PROBE", "1").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return
        from pathlib import Path
        from restoration_epoch_probe import run_epoch_probe

        result = run_epoch_probe(
            model=model,
            text_encoder=text_encoder,
            valid_loader=valid_loader,
            epoch=int(epoch),
            job_id=str(job_id),
            config=config,
            weights_root=Path(train_module.__file__).resolve().parent / "Weights",
            device=device,
            previous_patch_weight=probe_state["patch_weight"],
            previous_matrix=probe_state["matrix"],
            previous_path=probe_state["path"],
        )
        probe_state["patch_weight"] = result["patch_weight"]
        probe_state["matrix"] = result["matrix"]
        probe_state["path"] = result["path"]

    train_module.epoch_diagnostic_hook = epoch_diagnostic_hook

    # The historical visualization assumes contextual Span-DTW/negative losses.
    # Disable it rather than silently plotting a different objective.
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
                        "minimal/restoration": float(
                            train_stats.get("restoration_loss", 0.0)
                        ),
                        "minimal/restoration_pixel": float(
                            train_stats.get("restoration_pixel", 0.0)
                        ),
                        "minimal/restoration_edge": float(
                            train_stats.get("restoration_edge", 0.0)
                        ),
                        "minimal/dtw_windows": float(train_stats.get("dtw_windows", 0.0)),
                        "minimal/dtw_letters": float(train_stats.get("dtw_letters", 0.0)),
                    },
                    step=int(epoch),
                    commit=False,
                )
            return original_logger(run, epoch, train_loss, val_loss, train_stats)

        train_module.wandb_log_epoch_metrics = wandb_log_epoch_metrics


def model_config(P):
    return {
        "architecture_family": "restoration-positive-dtw-window-encoder",
        "training_supervision": "positive-letter-dtw + stroke-restoration only",
        "primary_representation": (
            "primitive-direct-letter-aligned-window"
            if str(P.restoration_semantic_adapter) == "identity"
            else "semantic-letter-aligned-window"
        ),
        "local_representation": "primitive-stroke-window",
        "restoration_semantic_adapter": str(P.restoration_semantic_adapter),
        "semantic_projection_trainable": bool(
            str(P.restoration_semantic_adapter) == "residual_mlp"
        ),
        "dtw_supervises_primitive_directly": bool(
            str(P.restoration_semantic_adapter) == "identity"
        ),
        "context_transformer_active": False,
        "negative_transcripts": 0,
        "image_pair_supervision": False,
        "text_encoder_type": "char",
        "letter_codebook": "frozen-orthogonal-character-identities",
        "letter_codebook_seed": int(P.letter_codebook_seed),
        "letter_codebook_vocab_size": int(P.letter_codebook_vocab_size),
        "letter_inventory": str(P.letter_inventory),
        "positive_letter_dtw_weight": float(P.positive_letter_dtw_weight),
        "positive_letter_dtw_gamma": float(P.positive_letter_dtw_gamma),
        "positive_letter_dtw_step_penalty": float(P.positive_letter_dtw_step_penalty),
        "positive_letter_dtw_min_ink": float(P.positive_letter_dtw_min_ink),
        "restoration_weight": float(P.restoration_weight),
        "restoration_pixel_weight": float(P.restoration_pixel_weight),
        "restoration_edge_weight": float(P.restoration_edge_weight),
        "restoration_foreground_weight": float(P.restoration_foreground_weight),
        "restoration_contrast_scale": float(P.restoration_contrast_scale),
        "restoration_decoder_channels": int(P.restoration_decoder_channels),
        "real_binarize": bool(P.real_binarize),
        "keep_paired_lines_for_independent_training": bool(
            P.keep_paired_lines_for_independent_training
        ),
    }

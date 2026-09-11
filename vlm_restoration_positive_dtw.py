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
    P.letter_codebook_seed = _env_int("LETTER_CODEBOOK_SEED", 1234)
    P.letter_codebook_vocab_size = _env_int("LETTER_CODEBOOK_VOCAB_SIZE", 4096)
    P.letter_inventory = DEFAULT_ARABIC_LETTERS

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

    # Positive-only weak letter grounding.
    P.positive_letter_dtw_weight = _env_float("POSITIVE_LETTER_DTW_WEIGHT", 1.0)
    P.positive_letter_dtw_gamma = _env_float("POSITIVE_LETTER_DTW_GAMMA", 0.05)
    P.positive_letter_dtw_step_penalty = _env_float(
        "POSITIVE_LETTER_DTW_STEP_PENALTY", 0.02
    )
    P.positive_letter_dtw_min_ink = _env_float("POSITIVE_LETTER_DTW_MIN_INK", 0.01)

    # Stroke restoration is one conceptual loss with pixel and edge components.
    P.restoration_weight = _env_float("RESTORATION_WEIGHT", 0.10)
    P.restoration_pixel_weight = _env_float("RESTORATION_PIXEL_WEIGHT", 1.0)
    P.restoration_edge_weight = _env_float("RESTORATION_EDGE_WEIGHT", 0.50)
    P.restoration_foreground_weight = _env_float(
        "RESTORATION_FOREGROUND_WEIGHT", 2.0
    )
    P.restoration_contrast_scale = _env_float("RESTORATION_CONTRAST_SCALE", 0.15)
    P.restoration_decoder_channels = _env_int("RESTORATION_DECODER_CHANNELS", 64)


def _clean_letters(text: str, inventory_set: set[str]) -> list[str]:
    letters = []
    for character in unicodedata.normalize("NFC", str(text)):
        if character.isspace() or character == "ـ":
            continue
        if unicodedata.category(character) in {"Mn", "Me", "Cf"}:
            continue
        if character in inventory_set:
            letters.append(character)
    return letters


def _softmin(values: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = max(float(gamma), 1e-5)
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def positive_monotonic_letter_dtw_cost(
    visual_tokens: torch.Tensor,
    target_prototypes: torch.Tensor,
    *,
    gamma: float,
    step_penalty: float,
) -> torch.Tensor:
    """Positive-only differentiable DTW between windows and transcript letters.

    diagonal: advance window and letter
    vertical: next window still depicts the same letter
    horizontal: same overlapping window can depict the next adjacent letter
    """
    if visual_tokens.ndim != 2 or target_prototypes.ndim != 2:
        raise ValueError("DTW expects [T,D] visual and [L,D] letter tensors")
    T, L = int(visual_tokens.shape[0]), int(target_prototypes.shape[0])
    if T <= 0 or L <= 0:
        return visual_tokens.sum() * 0.0

    visual = F.normalize(visual_tokens.float(), p=2, dim=-1)
    target = F.normalize(target_prototypes.float(), p=2, dim=-1)
    costs = 1.0 - torch.matmul(visual, target.T)

    large = costs.new_tensor(1e4)
    previous = [costs.new_tensor(0.0)] + [large for _ in range(L)]
    penalty = float(step_penalty)

    for i in range(T):
        current = [large]
        for j in range(L):
            predecessors = torch.stack(
                [
                    previous[j],
                    previous[j + 1] + penalty,
                    current[j] + penalty,
                ]
            )
            current.append(costs[i, j] + _softmin(predecessors, gamma))
        previous = current

    return previous[L] / float(max(1, T + L))


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

    vit.semantic_adapter = ResidualSemanticAdapter(dim).to(device=device, dtype=dtype)
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
    inventory = str(P.letter_inventory)
    inventory_set = set(inventory)
    losses = []
    windows_used = []
    letters_used = []

    for sample_index, text in enumerate(positive_texts):
        letters = _clean_letters(text, inventory_set)
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

        with torch.no_grad():
            target = text_encoder("".join(letters)).detach().to(visual.device)
        cost = positive_monotonic_letter_dtw_cost(
            visual,
            target,
            gamma=float(P.positive_letter_dtw_gamma),
            step_penalty=float(P.positive_letter_dtw_step_penalty),
        )
        losses.append(cost)
        windows_used.append(float(visual.shape[0]))
        letters_used.append(float(len(letters)))

    if not losses:
        zero = semantic_tokens.sum() * 0.0
        return zero, {"positive_letter_dtw": 0.0, "dtw_windows": 0.0, "dtw_letters": 0.0}

    loss = torch.stack(losses).mean()
    return loss, {
        "positive_letter_dtw": float(loss.detach().item()),
        "dtw_windows": sum(windows_used) / len(windows_used),
        "dtw_letters": sum(letters_used) / len(letters_used),
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
        "primary_representation": "semantic-letter-aligned-window",
        "local_representation": "primitive-stroke-window",
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

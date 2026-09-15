"""ResNet-18 local window encoder + ViT-Tiny context + letter-DTW.

Pipeline:
  crop outer margins -> proportional resize -> overlapping RGB windows ->
  shared ResNet-18 local encoder -> ViT-Tiny sequence context ->
  local/context fusion -> positive + negative transcript DTW.

There is intentionally no restoration decoder and no reconstruction loss.
Final evaluation remains image-only and uses the same fused image vectors.
"""
from __future__ import annotations

from types import MethodType
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from restoration_recommended_components import LocalContextFusion, line_padding_masks
from vlm_restoration_positive_dtw import (
    freeze_text_encoder,
    negative_letter_dtw_margin_loss,
    positive_letter_dtw_loss,
)


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
    """Configure ResNet-18 + ViT-Tiny + contrastive letter-DTW."""
    P.experiment_name = "resnet18_tinyvit_positive_dtw"

    P.epochs = _env_int("RESNET_TINYVIT_EPOCHS", int(P.epochs))
    P.finetune_epochs = P.epochs
    P.learning_rate = _env_float(
        "RESNET_TINYVIT_LEARNING_RATE", float(P.learning_rate)
    )
    P.finetune_learning_rate = P.learning_rate
    P.num_samples = _env_int("RESNET_TINYVIT_NUM_SAMPLES", int(P.num_samples))

    # Keep the previous crop/resize/window geometry.
    P.window_size = 32
    P.stride_ratio = _env_float("RESNET_TINYVIT_STRIDE_RATIO", 0.50)
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

    # Standard ViT-Tiny dimensions, applied to the ResNet window-token sequence.
    P.vector_size = _env_int("TINY_VIT_EMBED_DIM", 192)
    P.vit_layers = _env_int("TINY_VIT_LAYERS", 12)
    P.vit_heads = _env_int("TINY_VIT_HEADS", 3)
    P.vit_mlp_dim = _env_int("TINY_VIT_MLP_DIM", 768)
    P.vit_dropout = _env_float("TINY_VIT_DROPOUT", 0.10)
    P.vit_max_tokens = max(int(getattr(P, "vit_max_tokens", 256)), 256)
    P.resnet18_imagenet_pretrained = _env_flag(
        "RESNET18_IMAGENET_PRETRAINED", False
    )
    P.local_context_fusion = "concat_projection_norm"

    # Fixed character identity codebook; text side never trains.
    P.text_encoder_type = "char"
    P.max_text_span_chars = 1
    P.max_text_token_chars = 1
    P.letter_codebook_seed = _env_int("LETTER_CODEBOOK_SEED", 1234)
    P.letter_codebook_vocab_size = _env_int("LETTER_CODEBOOK_VOCAB_SIZE", 4096)
    P.letter_inventory = "unicode-arabic-letters-after-nfkc"
    P.span_dtw_backend = "torch"

    P.keep_paired_lines_for_independent_training = True
    P.image_text_loss_on_both_lines = True
    P.num_negatives = _env_int("RESNET_TINYVIT_NUM_NEGATIVES", 10)
    P.contrastive_dtw_weight = _env_float(
        "RESNET_TINYVIT_CONTRASTIVE_WEIGHT", 0.50
    )
    P.contrastive_dtw_margin = _env_float(
        "RESNET_TINYVIT_CONTRASTIVE_MARGIN", 0.20
    )

    # Compatibility aliases consumed by the existing DTW helper.
    P.restoration_training_stage = "disabled"
    P.restoration_weight = 0.0
    P.restoration_contrastive_weight = P.contrastive_dtw_weight
    P.restoration_contrastive_margin = P.contrastive_dtw_margin

    P.use_local_hard_negatives = False
    P.local_hard_negative_weight = 0.0
    P.use_image_pair_contrastive = False
    P.image_pair_loss_weight = 0.0
    P.sequence_consistency_loss_weight = 0.0
    P.image_variance_loss_weight = 0.0
    P.real_filter_infeasible_span_dtw = False

    P.positive_letter_dtw_weight = _env_float("POSITIVE_LETTER_DTW_WEIGHT", 1.0)
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


class WindowSequenceResNet18Encoder(nn.Module):
    """Apply the same ResNet-18 to every overlapping 128x32 RGB window."""

    def __init__(
        self,
        *,
        input_height: int,
        window_size: int,
        stride: int,
        embed_dim: int,
        imagenet_pretrained: bool = False,
    ):
        super().__init__()
        if int(input_height) != 128 or int(window_size) != 32:
            raise ValueError("ResNet18 window encoder expects 128x32 windows")
        self.input_height = int(input_height)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embed_dim = int(embed_dim)

        weights = ResNet18_Weights.DEFAULT if bool(imagenet_pretrained) else None
        backbone = resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def extract_windows(self, line: torch.Tensor) -> torch.Tensor:
        if line.ndim != 4 or int(line.shape[1]) != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(line.shape)}")
        if int(line.shape[2]) != self.input_height:
            raise ValueError(
                f"Expected line height {self.input_height}, got {line.shape[2]}"
            )
        if int(line.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than the window width")
        patches = line.unfold(3, self.window_size, self.stride)
        return patches.permute(0, 3, 1, 2, 4).contiguous()

    def forward(self, line: torch.Tensor) -> torch.Tensor:
        patches = self.extract_windows(line)
        batch, count, channels, height, width = patches.shape
        flat = patches.reshape(batch * count, channels, height, width)
        features = self.backbone(flat)
        tokens = self.projection(features).reshape(batch, count, self.embed_dim)
        # Match the historical patch_embedding contract: [B,D,1,T].
        return tokens.transpose(1, 2).unsqueeze(2).contiguous()


def attach_resnet18_tinyvit_stages(model, P):
    """Replace the local encoder with ResNet-18 and keep ViT-Tiny context."""
    if getattr(model, "_resnet18_tinyvit_dtw_installed", False):
        return model
    if not hasattr(model, "vit_encoder"):
        raise TypeError("ResNet18+TinyViT branch expects the shared ViT container")

    vit = model.vit_encoder
    dim = int(vit.embed_dim)
    reference_parameter = next(vit.patch_embedding.parameters())
    device = reference_parameter.device
    dtype = reference_parameter.dtype

    vit.patch_embedding = WindowSequenceResNet18Encoder(
        input_height=int(vit.input_height),
        window_size=int(vit.window_size),
        stride=int(vit.stride),
        embed_dim=dim,
        imagenet_pretrained=bool(P.resnet18_imagenet_pretrained),
    ).to(device=device, dtype=dtype)
    vit.fusion_head = LocalContextFusion(dim).to(device=device, dtype=dtype)

    def encode_sequence(self, image, *, use_flip):
        if image.ndim != 4 or int(image.shape[1]) != 3:
            raise ValueError("Expected image [B,3,H,W]")
        if int(image.shape[2]) != self.input_height:
            raise ValueError("Unexpected input height")
        if int(image.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than the window width")

        model_input = image
        tokens = self.patch_embedding(model_input)
        if int(tokens.shape[2]) != 1:
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
        return fused, local, contextual, token_valid

    vit.encode_resnet18_tinyvit_sequence = MethodType(encode_sequence, vit)

    def model_forward(
        self,
        image,
        show_dims=False,
        return_local=False,
        return_ink=False,
        return_grouped=False,
        return_training_bundle=False,
    ):
        fused, local, contextual, token_valid = (
            self.vit_encoder.encode_resnet18_tinyvit_sequence(
                image, use_flip=self.use_flip
            )
        )
        fused_out = F.normalize(
            self.vision_norm(fused).float(), p=2, dim=-1
        ).to(dtype=fused.dtype)
        local_out = self.vision_norm(local)
        contextual_out = self.vision_norm(contextual)

        if return_training_bundle:
            return {
                "semantic": fused_out,
                "fused": fused_out,
                "primitive": local_out,
                "primitive_raw": local,
                "contextual": contextual_out,
                "ink": token_valid.float(),
                "token_valid": token_valid,
            }

        if show_dims:
            print(
                "image embeddings: resnet18 + vit-tiny "
                f"fused={tuple(fused_out.shape)} local={tuple(local_out.shape)} "
                f"context={tuple(contextual_out.shape)}",
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

    model.forward = MethodType(model_forward, model)
    model._resnet18_tinyvit_dtw_installed = True
    return model


def install_training_objective(train_module):
    """Use only positive DTW and negative-transcript contrastive DTW."""
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
        original_load_initial_states = train_module._load_initial_states

        def load_initial_states(args, model, text_encoder):
            if args.resume:
                return original_load_initial_states(args, model, text_encoder)
            if not args.pretrained_weights:
                return None

            loaded = torch.load(
                args.pretrained_weights, map_location=train_module.P.device
            )
            state = train_module.extract_model_state(loaded)
            current = model.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key in current and tuple(current[key].shape) == tuple(value.shape)
            }
            incompatible = model.load_state_dict(compatible, strict=False)
            skipped = len(state) - len(compatible)
            if train_module.CTX.is_main:
                print(
                    "Loaded compatible visual initialization: "
                    f"loaded={len(compatible)} skipped_shape_or_name={skipped} "
                    f"missing={len(incompatible.missing_keys)}",
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
            + float(train_module.P.contrastive_dtw_weight) * contrastive
        )
        stats = {
            **dtw_stats,
            **contrastive_stats,
            "norm_pos": float(dtw.detach().item()),
            "norm_neg": float(contrastive_stats["negative_letter_dtw"]),
            "cost_pos": float(dtw.detach().item()),
            "cost_neg": float(contrastive_stats["negative_letter_dtw"]),
            "gap": float(
                contrastive_stats["negative_letter_dtw"]
                - float(dtw.detach().item())
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
        anneal_epochs = max(
            1, int(train_module.P.positive_letter_dtw_anneal_epochs)
        )
        progress = min(
            1.0,
            max(0.0, (int(epoch) - 1) / max(1, anneal_epochs - 1)),
        )
        gamma = start * ((end / start) ** progress)
        train_module.P.positive_letter_dtw_gamma = float(gamma)
        if train_module.CTX.is_main:
            print(
                "[resnet18-tinyvit-dtw] "
                f"epoch={epoch} gamma={gamma:.5f} "
                f"v_penalty={train_module.P.positive_letter_dtw_vertical_penalty:.3f} "
                f"h_penalty={train_module.P.positive_letter_dtw_horizontal_penalty:.3f} "
                f"position_prior={train_module.P.positive_letter_dtw_position_prior:.3f}",
                flush=True,
            )

    train_module.epoch_start_hook = epoch_start_hook

    # The historical visualization is tied to the old Span-DTW objective.
    train_module.save_d3tw_visualization = lambda *args, **kwargs: None


def model_config(P):
    return {
        "architecture_family": "resnet18-tinyvit-positive-dtw-window-encoder",
        "training_supervision": "positive+negative letter-dtw only",
        "local_encoder": "resnet18",
        "resnet18_imagenet_pretrained": bool(P.resnet18_imagenet_pretrained),
        "context_encoder": "vit-tiny-sequence",
        "vit_tiny_embed_dim": int(P.vector_size),
        "vit_tiny_layers": int(P.vit_layers),
        "vit_tiny_heads": int(P.vit_heads),
        "vit_tiny_mlp_dim": int(P.vit_mlp_dim),
        "primary_representation": "normalized-local-context-fusion",
        "fusion_mode": str(P.local_context_fusion),
        "decoder_active": False,
        "restoration_loss_active": False,
        "negative_transcripts": int(P.num_negatives),
        "contrastive_dtw_weight": float(P.contrastive_dtw_weight),
        "contrastive_dtw_margin": float(P.contrastive_dtw_margin),
        "text_encoder_type": "char",
        "letter_codebook": "frozen-orthogonal-character-identities",
        "letter_codebook_seed": int(P.letter_codebook_seed),
        "letter_codebook_vocab_size": int(P.letter_codebook_vocab_size),
        "letter_inventory": str(P.letter_inventory),
        "positive_letter_dtw_weight": float(P.positive_letter_dtw_weight),
        "positive_letter_dtw_gamma": float(P.positive_letter_dtw_gamma),
        "positive_letter_dtw_gamma_start": float(P.positive_letter_dtw_gamma_start),
        "positive_letter_dtw_gamma_end": float(P.positive_letter_dtw_gamma_end),
        "positive_letter_dtw_anneal_epochs": int(
            P.positive_letter_dtw_anneal_epochs
        ),
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
        "real_binarize": bool(P.real_binarize),
        "synthetic_binarize": False,
        "zero_shot_preprocess": True,
        "zero_shot_preserve_aspect": True,
        "zero_shot_foreground_crop": True,
        "keep_paired_lines_for_independent_training": bool(
            P.keep_paired_lines_for_independent_training
        ),
    }

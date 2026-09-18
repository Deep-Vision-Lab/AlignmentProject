"""Point-2 evaluation helpers for local/context/fused ablations.

This module reconstructs either restoration checkpoint architecture and exposes
four image-only representations from the SAME trained checkpoint:

* local: ResNet local vector L_t
* context: raw ViT contextual vector C_t
* fused: trained fusion F(L_t, C_t)
* fused_wrong_context: trained fusion F(L_t, C_pi(t)) with deterministic,
  DIFFERENT within-line context permutations for the two paired lines

The evaluator is image-to-image only; no text encoder is loaded.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from embeddingModel import EmbeddingModel
from Evaluation._eval_utils import (
    EvaluationModels,
    ImageFeatures,
    _model_state,
    _force_odd_head_mha_reference_path,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

POINT2_MODES = ("local", "context", "fused", "fused_wrong_context")


def _disable_incompatible_mha_fastpath() -> bool:
    """Disable PyTorch's native MHA inference fastpath when available.

    The Tiny ViT used by this project has three attention heads. PyTorch 2.0's
    native inference fastpath can raise ``Only support when num_heads is even``
    for odd-head MultiheadAttention. Training does not hit that path, but
    eval+inference_mode does. Disabling only this optimization keeps the exact
    learned attention computation while avoiding the incompatible kernel.
    """
    backend = getattr(torch.backends, "mha", None)
    setter = getattr(backend, "set_fastpath_enabled", None)
    if callable(setter):
        setter(False)
        return True
    return False


_MHA_FASTPATH_DISABLED = _disable_incompatible_mha_fastpath()


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _is_physical_window_checkpoint(config: dict) -> bool:
    return (
        str(config.get("context_input", ""))
        == "same_physical_rgb_128x32_window"
        or str(config.get("model_backend", ""))
        == "resnet18_physical_window_tinyvit_positive_dtw"
    )


def load_point2_visual_models(checkpoint, device="auto", expected_branch="auto"):
    """Reconstruct either Point-2 architecture exactly, then load checkpoint."""
    config = dict(checkpoint["model_config"])
    family = str(config.get("architecture_family", ""))
    if family != "restoration-positive-dtw-window-encoder":
        raise ValueError(
            "Point-2 evaluation requires a restoration-positive-dtw-window-encoder checkpoint"
        )
    if expected_branch not in {"auto", "restoration"}:
        raise ValueError(f"Requested {expected_branch}, but Point-2 checkpoint is restoration")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = EmbeddingModel(
        window_size=int(config.get("window_size", 32)),
        stride=int(config.get("stride", 16)),
        vector_size=int(config.get("vector_size", 192)),
        device=dev,
        use_flip=str(config.get("lang", "Arabic")).lower() == "arabic",
        input_height=int(config.get("vit_input_height", 128)),
        vit_layers=int(config.get("vit_layers", 12)),
        vit_heads=int(config.get("vit_heads", 3)),
        vit_mlp_dim=int(config.get("vit_mlp_dim", 768)),
        vit_dropout=float(config.get("vit_dropout", 0.0)),
        vit_max_tokens=int(config.get("vit_max_tokens", 256)),
        vit_position_base_tokens=int(config.get("vit_position_base_tokens", 63)),
        vit_binarize_input=_flag(config.get("vit_binarize_input", False)),
        vit_binarize_contrast_threshold=float(
            config.get("vit_binarize_contrast_threshold", 0.15)
        ),
    ).to(dev)

    attach_config = SimpleNamespace(
        restoration_decoder_channels=int(config.get("restoration_decoder_channels", 64)),
        restoration_contrast_scale=float(config.get("restoration_contrast_scale", 0.15)),
        restoration_semantic_adapter=str(
            config.get("restoration_semantic_adapter", "identity")
        ),
        restoration_local_encoder=str(
            config.get("restoration_local_encoder", "resnet18")
        ),
        resnet18_pretrained=False,
        tiny_vit_pretrained=False,
        tiny_vit_pretrained_model=str(
            config.get("tiny_vit_pretrained_model", "facebook/deit-tiny-patch16-224")
        ),
        pretrained_local_only=True,
    )

    if _is_physical_window_checkpoint(config):
        from physical_window_vit_branch import attach_physical_window_vit_stages

        model = attach_physical_window_vit_stages(model, attach_config)
    else:
        from vlm_restoration_positive_dtw import attach_restoration_dtw_stages

        model = attach_restoration_dtw_stages(model, attach_config)

    model.load_state_dict(_model_state(checkpoint), strict=True)
    model.eval()
    odd_head_mha_modules = _force_odd_head_mha_reference_path(model)
    if odd_head_mha_modules:
        print(
            "Point-2/6 attention compatibility: forced reference MHA path for "
            f"{len(odd_head_mha_modules)} odd-head module(s): "
            + ", ".join(odd_head_mha_modules),
            flush=True,
        )

    models = EvaluationModels(model, None, config, checkpoint, dev)
    models.pair_cross_attention = None
    return models


def _permuted_context(
    contextual: torch.Tensor,
    valid: torch.Tensor,
    *,
    side: int,
) -> torch.Tensor:
    """Move valid context tokens differently on line 1 and line 2.

    Using one identical roll on both manuscript lines could preserve an
    artificial correspondence. Side 1 therefore rolls by about half the valid
    line, while side 2 rolls in the opposite direction by about one third.
    Artificial padding is excluded from both permutations.
    """
    if int(side) not in {1, 2}:
        raise ValueError("side must be 1 or 2")

    permuted = contextual.clone()
    for sample_index in range(int(contextual.shape[0])):
        valid_indices = torch.where(valid[sample_index].bool())[0]
        count = int(valid_indices.numel())
        if count <= 1:
            continue
        shift = max(1, count // 2) if int(side) == 1 else -max(1, count // 3)
        source = torch.roll(valid_indices, shifts=shift, dims=0)
        permuted[sample_index, valid_indices] = contextual[sample_index, source]
    return permuted


def _encode_line(models, image_path, mode: str, *, side: int) -> ImageFeatures:
    mode = str(mode).strip().lower()
    if mode not in POINT2_MODES:
        raise ValueError(f"Unknown Point-2 representation {mode!r}")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        original_size = image.size
        # The image has already been prepared by yelda_geometry. Do not resize
        # it again here: tight evaluation intentionally keeps variable width.
        tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )(image).unsqueeze(0).to(models.device)

    image_model = models.image_model
    vit = image_model.vit_encoder
    encoder = getattr(vit, "encode_physical_window_sequence", None)
    if encoder is None:
        encoder = getattr(vit, "encode_restoration_sequence", None)
    if encoder is None:
        raise RuntimeError("Checkpoint model does not expose a restoration sequence encoder")

    # Normal path: the MHA fastpath has been disabled, so inference_mode keeps
    # evaluation cheap. Fallback: if this PyTorch build has no public MHA
    # fastpath switch, keep autograd enabled; that also prevents selection of
    # the incompatible native inference kernel while leaving model.eval() set.
    forward_context = torch.inference_mode if _MHA_FASTPATH_DISABLED else torch.enable_grad
    with forward_context():
        fused_raw, local_raw, context_raw, _model_input, token_valid = encoder(
            tensor, use_flip=image_model.use_flip
        )

        local = F.normalize(
            image_model.vision_norm(local_raw).float(), p=2, dim=-1
        )
        context = F.normalize(
            image_model.vision_norm(context_raw).float(), p=2, dim=-1
        )
        fused = F.normalize(
            image_model.vision_norm(fused_raw).float(), p=2, dim=-1
        )

        wrong_context_raw = _permuted_context(
            context_raw, token_valid, side=side
        )
        fused_wrong_raw = vit.fusion_head(local_raw, wrong_context_raw)
        fused_wrong = F.normalize(
            image_model.vision_norm(fused_wrong_raw).float(), p=2, dim=-1
        )

    selected = {
        "local": local,
        "context": context,
        "fused": fused,
        "fused_wrong_context": fused_wrong,
    }[mode][0].detach()

    local_out = local[0].detach()
    if not torch.isfinite(selected).all() or not torch.isfinite(local_out).all():
        raise ValueError("Non-finite Point-2 image embeddings")

    return ImageFeatures(
        contextual=selected,
        local=local_out,
        grouped=local_out,
        ink=token_valid[0].float().detach(),
        image_size=original_size,
    )


def point2_pair_features(models, image1, image2, mode: str):
    """Return pair-local features for one of the four Point-2 ablations."""
    first = _encode_line(models, image1, mode, side=1)
    second = _encode_line(models, image2, mode, side=2)
    return first, second

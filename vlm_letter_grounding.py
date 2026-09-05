"""Hierarchical VLM grounding for Arabic manuscript windows.

This branch changes the interpretation of a visual window from a generic free
embedding into a *depiction* vector: the local representation is explicitly
trained to explain which Arabic letters are visible in that window sequence.

Hierarchy
---------
1. pixels -> trainable full-height patch projection
2. local depiction head -> local depiction tokens
3. letter-level monotonic soft-DTW against single-letter text prototypes
4. 4-layer visual Transformer -> contextual tokens
5. existing contextual Span-DTW + image-image/order losses

The local DTW uses the transcript only; it does not require synthetic bounding
boxes. Horizontal and vertical DTW moves allow one 32px window to depict more
than one letter and allow one letter to occupy more than one overlapping window.
Whitespace is structural and is therefore excluded from the semantic letter
sequence. Low-ink windows are excluded from the local depiction objective.
"""
from __future__ import annotations

from types import MethodType
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F


# Broad Arabic inventory used by the manuscript/synthetic corpora.  Whitespace
# and diacritics are intentionally absent: they are structural/noise states for
# this local semantic objective rather than letter identities.
DEFAULT_ARABIC_LETTERS = "ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويىة"


def apply_branch_config(P) -> None:
    """Apply branch-only settings without changing the proven base Parameters.py."""
    P.experiment_name = "vit_vlm_letter_depiction"

    # Keep the requested 10k dataset cap and the proven 4-layer/raw-RGB visual
    # baseline.  These assignments are deliberate guards against stale env/config.
    P.num_samples = 10000
    P.vit_layers = 4
    P.vit_binarize_input = False
    P.max_text_span_chars = 3

    # Ten transcript negatives are generated.  Contextual Span-DTW evaluates all
    # candidates and uses the hardest negative for gradient, preserving the
    # stronger historical objective without rotating/subsampling negatives.
    P.num_negatives = 10
    P.span_dtw_active_negatives_per_sample = 0
    P.span_negative_grad_mode = "hardest"

    # Quality-first: no every-N-batch or max-sample skipping of auxiliary stages.
    P.local_hard_negative_every_n_batches = 1
    P.local_hard_negative_max_samples_per_batch = 0
    P.image_pair_every_n_batches = 1
    P.image_pair_max_samples_per_batch = 0
    P.image_text_loss_on_both_lines = True

    # Remove the only approximate text cache from this experimental branch.  The
    # AraBERT backbone is still frozen, but every requested surface is recomputed
    # rather than restored from a float16 cache.  This intentionally favors
    # experimental clarity/quality over speed.
    P.span_feature_cache_size = 0
    P.clear_span_cache_each_epoch = True

    # Local depiction objective.
    P.letter_depiction_enabled = True
    P.letter_depiction_weight = 0.35
    P.letter_depiction_positive_weight = 0.35
    P.letter_depiction_margin = 0.15
    P.letter_depiction_gamma = 0.05
    P.letter_depiction_step_penalty = 0.02
    P.letter_depiction_min_ink = 0.01
    P.letter_depiction_inventory = DEFAULT_ARABIC_LETTERS


def _clean_letters(text: str, inventory_set: set[str]) -> list[str]:
    result: list[str] = []
    for character in unicodedata.normalize("NFC", str(text)):
        if character.isspace() or character == "ـ":
            continue
        if unicodedata.category(character) in {"Mn", "Me", "Cf"}:
            continue
        if character in inventory_set:
            result.append(character)
    return result


def _project_letter_inventory(text_encoder, inventory: str) -> torch.Tensor:
    """Return trainable text-side prototypes [letters, D].

    AraBERT itself remains frozen; its projection/norm remain trainable.  Calling
    the encoder projection here therefore gives the local depiction stage a
    genuine vision-language target rather than a fixed integer classifier.
    """
    letters = list(inventory)
    if hasattr(text_encoder, "_project_surfaces"):
        prototypes = text_encoder._project_surfaces(letters)
    else:
        encodings = text_encoder.encode_many(letters)
        vectors = []
        for encoding in encodings:
            chosen = None
            for index, (length, is_blank) in enumerate(
                zip(encoding.lengths, encoding.is_blank or [False] * len(encoding.lengths))
            ):
                if int(length) == 1 and not bool(is_blank):
                    chosen = encoding.embeddings[index]
                    break
            if chosen is None:
                raise RuntimeError("Could not extract a single-letter text prototype")
            vectors.append(chosen)
        prototypes = torch.stack(vectors, dim=0)
    return F.normalize(prototypes.float(), p=2, dim=-1)


def _softmin(values: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = max(float(gamma), 1e-5)
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def monotonic_letter_dtw_cost(
    visual_tokens: torch.Tensor,
    target_prototypes: torch.Tensor,
    *,
    gamma: float,
    step_penalty: float,
) -> torch.Tensor:
    """Differentiable monotonic local alignment cost.

    visual_tokens: [T, D]
    target_prototypes: [L, D]

    Allowed moves:
      diagonal  -> advance visual window and letter
      vertical  -> another visual window depicts the same letter
      horizontal-> the same visual window depicts another adjacent letter

    The horizontal move is important for 32px windows because one window can
    visibly contain strokes from multiple neighboring Arabic letters.
    """
    if visual_tokens.ndim != 2 or target_prototypes.ndim != 2:
        raise ValueError("letter DTW expects [T,D] visual and [L,D] text tensors")
    T, L = int(visual_tokens.shape[0]), int(target_prototypes.shape[0])
    if T <= 0 or L <= 0:
        return visual_tokens.new_tensor(0.0)

    visual = F.normalize(visual_tokens.float(), p=2, dim=-1)
    text = F.normalize(target_prototypes.float(), p=2, dim=-1)
    costs = 1.0 - torch.matmul(visual, text.T)

    large = costs.new_tensor(1e4)
    zero = costs.new_tensor(0.0)
    previous = [zero] + [large for _ in range(L)]
    penalty = float(step_penalty)

    for i in range(T):
        current = [large]
        for j in range(L):
            predecessors = torch.stack(
                [
                    previous[j],              # diagonal
                    previous[j + 1] + penalty, # repeat letter over next window
                    current[j] + penalty,      # same window depicts next letter
                ]
            )
            current.append(costs[i, j] + _softmin(predecessors, gamma))
        previous = current

    # Length normalization makes transcript negatives of different lengths
    # comparable and prevents long strings from winning/losing only by length.
    return previous[L] / float(max(1, T + L))


def _target_prototypes(
    text: str,
    prototypes: torch.Tensor,
    mapping: dict[str, int],
    inventory_set: set[str],
) -> torch.Tensor | None:
    letters = _clean_letters(text, inventory_set)
    if not letters:
        return None
    indices = torch.tensor(
        [mapping[character] for character in letters],
        device=prototypes.device,
        dtype=torch.long,
    )
    return prototypes.index_select(0, indices)


def _select_ink_tokens(local_tokens: torch.Tensor, ink: torch.Tensor | None, min_ink: float):
    if ink is None:
        return local_tokens
    mask = ink.to(local_tokens.device).float() >= float(min_ink)
    if int(mask.sum().item()) < 1:
        return local_tokens
    return local_tokens[mask]


def letter_depiction_loss(
    P,
    text_encoder,
    local_tokens: torch.Tensor,
    ink_ratios: torch.Tensor | None,
    positive_texts,
    negative_texts,
):
    """Local visual-language loss with explicit positive and negative sequences."""
    inventory = str(P.letter_depiction_inventory)
    inventory_set = set(inventory)
    mapping = {character: index for index, character in enumerate(inventory)}
    prototypes = _project_letter_inventory(text_encoder, inventory)

    losses = []
    positive_costs = []
    negative_costs = []
    margins = []
    considered_negatives = []

    for sample_index, positive_text in enumerate(positive_texts):
        visual = _select_ink_tokens(
            local_tokens[sample_index],
            ink_ratios[sample_index] if ink_ratios is not None else None,
            P.letter_depiction_min_ink,
        )
        positive_target = _target_prototypes(
            positive_text, prototypes, mapping, inventory_set
        )
        if positive_target is None:
            continue

        positive_cost = monotonic_letter_dtw_cost(
            visual,
            positive_target,
            gamma=P.letter_depiction_gamma,
            step_penalty=P.letter_depiction_step_penalty,
        )

        sample_negatives = list(negative_texts[sample_index]) if negative_texts else []
        valid_negative_texts = []
        detached_costs = []
        # Every supplied negative is evaluated for hard-negative selection.  The
        # selected hardest sequence is then recomputed with gradient.
        with torch.no_grad():
            for negative_text in sample_negatives:
                negative_target = _target_prototypes(
                    negative_text,
                    prototypes.detach(),
                    mapping,
                    inventory_set,
                )
                if negative_target is None:
                    continue
                detached_costs.append(
                    monotonic_letter_dtw_cost(
                        visual.detach(),
                        negative_target,
                        gamma=P.letter_depiction_gamma,
                        step_penalty=P.letter_depiction_step_penalty,
                    )
                )
                valid_negative_texts.append(negative_text)

        if detached_costs:
            detached_stack = torch.stack(detached_costs)
            hardest_index = int(torch.argmin(detached_stack).item())
            hardest_target = _target_prototypes(
                valid_negative_texts[hardest_index],
                prototypes,
                mapping,
                inventory_set,
            )
            negative_cost = monotonic_letter_dtw_cost(
                visual,
                hardest_target,
                gamma=P.letter_depiction_gamma,
                step_penalty=P.letter_depiction_step_penalty,
            )
            contrastive = torch.relu(
                positive_cost - negative_cost + float(P.letter_depiction_margin)
            )
            sample_loss = (
                float(P.letter_depiction_positive_weight) * positive_cost
                + contrastive
            )
            negative_costs.append(float(negative_cost.detach().item()))
            margins.append(float((negative_cost - positive_cost).detach().item()))
            considered_negatives.append(float(len(valid_negative_texts)))
        else:
            sample_loss = float(P.letter_depiction_positive_weight) * positive_cost
            considered_negatives.append(0.0)

        losses.append(sample_loss)
        positive_costs.append(float(positive_cost.detach().item()))

    if not losses:
        zero = local_tokens.sum() * 0.0
        return zero, {
            "letter_depiction_loss": 0.0,
            "letter_pos_cost": 0.0,
            "letter_neg_cost": 0.0,
            "letter_gap": 0.0,
            "letter_negative_candidates": 0.0,
        }

    loss = torch.stack(losses).mean()
    return loss, {
        "letter_depiction_loss": float(loss.detach().item()),
        "letter_pos_cost": sum(positive_costs) / len(positive_costs),
        "letter_neg_cost": (
            sum(negative_costs) / len(negative_costs) if negative_costs else 0.0
        ),
        "letter_gap": sum(margins) / len(margins) if margins else 0.0,
        "letter_negative_candidates": (
            sum(considered_negatives) / len(considered_negatives)
            if considered_negatives
            else 0.0
        ),
    }


def attach_depiction_head(model):
    """Insert a trainable local depiction stage before the visual Transformer."""
    if getattr(model, "_vlm_depiction_installed", False):
        return model
    if not hasattr(model, "vit_encoder"):
        raise TypeError("letter depiction branch expects a ViT EmbeddingModel")

    from embeddingModel import binarize_three_channel_input

    vit = model.vit_encoder
    dim = int(vit.embed_dim)
    device = vit.patch_embedding.weight.device
    dtype = vit.patch_embedding.weight.dtype

    vit.depiction_projection = nn.Sequential(
        nn.Linear(dim, dim),
        nn.GELU(),
        nn.Linear(dim, dim),
    ).to(device=device, dtype=dtype)
    vit.depiction_norm = nn.LayerNorm(dim).to(device=device, dtype=dtype)

    def grounded_forward(
        self,
        image: torch.Tensor,
        *,
        use_flip: bool,
        return_model_input: bool = False,
    ):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "ViT input must have shape [B,3,H,W], "
                f"got {tuple(image.shape)}"
            )
        if int(image.shape[2]) != self.input_height:
            raise ValueError(
                f"ViT expects input height {self.input_height}, got {image.shape[2]}"
            )
        if int(image.shape[3]) < self.window_size:
            raise ValueError(
                f"Input width {image.shape[3]} is smaller than window size {self.window_size}"
            )

        model_input = (
            binarize_three_channel_input(image)
            if self.binarize_input
            else image
        )
        tokens = self.patch_embedding(model_input)
        if tokens.shape[2] != 1:
            raise RuntimeError(
                "Full-height patch embedding should produce one vertical token row, "
                f"got {tuple(tokens.shape)}"
            )
        tokens = tokens.squeeze(2).transpose(1, 2).contiguous()
        if use_flip:
            tokens = torch.flip(tokens, dims=[1])

        primitive = self.local_norm(tokens)
        # Residual depiction head: retain the primitive visual evidence while
        # forcing a learned semantic stage that receives direct letter-level loss.
        depiction = self.depiction_norm(
            primitive + self.depiction_projection(primitive)
        )
        contextual = depiction + self._position_tokens(depiction.shape[1]).to(
            dtype=depiction.dtype, device=depiction.device
        )
        contextual = self.encoder(self.input_dropout(contextual))

        if return_model_input:
            return contextual, depiction, model_input
        return contextual, depiction

    vit.forward = MethodType(grounded_forward, vit)
    model._vlm_depiction_installed = True
    return model


def install_training_objective(train_module) -> None:
    """Add local letter grounding without duplicating the visual forward pass."""
    original = train_module.compute_single_image_text_loss

    def compute_single_image_text_loss(
        image_embedder,
        text_encoder,
        criterion,
        images,
        pos_texts,
        neg_texts,
        embeddings=None,
        local_enabled=True,
    ):
        loss, stats, embeddings = original(
            image_embedder,
            text_encoder,
            criterion,
            images,
            pos_texts,
            neg_texts,
            embeddings,
            local_enabled,
        )
        if not bool(getattr(train_module.P, "letter_depiction_enabled", False)):
            return loss, stats, embeddings

        _contextual, local_tokens, ink_ratios, _raw_local = embeddings
        depiction_loss, depiction_stats = letter_depiction_loss(
            train_module.P,
            text_encoder,
            local_tokens,
            ink_ratios,
            pos_texts,
            neg_texts,
        )
        loss = loss + float(train_module.P.letter_depiction_weight) * depiction_loss
        stats.update(depiction_stats)
        stats["letter_depiction_weighted"] = float(
            (float(train_module.P.letter_depiction_weight) * depiction_loss)
            .detach()
            .item()
        )
        return loss, stats, embeddings

    train_module.compute_single_image_text_loss = compute_single_image_text_loss


def model_config(P) -> dict:
    return {
        "vlm_hierarchy": "local-letter-depiction -> contextual-span -> image-pair",
        "letter_depiction_enabled": bool(P.letter_depiction_enabled),
        "letter_depiction_weight": float(P.letter_depiction_weight),
        "letter_depiction_positive_weight": float(P.letter_depiction_positive_weight),
        "letter_depiction_margin": float(P.letter_depiction_margin),
        "letter_depiction_gamma": float(P.letter_depiction_gamma),
        "letter_depiction_step_penalty": float(P.letter_depiction_step_penalty),
        "letter_depiction_min_ink": float(P.letter_depiction_min_ink),
        "letter_depiction_inventory": str(P.letter_depiction_inventory),
        "local_negative_policy": "all supplied candidates -> hardest sequence",
        "text_surface_cache": "disabled",
    }

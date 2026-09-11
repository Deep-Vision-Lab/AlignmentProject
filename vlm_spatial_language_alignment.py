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


# ============================================================================
# CFM-inspired spatial-language architecture overrides
# ============================================================================
#
# This section intentionally reuses the well-tested Arabic letter cleaning and
# monotonic DTW utilities above, but changes the representation hierarchy:
#
# primitive spatial window
#   -> bounded semantic-affinity voting
#   -> image-side language adapter = LOCAL LANGUAGE TOKEN
#   -> independent self-context Transformer
#   -> gated residual fusion that preserves the local token
#
# The Arabic text encoder is a frozen semantic anchor. There is no pair
# cross-attention in this branch.

import math
import os


def _cfm_env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _cfm_env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _cfm_logit(probability):
    probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


def apply_branch_config(P):
    """Configure the CFM-inspired spatial language branch."""
    P.experiment_name = "vit_cfm_spatial_language"

    # Geometry/backbone: fair first comparison with the proven stride-16 ViT.
    P.vit_layers = 4
    P.vit_binarize_input = False
    P.window_size = 32
    P.stride_ratio = _cfm_env_float("CFM_STRIDE_RATIO", 0.50)
    P.max_text_span_chars = 3
    P.vit_max_tokens = max(int(getattr(P, "vit_max_tokens", 256)), 256)

    # Examine all ten transcript negatives and backprop through the hardest.
    P.num_negatives = 10
    P.span_dtw_active_negatives_per_sample = 0
    P.span_negative_grad_mode = "hardest"
    P.image_text_loss_on_both_lines = True

    # Replaced by the direct local language objectives below.
    P.use_local_hard_negatives = False

    # Keep independent image-image supervision. No cross-line attention.
    P.use_image_pair_contrastive = True
    P.image_pair_loss_weight = 0.40
    P.image_pair_every_n_batches = 1
    P.image_pair_max_samples_per_batch = 0
    P.sequence_consistency_loss_weight = 0.05

    # CFM/DINOiser-inspired local semantic voting.
    P.spatial_affinity_radius = _cfm_env_int("SPATIAL_AFFINITY_RADIUS", 3)
    P.spatial_affinity_temperature = _cfm_env_float(
        "SPATIAL_AFFINITY_TEMPERATURE", 0.15
    )
    P.spatial_affinity_distance_penalty = _cfm_env_float(
        "SPATIAL_AFFINITY_DISTANCE_PENALTY", 0.12
    )
    P.spatial_affinity_initial_gate = _cfm_env_float(
        "SPATIAL_AFFINITY_INITIAL_GATE", 0.15
    )

    # Transformer context can refine, but not overwrite, local semantics.
    P.context_residual_initial_gate = _cfm_env_float(
        "CONTEXT_RESIDUAL_INITIAL_GATE", 0.25
    )
    P.local_context_consistency_weight = _cfm_env_float(
        "LOCAL_CONTEXT_CONSISTENCY_WEIGHT", 0.05
    )

    # Local letter-level vision-language objective.
    P.spatial_letter_enabled = True
    P.spatial_letter_weight = _cfm_env_float("SPATIAL_LETTER_WEIGHT", 0.40)
    P.spatial_letter_positive_weight = _cfm_env_float(
        "SPATIAL_LETTER_POSITIVE_WEIGHT", 0.35
    )
    P.spatial_letter_margin = _cfm_env_float("SPATIAL_LETTER_MARGIN", 0.20)
    P.spatial_letter_gamma = _cfm_env_float("SPATIAL_LETTER_GAMMA", 0.05)
    P.spatial_letter_step_penalty = _cfm_env_float(
        "SPATIAL_LETTER_STEP_PENALTY", 0.02
    )
    P.spatial_letter_min_ink = _cfm_env_float("SPATIAL_LETTER_MIN_INK", 0.01)
    P.spatial_letter_inventory = DEFAULT_ARABIC_LETTERS

    # Segmentation-like dense vocabulary objective.
    P.dense_letter_weight = _cfm_env_float("DENSE_LETTER_WEIGHT", 0.15)
    P.dense_letter_temperature = _cfm_env_float(
        "DENSE_LETTER_TEMPERATURE", 0.07
    )

    # The whole language side is frozen, so frozen-surface caching is safe.
    P.span_feature_cache_size = max(
        int(getattr(P, "span_feature_cache_size", 8192)), 8192
    )
    P.clear_span_cache_each_epoch = False


def freeze_text_encoder(text_encoder):
    """Freeze backbone, projection, norm, space and blank embeddings."""
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_encoder.eval()
    return text_encoder


def maybe_load_text_anchor(text_encoder):
    """Load a previously learned Arabic projection before freezing it."""
    path = os.environ.get("TEXT_ANCHOR_WEIGHTS", "").strip()
    if not path:
        return "fresh-deterministic-text-anchor"
    if not os.path.isfile(path):
        raise FileNotFoundError("TEXT_ANCHOR_WEIGHTS does not exist: " + path)
    payload = torch.load(path, map_location=getattr(text_encoder, "device", "cpu"))
    state = None
    if isinstance(payload, dict):
        state = payload.get("text_encoder_state_dict")
        if state is None:
            state = payload.get("text_embedder_state_dict")
    if state is None:
        raise ValueError("No text encoder state found in " + path)
    text_encoder.load_state_dict(state, strict=False)
    return path


class SpatialAffinityLanguageAdapter(nn.Module):
    """Bounded neighbor voting followed by a residual language adapter."""

    def __init__(
        self,
        dim,
        radius=3,
        temperature=0.15,
        distance_penalty=0.12,
        initial_gate=0.15,
    ):
        super().__init__()
        self.dim = int(dim)
        self.radius = max(0, int(radius))
        self.temperature = max(float(temperature), 1e-4)
        self.distance_penalty = max(float(distance_penalty), 0.0)
        self.vote_gate_logit = nn.Parameter(
            torch.tensor(_cfm_logit(initial_gate), dtype=torch.float32)
        )
        self.vote_norm = nn.LayerNorm(self.dim)
        self.vote_adapter = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.language_adapter = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.language_norm = nn.LayerNorm(self.dim)

    @property
    def vote_gate(self):
        return torch.sigmoid(self.vote_gate_logit)

    def forward(self, primitive):
        if primitive.ndim != 3:
            raise ValueError(
                "Expected [B,T,D] primitive tokens, got " + str(tuple(primitive.shape))
            )
        _, token_count, dim = primitive.shape
        if int(dim) != self.dim:
            raise ValueError(
                "Expected embedding dim %d, got %d" % (self.dim, int(dim))
            )

        normalized = self.vote_norm(primitive)
        unit = F.normalize(normalized.float(), p=2, dim=-1)
        scores = torch.matmul(unit, unit.transpose(1, 2)) / self.temperature

        positions = torch.arange(token_count, device=primitive.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        if self.radius > 0:
            allowed = distance <= self.radius
        else:
            allowed = torch.eye(
                token_count, device=primitive.device, dtype=torch.bool
            )

        scores = scores - self.distance_penalty * distance.to(scores.dtype)
        scores = scores.masked_fill(
            ~allowed.unsqueeze(0), torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(scores, dim=-1).to(dtype=primitive.dtype)

        # Nearby semantically similar windows vote, but the residual gate starts
        # small so a window never loses its own physical evidence.
        voted = torch.matmul(weights, primitive)
        semantic = primitive + self.vote_gate.to(
            dtype=primitive.dtype
        ) * self.vote_adapter(voted)

        # This output itself, not a classifier head, is pulled into text space.
        local_language = self.language_norm(
            semantic + self.language_adapter(semantic)
        )
        return local_language, weights


class LocalPreservingContextFusion(nn.Module):
    """Interpolate from L_i toward C_i instead of replacing L_i."""

    def __init__(self, dim, initial_gate=0.25):
        super().__init__()
        self.gate_logit = nn.Parameter(
            torch.tensor(_cfm_logit(initial_gate), dtype=torch.float32)
        )
        self.norm = nn.LayerNorm(int(dim))

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, local_language, contextual):
        gate = self.gate.to(dtype=local_language.dtype)
        return self.norm(
            local_language + gate * (contextual - local_language)
        )


def attach_spatial_language_stages(model, P):
    """Install spatial voting + language adapter + gated context on the ViT."""
    if getattr(model, "_cfm_spatial_language_installed", False):
        return model
    if not hasattr(model, "vit_encoder"):
        raise TypeError("CFM spatial-language branch expects a ViT EmbeddingModel")

    from embeddingModel import binarize_three_channel_input

    vit = model.vit_encoder
    dim = int(vit.embed_dim)
    device = vit.patch_embedding.weight.device
    dtype = vit.patch_embedding.weight.dtype

    vit.spatial_language_adapter = SpatialAffinityLanguageAdapter(
        dim,
        radius=int(P.spatial_affinity_radius),
        temperature=float(P.spatial_affinity_temperature),
        distance_penalty=float(P.spatial_affinity_distance_penalty),
        initial_gate=float(P.spatial_affinity_initial_gate),
    ).to(device=device, dtype=dtype)
    vit.context_fusion = LocalPreservingContextFusion(
        dim, initial_gate=float(P.context_residual_initial_gate)
    ).to(device=device, dtype=dtype)

    def spatial_forward(
        self, image, *, use_flip, return_model_input=False
    ):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "ViT input must have shape [B,3,H,W], got "
                + str(tuple(image.shape))
            )
        if int(image.shape[2]) != self.input_height:
            raise ValueError(
                "ViT expects input height %d, got %d"
                % (self.input_height, int(image.shape[2]))
            )
        if int(image.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than window size")

        model_input = (
            binarize_three_channel_input(image)
            if self.binarize_input
            else image
        )
        tokens = self.patch_embedding(model_input)
        if tokens.shape[2] != 1:
            raise RuntimeError(
                "Full-height patch embedding must produce one vertical token row"
            )
        primitive = tokens.squeeze(2).transpose(1, 2).contiguous()
        if use_flip:
            primitive = torch.flip(primitive, dims=[1])
        primitive = self.local_norm(primitive)

        local_language, _affinity = self.spatial_language_adapter(primitive)
        positional = self._position_tokens(local_language.shape[1]).to(
            dtype=local_language.dtype, device=local_language.device
        )
        context_input = local_language + positional
        transformer_context = self.encoder(self.input_dropout(context_input))
        fused_context = self.context_fusion(
            local_language, transformer_context
        )

        if return_model_input:
            return fused_context, local_language, model_input
        return fused_context, local_language

    vit.forward = MethodType(spatial_forward, vit)
    model._cfm_spatial_language_installed = True
    return model


@torch.no_grad()
def _cfm_hard_monotonic_path(
    visual_tokens, target_prototypes, step_penalty
):
    visual = F.normalize(visual_tokens.float(), p=2, dim=-1)
    text = F.normalize(target_prototypes.float(), p=2, dim=-1)
    costs = 1.0 - torch.matmul(visual, text.T)
    token_count, letter_count = costs.shape
    if token_count == 0 or letter_count == 0:
        return []

    dp = torch.full_like(costs, float("inf"))
    back = torch.zeros_like(costs, dtype=torch.int8)
    dp[0, 0] = costs[0, 0]

    for i in range(token_count):
        for j in range(letter_count):
            if i == 0 and j == 0:
                continue
            candidates = []
            moves = []
            if i > 0 and j > 0:
                candidates.append(dp[i - 1, j - 1])
                moves.append(0)
            if i > 0:
                candidates.append(dp[i - 1, j] + float(step_penalty))
                moves.append(1)
            if j > 0:
                candidates.append(dp[i, j - 1] + float(step_penalty))
                moves.append(2)
            stacked = torch.stack(candidates)
            choice = int(torch.argmin(stacked).item())
            dp[i, j] = costs[i, j] + stacked[choice]
            back[i, j] = moves[choice]

    i = token_count - 1
    j = letter_count - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        move = int(back[i, j].item())
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    path.reverse()
    return path


def dense_letter_inventory_loss(
    P, text_encoder, local_tokens, ink_ratios, positive_texts
):
    """Dense multi-positive InfoNCE over the complete Arabic letter inventory."""
    inventory = str(P.spatial_letter_inventory)
    inventory_set = set(inventory)
    mapping = {character: index for index, character in enumerate(inventory)}
    with torch.no_grad():
        prototypes = _project_letter_inventory(
            text_encoder, inventory
        ).detach().to(local_tokens.device)

    sample_losses = []
    for sample_index, positive_text in enumerate(positive_texts):
        visual = _select_ink_tokens(
            local_tokens[sample_index],
            ink_ratios[sample_index] if ink_ratios is not None else None,
            P.spatial_letter_min_ink,
        )
        letters = _clean_letters(positive_text, inventory_set)
        if not letters or visual.shape[0] == 0:
            continue

        target_indices = torch.tensor(
            [mapping[ch] for ch in letters],
            device=visual.device,
            dtype=torch.long,
        )
        target = prototypes.index_select(0, target_indices)
        path = _cfm_hard_monotonic_path(
            visual.detach(),
            target,
            P.spatial_letter_step_penalty,
        )
        if not path:
            continue

        positives_by_window = {}
        for window_index, letter_index in path:
            positives_by_window.setdefault(int(window_index), set()).add(
                int(target_indices[letter_index].item())
            )

        logits = torch.matmul(
            F.normalize(visual.float(), p=2, dim=-1),
            prototypes.T,
        ) / max(float(P.dense_letter_temperature), 1e-4)
        log_probabilities = F.log_softmax(logits, dim=-1)

        window_losses = []
        for window_index, positive_ids in positives_by_window.items():
            ids = torch.tensor(
                sorted(positive_ids),
                device=visual.device,
                dtype=torch.long,
            )
            # If a 32px window spans two neighboring letters, probability mass
            # assigned to either aligned letter is counted as correct.
            window_losses.append(
                -torch.logsumexp(
                    log_probabilities[window_index].index_select(0, ids),
                    dim=0,
                )
            )
        if window_losses:
            sample_losses.append(torch.stack(window_losses).mean())

    if not sample_losses:
        return local_tokens.sum() * 0.0
    return torch.stack(sample_losses).mean()


def local_context_consistency_loss(
    contextual, local, ink, min_ink
):
    """Prevent the Transformer from erasing window-level semantic identity."""
    target = F.normalize(local.detach().float(), p=2, dim=-1)
    prediction = F.normalize(contextual.float(), p=2, dim=-1)
    cost = 1.0 - (prediction * target).sum(dim=-1)
    if ink is not None:
        mask = ink.to(cost.device).float() >= float(min_ink)
        if bool(mask.any()):
            cost = cost[mask]
    return cost.mean() if cost.numel() else contextual.sum() * 0.0


def install_training_objective(train_module):
    """Freeze text, add local letter DTW/dense loss, retain contextual Span-DTW."""
    original_build_text_encoder = train_module.build_text_encoder

    def build_text_encoder():
        # trainer_core uses seed+rank. A frozen semantic anchor must instead be
        # byte-identical on every DDP rank, so construct it under a fixed RNG
        # fork and then restore each rank's normal RNG state.
        devices = (
            [torch.cuda.current_device()]
            if torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=devices):
            anchor_seed = int(getattr(train_module.P, "train_seed", 42)) + 1907
            torch.manual_seed(anchor_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(anchor_seed)
            encoder = original_build_text_encoder()

        source = maybe_load_text_anchor(encoder)
        freeze_text_encoder(encoder)
        encoder._cfm_text_anchor_source = source
        return encoder

    train_module.build_text_encoder = build_text_encoder

    # If --weights/--resume subsequently loads a text state, keep it frozen and
    # re-apply an explicitly requested anchor after checkpoint initialization.
    if hasattr(train_module, "_load_initial_states"):
        original_load_states = train_module._load_initial_states

        def load_initial_states(args, model, text_encoder):
            payload = original_load_states(args, model, text_encoder)
            if os.environ.get("TEXT_ANCHOR_WEIGHTS", "").strip():
                maybe_load_text_anchor(text_encoder)
            freeze_text_encoder(text_encoder)
            return payload

        train_module._load_initial_states = load_initial_states

    original_single = train_module.compute_single_image_text_loss

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
        # Base contextual Span-DTW acts on F_i. Disable the old local-hard-neg
        # heuristic because direct language grounding supersedes it.
        loss, stats, embeddings = original_single(
            image_embedder,
            text_encoder,
            criterion,
            images,
            pos_texts,
            neg_texts,
            embeddings,
            False,
        )
        if embeddings is None:
            return loss, stats, embeddings

        contextual, local_tokens, ink_ratios, _raw_local = embeddings

        if bool(getattr(train_module.P, "spatial_letter_enabled", True)):
            # Reuse the tested differentiable monotonic local DTW implementation
            # above, now with fully frozen text prototypes.
            local_loss, local_stats = letter_depiction_loss(
                train_module.P,
                text_encoder,
                local_tokens,
                ink_ratios,
                pos_texts,
                neg_texts,
            )
            dense_loss = dense_letter_inventory_loss(
                train_module.P,
                text_encoder,
                local_tokens,
                ink_ratios,
                pos_texts,
            )
            loss = (
                loss
                + float(train_module.P.spatial_letter_weight) * local_loss
                + float(train_module.P.dense_letter_weight) * dense_loss
            )

            # Rename legacy local-DTW metrics to the new representation semantics.
            stats.update(
                {
                    "spatial_letter_loss": float(local_loss.detach().item()),
                    "spatial_letter_pos_cost": local_stats.get(
                        "letter_pos_cost", 0.0
                    ),
                    "spatial_letter_neg_cost": local_stats.get(
                        "letter_neg_cost", 0.0
                    ),
                    "spatial_letter_gap": local_stats.get(
                        "letter_gap", 0.0
                    ),
                    "spatial_letter_negative_candidates": local_stats.get(
                        "letter_negative_candidates", 0.0
                    ),
                    "dense_letter_loss": float(dense_loss.detach().item()),
                }
            )

        consistency = local_context_consistency_loss(
            contextual,
            local_tokens,
            ink_ratios,
            train_module.P.spatial_letter_min_ink,
        )
        if (
            torch.is_grad_enabled()
            and float(train_module.P.local_context_consistency_weight) > 0
        ):
            loss = loss + (
                float(train_module.P.local_context_consistency_weight)
                * consistency
            )
        stats["local_context_consistency"] = float(
            consistency.detach().item()
        )
        stats["total"] = float(loss.detach().item())
        return loss, stats, embeddings

    train_module.compute_single_image_text_loss = compute_single_image_text_loss

    # Log branch-specific terms before the shared W&B logger commits the step.
    if hasattr(train_module, "wandb_log_epoch_metrics"):
        original_logger = train_module.wandb_log_epoch_metrics

        def wandb_log_epoch_metrics(
            run, epoch, train_loss, val_loss, train_stats
        ):
            if (
                run is not None
                and getattr(train_module, "wandb", None) is not None
            ):
                train_module.wandb.log(
                    {
                        "spatial/letter_dtw": float(
                            train_stats.get("spatial_letter_loss", 0.0)
                        ),
                        "spatial/letter_pos_cost": float(
                            train_stats.get("spatial_letter_pos_cost", 0.0)
                        ),
                        "spatial/letter_neg_cost": float(
                            train_stats.get("spatial_letter_neg_cost", 0.0)
                        ),
                        "spatial/letter_gap": float(
                            train_stats.get("spatial_letter_gap", 0.0)
                        ),
                        "spatial/dense_letter": float(
                            train_stats.get("dense_letter_loss", 0.0)
                        ),
                        "spatial/local_context_consistency": float(
                            train_stats.get(
                                "local_context_consistency", 0.0
                            )
                        ),
                    },
                    step=int(epoch),
                    commit=False,
                )
            return original_logger(
                run, epoch, train_loss, val_loss, train_stats
            )

        train_module.wandb_log_epoch_metrics = wandb_log_epoch_metrics


def model_config(P):
    return {
        "architecture_family": "cfm-inspired-spatial-language-alignment",
        "spatial_identity_preserved": True,
        "local_representation":
            "semantic-affinity-vote -> image-side-language-adapter",
        "context_representation":
            "4-layer-self-attention -> gated-local-preserving-fusion",
        "text_encoder_policy": "fully-frozen-anchor",
        "text_anchor_weights": os.environ.get("TEXT_ANCHOR_WEIGHTS", ""),
        "spatial_affinity_radius": int(P.spatial_affinity_radius),
        "spatial_affinity_temperature": float(
            P.spatial_affinity_temperature
        ),
        "spatial_affinity_distance_penalty": float(
            P.spatial_affinity_distance_penalty
        ),
        "spatial_affinity_initial_gate": float(
            P.spatial_affinity_initial_gate
        ),
        "context_residual_initial_gate": float(
            P.context_residual_initial_gate
        ),
        "spatial_letter_weight": float(P.spatial_letter_weight),
        "dense_letter_weight": float(P.dense_letter_weight),
        "dense_letter_temperature": float(P.dense_letter_temperature),
        "local_context_consistency_weight": float(
            P.local_context_consistency_weight
        ),
        "local_negative_policy": "all-10-candidates -> hardest-sequence",
        "pair_interaction": "none-before-cosine; independent-line-encoders",
        "final_pair_similarity": "cosine-on-independent-fused-context",
    }

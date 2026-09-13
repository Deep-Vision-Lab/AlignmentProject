from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from embeddingModel import EmbeddingModel, sliding_window
from restoration_epoch_probe import _hard_dtw
from restoration_recommended_components import (
    LocalContextFusion,
    contrastive_margin_from_costs,
    denormalize_imagenet_windows,
    line_padding_masks,
)
from restoration_window_seq2seq import (
    WindowSequenceCNNEncoder,
    WindowSequenceStrokeDecoder,
)
from vlm_restoration_positive_dtw import (
    _soft_dtw_cost_matrix,
    attach_restoration_dtw_stages,
    stroke_restoration_loss,
)
from zero_shot_preprocessing import ManuscriptLinePreprocessor


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _normalize(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - MEAN.to(rgb)) / STD.to(rgb)


def _recommended_model(vector_size=32, width=64):
    del width
    model = EmbeddingModel(
        window_size=32,
        stride=16,
        vector_size=vector_size,
        device="cpu",
        use_flip=False,
        input_height=128,
        vit_layers=1,
        vit_heads=4,
        vit_mlp_dim=max(64, vector_size * 2),
        vit_dropout=0.0,
        vit_max_tokens=64,
        vit_position_base_tokens=7,
        vit_binarize_input=False,
    )
    config = SimpleNamespace(
        restoration_decoder_channels=64,
        restoration_contrast_scale=0.15,
        restoration_semantic_adapter="identity",
        restoration_local_encoder="cnn_seq2seq",
        restoration_training_stage="align",
    )
    return attach_restoration_dtw_stages(model, config).eval()


def test_point_01_crop_outer_margin_preserve_internal_space_rgb_and_geometry():
    array = np.full((80, 300, 3), 255, dtype=np.uint8)
    array[18:55, 20:82] = np.asarray([180, 20, 20], dtype=np.uint8)
    array[8:12, 76:80] = np.asarray([25, 25, 25], dtype=np.uint8)  # tiny dot
    array[25:62, 220:282] = np.asarray([20, 45, 185], dtype=np.uint8)
    source = Image.fromarray(array, mode="RGB")
    processor = ManuscriptLinePreprocessor(
        size=(128, 256),
        training=False,
        augment=False,
        binarize=False,
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
        autocontrast=False,
    )
    processed, meta = processor.preprocess_with_metadata(source)
    out = np.asarray(processed)
    assert processed.mode == "RGB"
    assert meta["crop_left"] <= 20
    assert meta["crop_right"] >= 282
    assert meta["crop_top"] <= 8  # safety margin preserved the tiny dot
    assert meta["crop_width"] > 200  # internal gap was not deleted
    assert meta["scale_x"] > 0 and meta["scale_y"] > 0
    assert "offset_x" in meta and "resize_scale" in meta
    # Color survives; the model image was not replaced by the temporary gray mask.
    assert np.max(np.ptp(out.astype(np.int16), axis=2)) > 20


def test_point_02_reconstruction_target_is_exact_original_rgb_window():
    torch.manual_seed(2)
    model = _recommended_model()
    rgb = torch.rand(1, 3, 128, 64)
    image = _normalize(rgb)
    with torch.no_grad():
        bundle = model(image, return_training_bundle=True)
    expected = denormalize_imagenet_windows(sliding_window(image, 32, 16))
    assert bundle["restoration_target"].shape == (1, 3, 3, 128, 32)
    assert torch.allclose(bundle["restoration_target"], expected, atol=1e-6)
    assert bundle["restoration"].shape == expected.shape


def test_point_03_restormer_pretrained_probe_contract():
    from restormer_pretrained_probe import find_restormer_assets, load_restormer

    assets = find_restormer_assets()
    if not assets.ready:
        pytest.skip(assets.message)
    model = load_restormer(assets, device="cpu")
    assert hasattr(model, "latent")
    assert sum(p.numel() for p in model.parameters()) > 0


def test_point_04_encoder_preserves_horizontal_spatial_detail():
    encoder = WindowSequenceCNNEncoder(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=32,
        base_channels=16,
    )
    assert encoder.encoder[1].block[0].stride == (2, 1)
    assert encoder.encoder[2].block[0].stride == (2, 1)
    spatial, batch, count = encoder._spatial_features(
        torch.randn(1, 3, 128, 64)
    )
    assert (batch, count) == (1, 3)
    assert spatial.shape[-2:] == (8, 8)


def test_point_05_decoder_has_no_image_skip_path_and_depends_on_local_features():
    torch.manual_seed(5)
    decoder = WindowSequenceStrokeDecoder(
        dim=32, output_height=128, output_width=32, channels=64
    ).eval()
    tokens = torch.randn(1, 2, 32)
    with torch.no_grad():
        original = decoder(tokens)
        swapped = decoder(tokens.flip(1))
    assert not torch.allclose(original[:, 0], original[:, 1])
    assert torch.allclose(swapped[:, 0], original[:, 1], atol=1e-6)
    assert torch.allclose(swapped[:, 1], original[:, 0], atol=1e-6)


def test_point_06_transformer_context_changes_a_window_when_neighbor_changes():
    torch.manual_seed(6)
    model = _recommended_model()
    vit = model.vit_encoder
    local = torch.randn(1, 3, vit.embed_dim)
    valid = torch.ones(1, 3, dtype=torch.bool)
    positional = local + vit._position_tokens(3)
    with torch.no_grad():
        first = vit.encoder(positional, src_key_padding_mask=~valid)
        changed = local.clone()
        changed[:, 0] += 25.0
        second = vit.encoder(
            changed + vit._position_tokens(3),
            src_key_padding_mask=~valid,
        )
    assert not torch.allclose(first[:, 2], second[:, 2], atol=1e-7, rtol=1e-6)


def test_point_07_fusion_uses_both_local_and_context_and_normalizes():
    torch.manual_seed(7)
    fusion = LocalContextFusion(32).eval()
    local = torch.randn(1, 4, 32)
    context = torch.randn(1, 4, 32)
    with torch.no_grad():
        fused = fusion(local, context)
        changed_local = fusion(local + 2.0, context)
        changed_context = fusion(local, context - 2.0)
    assert fused.shape == local.shape
    assert torch.allclose(
        torch.linalg.vector_norm(fused.float(), dim=-1),
        torch.ones(1, 4),
        atol=1e-5,
    )
    assert not torch.allclose(fused, changed_local)
    assert not torch.allclose(fused, changed_context)


def test_point_08_reconstruction_and_negative_margin_losses_both_train():
    positive = torch.tensor(0.55, requires_grad=True)
    negatives = [torch.tensor(0.25), torch.tensor(0.35)]
    contrastive = contrastive_margin_from_costs(positive, negatives, margin=0.20)
    assert float(contrastive) > 0
    contrastive.backward()
    assert positive.grad is not None and float(positive.grad) > 0

    settings = SimpleNamespace(
        restoration_pixel_weight=1.0,
        restoration_edge_weight=0.5,
        restoration_structure_weight=0.25,
    )
    target = torch.rand(1, 3, 3, 16, 16)
    faithful = target.clone()
    collapsed = target[:, :1].repeat(1, 3, 1, 1, 1)
    valid = torch.ones(1, 3, 1, 16, 16)
    faithful_loss, *_ = stroke_restoration_loss(
        settings, faithful, target, valid
    )
    collapsed_loss, *_ = stroke_restoration_loss(
        settings, collapsed, target, valid
    )
    assert float(faithful_loss) < float(collapsed_loss)


def test_point_09_dtw_supports_many_windows_one_letter_and_one_window_many_letters():
    many_windows = torch.zeros(4, 1, requires_grad=True)
    cost_many_to_one = _soft_dtw_cost_matrix(
        many_windows,
        gamma=0.05,
        vertical_penalty=0.05,
        horizontal_penalty=0.30,
        disable_horizontal_when_feasible=True,
    )
    one_window = torch.zeros(1, 4, requires_grad=True)
    cost_one_to_many = _soft_dtw_cost_matrix(
        one_window,
        gamma=0.05,
        vertical_penalty=0.05,
        horizontal_penalty=0.30,
        disable_horizontal_when_feasible=True,
    )
    assert torch.isfinite(cost_many_to_one)
    assert torch.isfinite(cost_one_to_many)
    (cost_many_to_one + cost_one_to_many).backward()
    assert many_windows.grad is not None
    assert one_window.grad is not None


def test_point_10_dtw_path_is_recomputed_from_current_cost_matrix():
    first = np.asarray([[0.0, 9.0], [0.0, 9.0], [9.0, 0.0]], dtype=np.float32)
    second = np.asarray([[0.0, 9.0], [9.0, 0.0], [9.0, 0.0]], dtype=np.float32)
    path1, _ = _hard_dtw(first, 0.05)
    path2, _ = _hard_dtw(second, 0.05)
    assert path1 != path2
    assert path1[0] == path2[0] == (0, 0)
    assert path1[-1] == path2[-1] == (2, 1)


def test_point_11_multiple_spatial_vectors_per_window_are_available_as_ablation():
    encoder = WindowSequenceCNNEncoder(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=32,
        base_channels=16,
    ).eval()
    with torch.no_grad():
        vectors = encoder.spatial_vectors(
            torch.randn(1, 3, 128, 64), vectors_per_window=4
        )
    assert vectors.shape == (1, 3, 4, 32)
    norms = torch.linalg.vector_norm(vectors, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_point_12_tiny_overfit_tool_exists_and_uses_eight_windows():
    source = Path("tools/restoration_tiny_overfit.py").read_text(encoding="utf-8")
    assert "WINDOW_COUNT = 8" in source
    assert "optimizer.step()" in source
    assert "swap_delta" in source


def test_point_13_primary_evaluation_representation_is_image_only_fused_vector():
    torch.manual_seed(13)
    model = _recommended_model()
    line1 = _normalize(torch.rand(1, 3, 128, 64))
    line2 = _normalize(torch.rand(1, 3, 128, 64))
    with torch.no_grad():
        fused1, local1, valid1 = model(line1, return_local=True, return_ink=True)
        fused2, local2, valid2 = model(line2, return_local=True, return_ink=True)
        similarity = fused1[0] @ fused2[0].T
    assert fused1.shape == local1.shape == (1, 3, 32)
    assert fused2.shape == local2.shape == (1, 3, 32)
    assert valid1.shape == valid2.shape == (1, 3)
    assert torch.isfinite(similarity).all()

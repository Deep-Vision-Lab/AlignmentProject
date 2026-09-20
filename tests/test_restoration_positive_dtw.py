import torch
from types import SimpleNamespace

from embeddingModel import EmbeddingModel
from physical_window_vit_branch import attach_physical_window_vit_stages
from physical_window_vit_encoder import PhysicalWindowEmbedding
from resnet18_window_encoder import ResNet18WindowEncoder
from vlm_restoration_positive_dtw import (
    _clean_letters,
    _soft_dtw_cost_matrix,
    positive_monotonic_letter_dtw_cost,
    strong_sigreg_loss,
)


def _model():
    model = EmbeddingModel(
        window_size=32,
        stride=16,
        vector_size=192,
        device="cpu",
        use_flip=True,
        input_height=128,
        vit_layers=12,
        vit_heads=3,
        vit_mlp_dim=768,
        vit_dropout=0.0,
        vit_max_tokens=64,
        vit_position_base_tokens=7,
        vit_binarize_input=False,
    )
    config = SimpleNamespace(
        resnet18_pretrained=False,
        tiny_vit_pretrained=False,
        pretrained_local_only=True,
        restoration_semantic_adapter="identity",
        restoration_local_encoder="resnet18",
        restoration_training_stage="align",
    )
    return attach_physical_window_vit_stages(model, config)


def test_positive_dtw_is_differentiable():
    visual = torch.randn(9, 192, requires_grad=True)
    target = torch.randn(4, 192)
    loss = positive_monotonic_letter_dtw_cost(
        visual,
        target,
        gamma=0.05,
        step_penalty=0.02,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert visual.grad is not None
    assert torch.isfinite(visual.grad).all()


def test_matching_order_costs_less_than_reversed_order():
    generator = torch.Generator().manual_seed(7)
    letters = torch.randn(4, 192, generator=generator)
    letters = torch.nn.functional.normalize(letters, p=2, dim=-1)
    visual = torch.repeat_interleave(letters, repeats=2, dim=0)
    correct = positive_monotonic_letter_dtw_cost(
        visual, letters, gamma=0.01, step_penalty=0.02
    )
    reversed_target = positive_monotonic_letter_dtw_cost(
        visual, torch.flip(letters, dims=[0]), gamma=0.01, step_penalty=0.02
    )
    assert float(correct) < float(reversed_target)


def test_unicode_arabic_cleaning_keeps_wasla_and_decomposes_ligature():
    cleaned = _clean_letters(" ٱلْﻻ! ")
    assert cleaned == ["ٱ", "ل", "ل", "ا"]


def test_resnet18_window_encoder_accepts_true_grayscale_windows():
    encoder = ResNet18WindowEncoder(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=192,
        pretrained=False,
        input_channels=1,
    )
    image = torch.randn(1, 1, 128, 64)
    tokens = encoder(image)
    assert tokens.shape == (1, 192, 1, 3)
    assert encoder.backbone.conv1.in_channels == 1
    assert tuple(encoder.backbone.conv1.weight.shape[:2]) == (64, 1)


def test_resnet18_window_encoder_keeps_one_token_per_window():
    encoder = ResNet18WindowEncoder(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=192,
        pretrained=False,
    )
    image = torch.randn(1, 3, 128, 64)
    tokens = encoder(image)
    assert tokens.shape == (1, 192, 1, 3)
    assert encoder.projection[0].in_features == 512
    assert encoder.projection[0].out_features == 192


def test_physical_window_vit_uses_one_whole_window_per_token():
    encoder = PhysicalWindowEmbedding(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=192,
    )
    image = torch.randn(1, 3, 128, 64)
    windows = encoder.extract_windows(image)
    tokens = encoder(image)
    assert windows.shape == (1, 3, 3, 128, 32)
    assert tokens.shape == (1, 192, 1, 3)
    assert encoder.projection.in_features == 3 * 128 * 32
    assert encoder.projection.out_features == 192


def test_local_and_context_paths_receive_identical_physical_windows():
    model = _model()
    image = torch.arange(1 * 3 * 128 * 64, dtype=torch.float32).reshape(
        1, 3, 128, 64
    )
    local_encoder = model.vit_encoder.patch_embedding
    context_encoder = model.vit_encoder.encoder.context_window_embedding
    local_windows = local_encoder.extract_windows(image)
    context_windows = context_encoder.extract_windows(image)

    assert local_windows.shape == context_windows.shape == (1, 3, 3, 128, 32)
    assert torch.equal(local_windows, context_windows)
    # Explicit geometry: x=[0:32], [16:48], [32:64].
    assert torch.equal(local_windows[:, 0], image[:, :, :, 0:32])
    assert torch.equal(local_windows[:, 1], image[:, :, :, 16:48])
    assert torch.equal(local_windows[:, 2], image[:, :, :, 32:64])


def test_full_model_is_resnet18_plus_direct_window_tinyvit_without_decoder():
    model = _model()
    image = torch.randn(1, 3, 128, 64)
    bundle = model(image, return_training_bundle=True)

    assert isinstance(model.vit_encoder.patch_embedding, ResNet18WindowEncoder)
    assert isinstance(
        model.vit_encoder.encoder.context_window_embedding,
        PhysicalWindowEmbedding,
    )
    assert model.vit_encoder.context_window_height == 128
    assert model.vit_encoder.context_window_width == 32
    assert model.vit_encoder.context_window_stride == 16
    assert model.vit_encoder.context_window_subdivision == "none"
    assert model.vit_encoder.embed_dim == 192
    assert len(model.vit_encoder.encoder.layers) == 12
    first = model.vit_encoder.encoder.layers[0]
    assert first.self_attn.num_heads == 3
    assert first.linear1.out_features == 768

    assert bundle["primitive"].shape == (1, 3, 192)
    assert bundle["contextual"].shape == (1, 3, 192)
    assert bundle["semantic"].shape == (1, 3, 192)
    assert bundle["token_valid"].shape == (1, 3)
    assert "restoration" not in bundle
    assert "restoration_target" not in bundle
    assert not hasattr(model.vit_encoder, "stroke_decoder")
    assert not any("stroke_decoder" in key for key in model.state_dict())


def test_disabling_horizontal_moves_removes_soft_alternative_paths_when_feasible():
    costs = torch.zeros(2, 2)
    allowed = _soft_dtw_cost_matrix(
        costs,
        gamma=0.1,
        vertical_penalty=0.0,
        horizontal_penalty=0.0,
        disable_horizontal_when_feasible=False,
    )
    disabled = _soft_dtw_cost_matrix(
        costs,
        gamma=0.1,
        vertical_penalty=0.0,
        horizontal_penalty=0.0,
        disable_horizontal_when_feasible=True,
    )
    assert float(disabled) > float(allowed)


def test_horizontal_moves_remain_available_when_text_is_longer_than_windows():
    costs = torch.zeros(2, 3)
    value = _soft_dtw_cost_matrix(
        costs,
        gamma=0.05,
        vertical_penalty=0.05,
        horizontal_penalty=0.30,
        disable_horizontal_when_feasible=True,
    )
    assert torch.isfinite(value)


def test_gradient_probes_cover_every_active_stage():
    model = _model()
    model.train()
    model._gradient_probe_records = []
    image = torch.randn(1, 3, 128, 64)
    bundle = model(image, return_training_bundle=True)

    generator = torch.Generator().manual_seed(19)
    direction = torch.randn(
        bundle["semantic"].shape,
        generator=generator,
        device=bundle["semantic"].device,
        dtype=bundle["semantic"].dtype,
    )
    loss = (bundle["semantic"] * direction).sum()
    loss.backward()

    assert len(model._gradient_probe_records) == 1
    record = model._gradient_probe_records[0]
    for stage in (
        "after_resnet18",
        "after_vit_tiny",
        "after_fusion",
        "final_fused",
    ):
        gradient = record[stage].grad
        assert gradient is not None, stage
        assert torch.isfinite(gradient).all(), stage
        assert float(gradient.float().norm()) > 0.0, stage

    assert any(
        parameter.grad is not None
        for parameter in model.vit_encoder.patch_embedding.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.vit_encoder.encoder.context_window_embedding.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.vit_encoder.encoder.layers.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.vit_encoder.fusion_head.parameters()
    )



def test_strong_sigreg_penalizes_collapsed_pre_l2_embeddings_and_backpropagates():
    generator = torch.Generator().manual_seed(123)
    gaussian = torch.randn(8, 16, 192, generator=generator, requires_grad=True)
    valid = torch.ones(8, 16, dtype=torch.bool)

    # Fix the global RNG before each call so both distributions see the same
    # random observer directions.
    torch.manual_seed(77)
    gaussian_loss, gaussian_stats = strong_sigreg_loss(
        gaussian,
        valid,
        sketch_dim=32,
        num_knots=17,
        t_min=0.0,
        t_max=3.0,
        min_samples=32,
        slice_chunk=8,
    )

    collapsed = torch.zeros(8, 16, 192, requires_grad=True)
    collapsed = collapsed + 0.25
    torch.manual_seed(77)
    collapsed_loss, collapsed_stats = strong_sigreg_loss(
        collapsed,
        valid,
        sketch_dim=32,
        num_knots=17,
        t_min=0.0,
        t_max=3.0,
        min_samples=32,
        slice_chunk=8,
    )

    assert torch.isfinite(gaussian_loss)
    assert torch.isfinite(collapsed_loss)
    assert float(collapsed_loss) > float(gaussian_loss)
    assert gaussian_stats["sigreg_samples"] == 128.0
    assert collapsed_stats["sigreg_dim_std_min"] == 0.0

    gaussian_loss.backward()
    assert gaussian.grad is not None
    assert torch.isfinite(gaussian.grad).all()
    assert float(gaussian.grad.norm()) > 0.0

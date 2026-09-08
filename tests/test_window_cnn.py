"""Pixel preservation and real optimized-update checks without HF downloads."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from embeddingModel import EmbeddingModel, sliding_window
from window_cnn import SlidingWindowCNN, extract_rgb_windows
from vlm_letter_grounding import attach_depiction_head, monotonic_letter_dtw_cost


@pytest.mark.parametrize("width,stride", [(32, 16), (1024, 16), (1024, 8), (1031, 16)])
def test_windows_are_exact_pixel_copies(width, stride):
    image = torch.arange(3 * 128 * width).reshape(1, 3, 128, width).float()
    original = image.clone()
    windows = extract_rgb_windows(image, 32, stride)
    assert windows.shape == (1, 1 + (width - 32) // stride, 3, 128, 32)
    for index in range(windows.shape[1]):
        assert torch.equal(windows[:, index], image[..., index * stride:index * stride + 32])
    assert torch.equal(windows, sliding_window(image, 32, stride))
    windows.zero_()
    assert torch.equal(image, original), "Extracted windows must not alias input storage"


def test_extraction_preserves_noncontiguous_pixels_and_overlap_gradients():
    image = torch.randn(1, 3, 128, 128, requires_grad=True)
    windows = extract_rgb_windows(image[..., ::2], 32, 16)
    for index in range(3):
        assert torch.equal(windows[:, index], image[..., ::2][..., index * 16:index * 16 + 32])
    windows.sum().backward()
    expected = torch.zeros_like(image)
    for index in range(3):
        expected[..., index * 32:index * 32 + 64:2] += 1
    assert torch.equal(image.grad, expected)


def test_cnn_tokens_stay_local_and_do_not_change_input_pixels():
    cnn = SlidingWindowCNN(128, 32, 16, 128).train()
    image = torch.randn(1, 3, 128, 80)
    original = image.clone()
    windows = extract_rgb_windows(image)
    tokens = cnn.encode_windows(windows)
    single = cnn.encode_windows(windows[:, :1])
    torch.testing.assert_close(single, tokens[:, :1], rtol=1e-5, atol=1e-6)
    assert torch.equal(image, original)
    assert torch.equal(windows, extract_rgb_windows(image))


def test_actual_branch_builds_direct_cnn_tokens_and_preserves_rtl():
    import model_backend
    model = model_backend.build_visual_model(
        window_size=32, stride=16, vector_size=128, device="cpu", use_flip=False
    ).eval()
    assert isinstance(model.vit_encoder.patch_embedding, SlidingWindowCNN)
    assert not hasattr(model.vit_encoder, "depiction_projection")
    assert not hasattr(model.vit_encoder, "depiction_norm")
    assert model.vit_layers == 4
    assert model_backend.visual_model_config()["letter_depiction_head"] is False
    image = torch.randn(1, 3, 128, 1024)
    with torch.no_grad():
        context, local, ink = model(image, return_local=True, return_ink=True)
        model._use_flip_state.fill_(1)
        _, reversed_local, reversed_ink = model(image, return_local=True, return_ink=True)
    assert context.shape == local.shape == (1, 63, 128)
    torch.testing.assert_close(reversed_local, local.flip(1))
    torch.testing.assert_close(reversed_ink, ink.flip(1))


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_and_cnn_state_dict_round_trips(enabled):
    kwargs = dict(window_size=32, stride=16, vector_size=16, device="cpu",
                  input_height=128, vit_layers=1, vit_heads=4, vit_mlp_dim=32,
                  vit_dropout=0.0, vit_binarize_input=False, window_cnn_enabled=enabled)
    first = attach_depiction_head(EmbeddingModel(**kwargs)).eval()
    second = attach_depiction_head(EmbeddingModel(**kwargs)).eval()
    second.load_state_dict(first.state_dict(), strict=True)
    assert first.model_config()["window_cnn_enabled"] is enabled
    assert first.model_config()["letter_depiction_head"] is (not enabled)
    image = torch.randn(1, 3, 128, 64)
    with torch.no_grad():
        torch.testing.assert_close(first(image), second(image), rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluation_loader_restores_exact_encoder(tmp_path, enabled):
    from Evaluation.vit_evaluation import install_vit_evaluation_loader
    from Evaluation import _eval_utils
    model = attach_depiction_head(EmbeddingModel(
        vector_size=16, device="cpu", vit_layers=1, vit_heads=4,
        vit_mlp_dim=32, vit_dropout=0.0, vit_binarize_input=False,
        window_cnn_enabled=enabled,
    )).eval()
    config = model.model_config()
    config.update(window_size=32, stride=16, vector_size=16, lang="English",
                  letter_depiction_enabled=True)
    if not enabled:
        config.pop("window_cnn_enabled")  # Original checkpoints have no flag.
    path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "model_config": config}, path)
    install_vit_evaluation_loader()
    restored = _eval_utils.load_evaluation_models(path, device="cpu", load_text_model=False)
    assert restored.image_model.window_cnn_enabled is enabled
    assert hasattr(restored.image_model.vit_encoder, "depiction_projection") is (not enabled)
    image = torch.randn(1, 3, 128, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(image), restored.image_model(image), rtol=0, atol=0)


def _offline_text_encoder(monkeypatch):
    # Keep the actual ArabicSpanTextEncoder constructor, freeze logic, pooling,
    # projection and cache. Only HF tokenizer/model downloads are replaced.
    import arabic_span_text_encoder_legacy as legacy
    from arabic_span_text_encoder import ArabicSpanTextEncoder

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16)
            self.embedding = nn.Embedding(64, 16)
            self.dropout = nn.Dropout(0.5)

        def forward(self, input_ids, attention_mask):
            assert not self.training
            assert not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))

    class Tokenizer:
        def __call__(self, strings, **kwargs):
            ids = torch.tensor([[1, 3 + sum(map(ord, text)) % 60, 2] for text in strings])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                    "special_tokens_mask": torch.tensor([[1, 0, 1]] * len(strings))}

    monkeypatch.setattr(legacy, "AutoModel", SimpleNamespace(from_pretrained=lambda *a, **k: Backbone()))
    monkeypatch.setattr(legacy, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
    return ArabicSpanTextEncoder(output_dim=128, freeze_backbone=True, device="cpu", cache_size=16)


def test_optimized_training_updates_cnn_transformer_and_text_fc_only(monkeypatch):
    import model_backend
    import training_optimizations as opt
    from training_stability import install_training_stability

    torch.manual_seed(42)
    monkeypatch.setenv("GRADIENT_ACCUMULATION_STEPS", "2")
    monkeypatch.setenv("USE_FUSED_ADAM", "0")
    monkeypatch.setenv("PROFILE_MAX_BATCHES", "0")
    model = model_backend.build_visual_model(
        window_size=32, stride=16, vector_size=128, device="cpu", use_flip=True
    )
    text = _offline_text_encoder(monkeypatch)
    if hasattr(model_backend.P, "cross_attention_enabled"):
        from vlm_pair_cross_attention import attach_pair_cross_attention
        attach_pair_cross_attention(text, model_backend.P)

    image = torch.randn(1, 3, 128, 64)
    original = image.clone()
    frozen_before = {name: p.detach().clone() for name, p in text.backbone.named_parameters()}
    watched = {name: p for name, p in model.named_parameters()
               if name.endswith("weight") and ("patch_embedding.features" in name
                   or "patch_embedding.projection" in name or "self_attn.in_proj" in name)}
    watched["text.projection.weight"] = text.projection.weight
    if hasattr(text, "pair_cross_attention"):
        watched["pair.attention"] = text.pair_cross_attention.attention.in_proj_weight
    before = {name: p.detach().clone() for name, p in watched.items()}
    captured = []
    original_adam = torch.optim.Adam

    def capture_adam(parameters, **kwargs):
        optimizer = original_adam(parameters, **kwargs)
        captured.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", capture_adam)

    def loss_fn(image_model, text_model, criterion, batch):
        context, local = image_model(batch, return_local=True)
        targets = F.normalize(text_model._project_surfaces(["ا", "ب"]), dim=-1)
        loss = monotonic_letter_dtw_cost(local[0], targets, gamma=0.05, step_penalty=0.02)
        loss = loss + 1 - (F.normalize(context[0, :2], dim=-1) * targets).sum(-1).mean()
        if hasattr(text_model, "pair_cross_attention"):
            other = image_model(batch.flip(-1))
            left, right, _, _ = text_model.pair_cross_attention(context, other)
            loss = loss + (left[:, :2] - targets).square().mean() + (right[:, :2] - targets.flip(0)).square().mean()
        return loss, {}

    def accumulate(total, stats, weight):
        for key, value in stats.items():
            total[key] = total.get(key, 0.0) + float(value) * weight

    runtime = SimpleNamespace(
        P=SimpleNamespace(device="cpu", valid_every_n_epochs=1, valid_max_batches=0),
        CTX=SimpleNamespace(enabled=False, is_main=False, world_size=1, barrier=lambda: None),
        USE_AMP=False, compute_batch_loss=loss_fn,
        has_trainable_parameters=lambda module: any(p.requires_grad for p in module.parameters()),
        _batch_size=lambda batch: len(batch), _accumulate_stats=accumulate,
        _merge_epoch_payload=lambda payload: (payload["loss_sum"] / payload["weight"], payload["stats_sum"]),
        init_wandb=lambda *a: None, validate=lambda *a, **k: (0.0, {}),
    )
    install_training_stability(runtime, {}, "window_cnn_test")
    args = SimpleNamespace(learning_rate=1e-3, epochs=1)
    # This executes the real optimized optimizer constructor and the real
    # stability/accumulation/backward/step loop, with an offline synthetic loss.
    opt.optimized_train(runtime)(model, text, None, [image, image], [], None, args, {})

    optimizer_ids = {id(p) for group in captured[0].param_groups for p in group["params"]}
    for name, parameter in watched.items():
        assert parameter.requires_grad and id(parameter) in optimizer_ids, name
        assert not torch.equal(parameter, before[name]), f"No update: {name}"
        assert torch.isfinite(parameter).all()
    for name, parameter in text.backbone.named_parameters():
        assert not parameter.requires_grad and parameter.grad is None
        assert id(parameter) not in optimizer_ids
        assert torch.equal(parameter, frozen_before[name])
    assert not text.backbone.training
    assert torch.equal(image, original)
    assert torch.equal(extract_rgb_windows(image), extract_rgb_windows(original))

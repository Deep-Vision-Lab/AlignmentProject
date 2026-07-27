import torch

from vit_embedding_model import ViTEmbeddingModel


def _model(use_flip=False):
    return ViTEmbeddingModel(
        window_size=32,
        stride=16,
        vector_size=64,
        device="cpu",
        use_flip=use_flip,
        input_height=128,
        vit_layers=2,
        vit_heads=4,
        vit_mlp_dim=128,
        vit_dropout=0.0,
        vit_max_tokens=128,
    )


def test_vit_emits_one_token_per_existing_sliding_window():
    model = _model().eval()
    image = torch.randn(1, 3, 128, 1024)

    with torch.no_grad():
        contextual, local, grouped, ink = model(
            image,
            return_local=True,
            return_grouped=True,
            return_ink=True,
        )

    # floor((1024 - 32) / 16) + 1 = 63
    assert contextual.shape == (1, 63, 64)
    assert local.shape == (1, 63, 64)
    assert grouped.shape == (1, 63, 64)
    assert ink.shape == (1, 63)
    torch.testing.assert_close(grouped, local)


def test_vit_replaces_cnn_and_bilstm_modules():
    model = _model()

    assert model.use_bilstm is False
    assert model.use_local_grouping is False
    assert not any(isinstance(module, torch.nn.LSTM) for module in model.modules())
    assert model.model_config()["visual_encoder_type"] == "vit"
    assert model.model_config()["use_bilstm"] is False


def test_arabic_flip_reverses_local_window_token_order():
    model = _model(use_flip=False).eval()
    image = torch.randn(1, 3, 128, 256)

    with torch.no_grad():
        _contextual, local_forward = model(image, return_local=True)
        model._use_flip_state.fill_(1)
        _contextual_flipped, local_flipped = model(image, return_local=True)

    torch.testing.assert_close(local_flipped, torch.flip(local_forward, dims=[1]))


def test_vit_heads_must_divide_embedding_dimension():
    try:
        ViTEmbeddingModel(
            vector_size=62,
            vit_heads=4,
            device="cpu",
        )
    except ValueError as exc:
        assert "must divide" in str(exc)
    else:
        raise AssertionError("Expected invalid ViT head configuration to fail")

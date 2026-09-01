import torch

from embeddingModel import EmbeddingModel


def test_dense_stride_produces_125_tokens_from_trained_position_base():
    model = EmbeddingModel(
        window_size=32,
        stride=8,
        vector_size=32,
        device="cpu",
        use_flip=False,
        input_height=128,
        vit_layers=1,
        vit_heads=4,
        vit_mlp_dim=64,
        vit_dropout=0.0,
        vit_max_tokens=256,
        vit_position_base_tokens=63,
    )

    image = torch.randn(2, 3, 128, 1024)
    contextual, local = model(image, return_local=True)

    assert contextual.shape == (2, 125, 32)
    assert local.shape == (2, 125, 32)
    assert model.vit_encoder._position_tokens(125).shape == (1, 125, 32)


def test_dense_positions_do_not_read_untrained_table_tail():
    model = EmbeddingModel(
        window_size=32,
        stride=8,
        vector_size=32,
        device="cpu",
        input_height=128,
        vit_layers=1,
        vit_heads=4,
        vit_mlp_dim=64,
        vit_dropout=0.0,
        vit_max_tokens=256,
        vit_position_base_tokens=63,
    )

    model.zero_grad(set_to_none=True)
    model.vit_encoder._position_tokens(125).sum().backward()
    gradient = model.vit_encoder.position_embedding.grad

    assert gradient is not None
    assert torch.count_nonzero(gradient[:, :63]).item() > 0
    assert torch.count_nonzero(gradient[:, 63:]).item() == 0

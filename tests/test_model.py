import pytest
import torch

from dtw import cosine_similarity_matrix
from losses import compute_loss
from cnn_encoder import CNNEncoder
from model import AlignmentModel, extract_windows
from parameters import Config
from text_embedding import OrthogonalCharEmbedding
from train import build_model
from transformer_encoder import TransformerEncoder


def test_window_count_and_exact_pixels():
    image = torch.arange(1024).view(1,1,1,1024).expand(2,1,128,1024)
    windows = extract_windows(image, 32, 16)
    assert windows.shape == (2,63,1,128,32)
    assert windows[0,62,0,0].tolist() == list(range(992,1024))


@pytest.mark.parametrize('kind', ['simple','resnet18'])
def test_cnn(kind):
    encoder = CNNEncoder(kind, pretrained=False)
    assert encoder(torch.randn(2,1,128,32)).shape == (2,128)


def test_transformer_no_positions_and_independent_layers():
    encoder = TransformerEncoder().eval()
    assert len(encoder.layers) == 5 and encoder.heads == 1 and encoder.feedforward_dim == 512
    assert not torch.equal(encoder.layers[0].linear1.weight, encoder.layers[1].linear1.weight)
    x = torch.randn(2,7,128)
    permutation = torch.tensor([4,2,6,1,3,0,5])
    with torch.no_grad():
        y = encoder(x)
        torch.testing.assert_close(encoder(x[:,permutation]), y[:,permutation], atol=1e-6, rtol=1e-5)


def test_model_shapes_gradients_rtl_and_dropout():
    model = AlignmentModel(cnn_type='simple', pretrained_cnn=False)
    image = torch.randn(2,1,128,1024)
    output = model(image)
    for name in ('local','context','fused','fused_pre_l2'):
        assert output[name].shape == (2,63,128)
    torch.testing.assert_close(output['fused'].norm(dim=-1), torch.ones(2,63))
    assert output['token_valid'].all()
    assert output['physical_window_indices'][0,[0,1,2,60,61,62]].tolist() == [62,61,60,2,1,0]
    (output['fused'] * torch.randn_like(output['fused'])).sum().backward()
    assert model.cnn.projection[0].weight.grad.abs().sum() > 0
    assert model.transformer.layers[0].linear1.weight.grad.abs().sum() > 0
    assert not torch.equal(model(image)['local'], model(image)['local'])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image)['fused'], model(image)['fused'], rtol=0, atol=0)


def test_padding_mask_is_reversed_and_respected():
    model = AlignmentModel(cnn_type='simple', pretrained_cnn=False).eval()
    valid = torch.tensor([[True,True,False]])
    output = model(torch.randn(1,1,32,64), valid)
    assert output['token_valid'].tolist() == [[False,True,True]]
    assert torch.isfinite(output['fused']).all()


@pytest.mark.parametrize(('fusion_mode', 'use_gated_fusion'),
                         [('concat', 0), ('sum', 0), ('sum', 1)])
def test_fusion_modes_support_forward_similarity_loss_and_backward(fusion_mode, use_gated_fusion):
    model = AlignmentModel(cnn_type='simple', pretrained_cnn=False, local_dropout=0.,
                           fusion_mode=fusion_mode, use_gated_fusion=use_gated_fusion).eval()
    image = torch.randn(2,1,32,64)
    output = model(image)
    for name in ('local', 'context', 'fused', 'fused_pre_l2'):
        assert output[name].shape == (2,3,128)
        assert torch.isfinite(output[name]).all()
    similarity = cosine_similarity_matrix(output['fused'][0], OrthogonalCharEmbedding(vocab_size=4096).encode('باب'))
    assert similarity.shape == (3,3)
    assert torch.isfinite(similarity).all()
    config = Config(cnn_type='simple', cnn_pretrained=False, image_height=32, image_width=64,
                    batch_size=2, num_workers=0, sigreg_weight=0., fusion_mode=fusion_mode,
                    use_gated_fusion=use_gated_fusion)
    text = OrthogonalCharEmbedding(vocab_size=4096)
    loss, stats = compute_loss(output, ['باب', 'سلام'], text, config, sketch_seed=7)
    assert torch.isfinite(loss)
    assert stats['evaluated'] == 2
    loss.backward()
    assert model.cnn.projection[0].weight.grad.abs().sum() > 0
    assert model.transformer.layers[0].linear1.weight.grad.abs().sum() > 0
    if use_gated_fusion:
        gate = output['fusion_gate']
        assert gate.shape == (2,3,128)
        assert output['fusion_gate_stats'] is not None
        assert torch.all((gate >= 0) & (gate <= 1))
        grads = [parameter.grad for name, parameter in model.named_parameters() if 'fusion.gate_mlp' in name]
        assert grads and sum(float(grad.abs().sum()) for grad in grads if grad is not None) > 0
    else:
        assert output['fusion_gate'] is None
        assert output['fusion_gate_stats'] is None


def test_concat_with_gated_fusion_is_rejected():
    with pytest.raises(ValueError, match='concat \+ gate ON'):
        build_model(Config(cnn_type='simple', cnn_pretrained=False, fusion_mode='concat', use_gated_fusion=1))

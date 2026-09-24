import pytest
import torch

from cnn_encoder import CNNEncoder
from model import AlignmentModel, extract_windows
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

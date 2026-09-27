import pytest
import torch
from torch import nn

from dtw import cosine_similarity_matrix
from losses import compute_loss
from cnn_encoder import CNNEncoder, simple_cnn_channels
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


@pytest.mark.parametrize('num_layers', [1,2,3,5])
def test_simple_cnn_configurable_depth(num_layers):
    encoder = CNNEncoder('simple', pretrained=False, local_dropout=0., num_layers=num_layers)
    assert encoder.num_layers == num_layers
    assert encoder.channels == simple_cnn_channels(num_layers)
    convs = [module for module in encoder.backbone if isinstance(module, nn.Conv2d)]
    assert len(convs) == num_layers
    output = encoder(torch.randn(4,1,128,32))
    assert output.shape == (4,128)
    assert torch.isfinite(output).all()
    loss = (output * torch.randn_like(output)).sum()
    assert torch.isfinite(loss)
    loss.backward()
    for conv in convs:  # gradients reach every CNN layer; none is disconnected
        assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
        assert conv.weight.grad.abs().sum() > 0
    projection_gradient = encoder.projection[0].weight.grad
    assert projection_gradient is not None and torch.isfinite(projection_gradient).all()
    assert projection_gradient.abs().sum() > 0


def test_simple_cnn_default_three_layers_matches_baseline():
    encoder = CNNEncoder('simple', pretrained=False, local_dropout=0., num_layers=3)
    backbone = list(encoder.backbone)
    assert [type(module) for module in backbone] == [
        nn.Conv2d, nn.GELU, nn.AdaptiveAvgPool2d,
        nn.Conv2d, nn.GELU, nn.AdaptiveAvgPool2d,
        nn.Conv2d, nn.GELU, nn.AdaptiveAvgPool2d, nn.Flatten]
    assert (backbone[0].in_channels, backbone[0].out_channels) == (1,32)
    assert (backbone[3].in_channels, backbone[3].out_channels) == (32,64)
    assert (backbone[6].in_channels, backbone[6].out_channels) == (64,128)
    for conv in (backbone[0], backbone[3], backbone[6]):
        assert conv.kernel_size == (3,3) and conv.padding == (1,1)
    assert backbone[2].output_size == 2
    assert backbone[5].output_size == 2
    assert backbone[8].output_size == 1
    assert encoder.projection[0].in_features == 128


def test_simple_cnn_channel_cap_and_depth_validation():
    assert simple_cnn_channels(1) == [32]
    assert simple_cnn_channels(2) == [32,64]
    assert simple_cnn_channels(3) == [32,64,128]
    assert simple_cnn_channels(4) == [32,64,128,256]
    assert simple_cnn_channels(5) == [32,64,128,256,512]
    assert simple_cnn_channels(7) == [32,64,128,256,512,512,512]
    with pytest.raises(ValueError, match='num_layers'):
        CNNEncoder('simple', pretrained=False, num_layers=0)


def test_resnet18_ignores_num_layers():
    shallow = CNNEncoder('resnet18', pretrained=False, num_layers=1)
    deep = CNNEncoder('resnet18', pretrained=False, num_layers=5)
    assert shallow.num_layers is None and shallow.channels is None
    assert hasattr(shallow.backbone, 'layer4')
    assert list(shallow.state_dict()) == list(deep.state_dict())
    assert shallow(torch.randn(2,1,128,32)).shape == (2,128)


def test_alignment_model_passes_cnn_layers():
    assert Config().cnn_layers == 3
    model = build_model(Config(cnn_type='simple', cnn_pretrained=False, cnn_layers=2,
                               image_height=32, image_width=64))
    convs = [module for module in model.cnn.backbone if isinstance(module, nn.Conv2d)]
    assert len(convs) == 2
    assert model.cnn.channels == [32,64]
    output = model(torch.randn(2,1,32,64))
    assert output['local'].shape == (2,3,128)
    assert torch.isfinite(output['local']).all()


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

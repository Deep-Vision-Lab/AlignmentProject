"""Synthetic geometry and alignment checks for the variable-width experiment."""
from dataclasses import replace

import numpy as np
from PIL import Image
import pytest
import torch

from dataloader import collate_samples
from dataset import prepare_image
from dtw import soft_drop_dtw
from evaluate import source_interval
from losses import compute_loss
from model import AlignmentModel
from parameters import Config, validate_objective
from text_embedding import OrthogonalCharEmbedding


def _config():
    return Config(cnn_type='simple', cnn_pretrained=False, image_height=32,
                  window_width=16, window_stride=8, embedding_dim=128,
                  preserve_aspect_ratio=True, variable_width_images=True,
                  variable_length_windows=True, alignment_loss='soft_drop_dtw',
                  enable_padding_drop=True, enable_space_drop=True,
                  enable_positive_coverage_constraint=True, dtw_normalization='aligned_mean',
                  position_prior=0, transformer_dropout=0, local_dropout=0)


def test_aspect_geometry_and_short_boundary(tmp_path):
    a = np.full((20, 80), 255, np.uint8)
    b = np.full((20, 35), 20, np.uint8)
    a[8:14, 15:70] = 20
    b[8:14, 5:30] = 230
    for name, arr in [('a', a), ('b', b)]:
        Image.fromarray(arr).save(tmp_path / f'{name}.png')
    _, ta, ga = prepare_image(tmp_path/'a.png', (32, 64), crop='none', preserve_aspect_ratio=True,
                              window_width=16, window_stride=8)
    _, tb, gb = prepare_image(tmp_path/'b.png', (32, 64), crop='none', preserve_aspect_ratio=True,
                              window_width=16, window_stride=8)
    assert ta.shape[-1] == 128 and tb.shape[-1] == 56
    assert ga['valid_windows'] > gb['valid_windows']
    assert ga['window_starts'][-1] == 112
    assert gb['window_starts'][-1] == 40
    assert source_interval(gb['valid_windows']-1, gb, 16, 8) == [25., 35.]
    assert max(gb['space_confidence']) <= 1 and min(gb['space_confidence']) >= 0
    short = tmp_path/'short.png'
    Image.fromarray(np.full((20, 4), 255, np.uint8)).save(short)
    _, ts, gs = prepare_image(short, (32, 64), crop='none', preserve_aspect_ratio=True,
                              window_width=16, window_stride=8)
    assert ts.shape[-1] == 16 and gs['real_width'] == 6
    assert gs['artificial_pixel_columns'] == 10 and gs['valid_windows'] == 1
    assert source_interval(0, gs, 16, 8) == [0., 4.]


def test_ragged_padding_does_not_change_valid_context_or_gradients():
    torch.manual_seed(2)
    model = AlignmentModel(cnn_type='simple', pretrained_cnn=False, window_width=16, stride=8,
                           embedding_dim=128, variable_length_windows=True,
                           use_positional_encoding=True, local_dropout=0).eval()
    a, b = torch.rand(1, 32, 49), torch.rand(1, 32, 97)
    one = model([a], space_confidence=[[0]*6])
    together = model([a, b], space_confidence=[[0]*6, [0]*12])
    from window_geometry import window_starts
    preextracted = [torch.stack([x[..., start:start+16] for start in
                                window_starts(x.shape[-1], 16, 8, shift_last=True)]) for x in (a,b)]
    reused = model([a,b], space_confidence=[[0]*6,[0]*12], window_tensors=preextracted)
    for key in ('local', 'context', 'fused'):
        torch.testing.assert_close(one[key][0, :6], together[key][0, :6], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(reused[key], together[key], rtol=1e-5, atol=1e-5)
    assert together['token_valid'][0].tolist() == [True]*6+[False]*6
    assert together['physical_window_indices'][0, 6:].eq(-1).all()
    assert together['window_starts'][0, 6:].eq(-1).all()


def test_mandatory_pad_drop_and_gradients():
    costs = torch.tensor([[.1, 4.], [4., 4.], [4., .1], [-100., -100.]], requires_grad=True)
    confidence = torch.tensor([0., 1., 0., 0.])
    mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    energy, occupancy, info = soft_drop_dtw(costs, confidence, valid_windows=mask,
        gamma=.02, position_prior=0, text_drop_cost=4., return_details=True)
    baseline = soft_drop_dtw(costs[:3], confidence[:3], gamma=.02,
                             position_prior=0, text_drop_cost=4.)
    torch.testing.assert_close(energy, baseline)
    assert info['dropped_pad'] == 1 and float(info['dropped_real']) > .99
    assert occupancy.shape == (3, 2) and float(occupancy[1].sum()) < .01
    energy.backward()
    assert costs.grad[-1].eq(0).all() and torch.isfinite(costs.grad).all()
    assert costs.grad[:3].abs().sum() > 0


def test_match_preference_coverage_no_space_and_separated_regions():
    positive = torch.tensor([[.05, 5., 5., 5.], [5., .05, 5., 5.],
                             [5., 5., 5., 5.], [5., 5., .05, 5.], [5., 5., 5., .05]])
    negative = torch.full_like(positive, 4.)
    background = torch.tensor([0., 0., 1., 0., 0.])
    ep, occ, info = soft_drop_dtw(positive, background, gamma=.02,
        position_prior=0, text_drop_cost=4., return_details=True)
    en = soft_drop_dtw(negative, background, gamma=.02, position_prior=0, text_drop_cost=4.)
    assert ep < en and float(info['dropped_real']) > .99
    assert float(info['text_coverage']) > .99
    assert float(occ[0,0]) > .9 and float(occ[4,3]) > .9
    no_space = soft_drop_dtw(positive[[0,1,3,4]], torch.zeros(4),
                             gamma=.02, position_prior=0)
    assert torch.isfinite(no_space)
    bad = soft_drop_dtw(torch.full((5,4), 5.), torch.zeros(5), gamma=.02,
                        position_prior=0, text_drop_cost=.2, min_text_coverage=.9,
                        enforce_coverage=True)
    unconstrained = soft_drop_dtw(torch.full((5,4), 5.), torch.zeros(5), gamma=.02,
                                  position_prior=0, text_drop_cost=.2, enforce_coverage=False)
    assert bad >= unconstrained


def test_mixed_length_loss_and_bfloat16():
    c = _config()
    validate_objective(c)
    model = AlignmentModel(cnn_type='simple', pretrained_cnn=False, window_width=16, stride=8,
                           variable_length_windows=True, embedding_dim=128, local_dropout=0)
    text = OrthogonalCharEmbedding(128, seed=1234)
    lines = [torch.rand(1,32,49), torch.rand(1,32,81)]
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = model(lines, space_confidence=[[0]*6,[0]*10])
    loss, stats = compute_loss(output, ['ابتث', 'ابتثجح'], text, c)
    assert stats['evaluated'] == 2 and stats['valid_tokens'] == 16
    assert stats['lengths'] == [(6,4),(10,6)]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.cnn.projection[0].weight.grad.abs().sum() > 0
    assert model.transformer.layers[0].linear1.weight.grad.abs().sum() > 0


def test_fixed_size_drop_ablation_has_real_window_confidence():
    c = replace(_config(), preserve_aspect_ratio=False, variable_width_images=False,
                variable_length_windows=False)
    validate_objective(c)
    model = AlignmentModel(cnn_type='simple', pretrained_cnn=False, window_width=16,
                           stride=8, embedding_dim=128, local_dropout=0)
    images = torch.rand(2, 1, 32, 64)
    confidence = torch.zeros(2, 7)
    output = model(images, space_confidence=confidence)
    loss, stats = compute_loss(output, ['ابتث','باب'], OrthogonalCharEmbedding(128), c)
    assert stats['evaluated'] == 2 and torch.isfinite(loss)


def test_legacy_defaults_unchanged():
    c = Config()
    assert c.alignment_loss == 'soft_dtw' and not c.preserve_aspect_ratio
    validate_objective(c)
    with pytest.raises(ValueError, match='Aspect-ratio'):
        validate_objective(replace(c, preserve_aspect_ratio=True))


def _ddp_drop_worker(rank, rendezvous, output):
    from datetime import timedelta
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(11)
        shared = torch.randn(1, 7, 8, requires_grad=True)
        texts = ['باب'] if rank == 0 else ['سلام', 'قال']
        lengths = [4] if rank == 0 else [7, 5]
        vectors = shared.expand(len(texts), -1, -1)
        valid = torch.arange(7)[None] < torch.tensor(lengths)[:, None]
        cfg = Config(embedding_dim=8, alignment_loss='soft_drop_dtw',
                     enable_padding_drop=True, dtw_normalization='aligned_mean',
                     position_prior=0, enable_positive_coverage_constraint=True)
        loss, _ = compute_loss(dict(fused=vectors, fused_pre_l2=vectors, token_valid=valid,
                                    space_confidence=torch.zeros(len(texts), 7)),
                               texts, OrthogonalCharEmbedding(8, 4096), cfg,
                               distributed_statistics=True)
        loss.backward()
        dist.all_reduce(shared.grad); shared.grad.div_(2)
        if rank == 0:
            torch.save(shared.grad, output)
    finally:
        dist.destroy_process_group()


def test_two_rank_drop_gradient_matches_single_population(tmp_path):
    torch.multiprocessing.spawn(_ddp_drop_worker,
        args=(str(tmp_path/'rendezvous'), str(tmp_path/'gradient.pt')), nprocs=2, join=True)
    torch.manual_seed(11)
    shared = torch.randn(1, 7, 8, requires_grad=True)
    vectors = shared.expand(3, -1, -1)
    valid = torch.arange(7)[None] < torch.tensor([4,7,5])[:, None]
    cfg = Config(embedding_dim=8, alignment_loss='soft_drop_dtw',
                 enable_padding_drop=True, dtw_normalization='aligned_mean',
                 position_prior=0, enable_positive_coverage_constraint=True)
    loss, _ = compute_loss(dict(fused=vectors, fused_pre_l2=vectors, token_valid=valid,
                                space_confidence=torch.zeros(3, 7)),
                           ['باب','سلام','قال'], OrthogonalCharEmbedding(8,4096), cfg)
    loss.backward()
    torch.testing.assert_close(torch.load(tmp_path/'gradient.pt', weights_only=False),
                               shared.grad, atol=1e-5, rtol=1e-4)

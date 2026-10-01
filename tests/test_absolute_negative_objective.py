"""Aligned-NLL normalization and absolute transcript rejection contracts."""
from dataclasses import replace

import pytest
import torch

from dataset import generate_negative_transcripts
from dtw import soft_dtw
from losses import absolute_rejection_loss, compute_loss
from parameters import Config, validate_objective
from text_embedding import OrthogonalCharEmbedding, clean_letters
from test_dataset import make_synthetic
from train import main, load_checkpoint
import json


def config(**changes):
    return replace(Config(embedding_dim=8, negative_dtw_weight=.5, negative_count=4,
                          negative_loss_type='absolute', dtw_normalization='aligned_mean',
                          negative_target_min=1.5, negative_target_max=2.5,
                          hard_negative_k=2, sigreg_weight=0.), **changes)


def test_extended_arabic_inventory_and_metadata():
    source = 'بسم ٱلله الرحمن الرحيم'
    values, info, metadata = generate_negative_transcripts(
        source, count=4, sample_id='arabic-rare', seed=42, epoch=3,
        curriculum_epochs=0, return_metadata=True)
    assert (info['requested'], info['generated'], info['shortfall']) == (4, 4, 0)
    assert len(set(values)) == 4 and all(v != ''.join(clean_letters(source)) for v in values)
    assert all(entry['text'] == value and 0 <= entry['corruption_ratio'] <= 1
               for entry, value in zip(metadata, values))
    assert all(entry['changed_positions'] is None for entry in metadata if entry['operation'] != 'substitute')
    assert (values, info, metadata) == generate_negative_transcripts(
        source, count=4, sample_id='arabic-rare', seed=42, epoch=3,
        curriculum_epochs=0, return_metadata=True)
    later = generate_negative_transcripts(source, count=4, sample_id='arabic-rare',
                                          seed=42, epoch=9, curriculum_epochs=0)[1]
    assert info['severity'] == later['severity'] == .35
    empty, rejected = generate_negative_transcripts(source, count=4, vocabulary='ابج')
    assert empty == [] and rejected['rejections']['unsupported_positive'] == 4


def test_aligned_mean_is_per_visit_nll_not_t_plus_l():
    for shape in ((3, 5), (5, 9), (12, 7)):
        cost = torch.full(shape, 2.25, requires_grad=True)
        value, occupancy = soft_dtw(cost, gamma=.05, position_prior=0,
                                    vertical_penalty=0, horizontal_penalty=0,
                                    normalization='aligned_mean', return_occupancy=True)
        assert value.item() == pytest.approx(2.25, abs=1e-4)
        assert occupancy.shape == shape and occupancy.sum() >= max(shape)-1e-3
        value.backward()
        assert torch.isfinite(cost.grad).all()
    with torch.no_grad():
        assert soft_dtw(torch.ones(3, 5), normalization='aligned_mean').item() == pytest.approx(1.)


def test_absolute_direction_bad_positive_not_accepted():
    # Even if a negative already clears its absolute target, a bad positive
    # remains expensive; unlike the old rank-only condition, 5/5.5 is not good.
    easy_at_default, _, _ = absolute_rejection_loss(torch.tensor([5.5]), [1.],
        target_min=1.5, target_max=2.5, softness=.1, hard_k=1)
    assert (5. + .5 * easy_at_default).item() >= 5.
    confused, _, _ = absolute_rejection_loss(torch.tensor([.5]), [1.],
        target_min=2., target_max=2., softness=.1, hard_k=1)
    rejected, _, _ = absolute_rejection_loss(torch.tensor([4.]), [1.],
        target_min=2., target_max=2., softness=.1, hard_k=1)
    assert .1 + .5 * confused.item() > .8
    assert .1 + .5 * rejected.item() < .101
    positive = torch.tensor(5., requires_grad=True)
    negative = torch.tensor([5.5], requires_grad=True)
    rejection, _, _ = absolute_rejection_loss(negative, [1.], target_min=6.,
                                               target_max=6., softness=.1, hard_k=1)
    total = positive + .5 * rejection
    assert total.item() > 5.2  # Old margin .2 would have yielded no negative penalty.
    total.backward()
    assert positive.grad.item() > 0 and negative.grad.item() < 0
    high = torch.tensor([8.], requires_grad=True)
    easy, _, _ = absolute_rejection_loss(high, [1.], target_min=6.,
                                        target_max=6., softness=.1, hard_k=1)
    assert easy.item() < 1e-6


def test_hard_negative_selection_and_gradients():
    energies = torch.tensor([3., 1., 2.5, 1.5], requires_grad=True)
    value, selected, targets = absolute_rejection_loss(energies, [.2, .2, .2, .2],
        target_min=2., target_max=3., softness=.1, hard_k=2)
    assert set(selected.tolist()) == {1, 3}
    assert targets.tolist() == pytest.approx([2.2]*4)
    value.backward()
    assert energies.grad[1] < 0 and energies.grad[3] < 0
    assert energies.grad[0] == 0 and energies.grad[2] == 0


def test_optional_substitution_unlikelihood_uses_soft_positive_alignment():
    text = OrthogonalCharEmbedding(8, 4096)
    values, info, metadata = generate_negative_transcripts(
        'بسمٱلله', count=4, operations='substitute', sample_id='wrong',
        return_metadata=True)
    assert info['generated'] == 4 and all(item['changed_positions'] for item in metadata)
    vectors = torch.randn(1, 9, 8, requires_grad=True)
    cfg = config(wrong_letter_unlikelihood_weight=.05)
    loss, stats = compute_loss(dict(fused=vectors, fused_pre_l2=vectors,
        token_valid=torch.ones(1, 9, dtype=torch.bool)),
        ['بسمٱلله'], text, cfg, negative_texts=[values], negative_metadata=[metadata])
    assert stats['wrong_letter_loss'] is not None and stats['wrong_letter_loss'] > 0
    loss.backward()
    assert torch.isfinite(vectors.grad).all()


def test_rectangular_masked_absolute_loss_and_disabled_mode():
    torch.manual_seed(4)
    text = OrthogonalCharEmbedding(8, 4096)
    texts = ['بسمٱلله', 'الرحمن']
    negative = []
    metadata = []
    for i, source in enumerate(texts):
        values, info, entries = generate_negative_transcripts(source, count=4,
            sample_id=f'sample-{i}', return_metadata=True)
        assert info['generated'] == 4
        negative.append(values); metadata.append(entries)
    valid = torch.tensor([[True]*9+[False]*3, [True]*7+[False]*5])
    base = torch.randn(2, 12, 8)
    cfg = config()
    validate_objective(cfg)

    def run(raw):
        vectors = raw.detach().clone().requires_grad_()
        output = dict(fused=vectors, fused_pre_l2=vectors, token_valid=valid)
        loss, stats = compute_loss(output, texts, text, cfg,
            negative_texts=negative, negative_metadata=metadata)
        loss.backward()
        return loss.detach(), stats, vectors.grad

    a, stats, grad = run(base)
    changed = base.clone(); changed[~valid] = float('nan')
    b, other, changed_grad = run(changed)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(grad[valid], changed_grad[valid])
    assert torch.count_nonzero(grad[~valid]) == 0
    assert stats['ranking_candidates'] == 8
    assert stats['absolute_negative_loss'] is not None
    assert stats['hard_count'] == 4  # two per line
    assert torch.isfinite(grad).all()
    disabled, report = compute_loss(dict(fused=base, fused_pre_l2=base, token_valid=valid),
                                    texts, text, replace(cfg, negative_dtw_weight=0.))
    assert report['ranking_candidates'] == 0 and report['negative_objective'] is None
    assert torch.isfinite(disabled)


def _absolute_ddp_worker(rank, rendezvous, result_path):
    from datetime import timedelta
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(11)
        raw = torch.randn(1, 5, 8, requires_grad=True)
        texts = ['باب'] if rank == 0 else ['سلام', 'قال']
        negative = [['ابب', 'ببا']] if rank == 0 else [[], []]
        vectors = raw.expand(len(texts), -1, -1)
        cfg = config(negative_count=2, hard_negative_k=2)
        loss, _ = compute_loss(dict(fused=vectors, fused_pre_l2=vectors,
            token_valid=torch.ones(len(texts), 5, dtype=torch.bool)),
            texts, OrthogonalCharEmbedding(8, 4096), cfg,
            negative_texts=negative, distributed_statistics=True)
        loss.backward()
        dist.all_reduce(raw.grad); raw.grad.div_(2)
        if rank == 0:
            torch.save(raw.grad, result_path)
    finally:
        dist.destroy_process_group()


def test_two_rank_absolute_gradient_matches_single_population(tmp_path):
    torch.multiprocessing.spawn(_absolute_ddp_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'gradient.pt')),
        nprocs=2, join=True)
    torch.manual_seed(11)
    raw = torch.randn(1, 5, 8, requires_grad=True)
    vectors = raw.expand(3, -1, -1)
    cfg = config(negative_count=2, hard_negative_k=2)
    loss, _ = compute_loss(dict(fused=vectors, fused_pre_l2=vectors,
        token_valid=torch.ones(3, 5, dtype=torch.bool)),
        ['باب', 'سلام', 'قال'], OrthogonalCharEmbedding(8, 4096), cfg,
        negative_texts=[['ابب', 'ببا'], [], []])
    loss.backward()
    torch.testing.assert_close(torch.load(tmp_path / 'gradient.pt', weights_only=True),
                               raw.grad, atol=1e-5, rtol=1e-4)


def test_cpu_smoke_checkpoint_records_absolute_objective(tmp_path, capsys):
    root = make_synthetic(tmp_path / 'data')
    output = main(['--dataset', str(root), '--dataset-type', 'synthetic',
                   '--run-name', 'absolute_smoke', '--output-root', str(tmp_path / 'weights'),
                   '--epochs', '1', '--max-batches', '1', '--batch-size', '2',
                   '--num-workers', '0', '--cnn-type', 'simple', '--no-cnn-pretrained',
                   '--image-height', '32', '--image-width', '64', '--embedding-dim', '16',
                   '--transformer-layers', '1', '--transformer-heads', '1',
                   '--negative-dtw-weight', '0.5', '--negative-count', '4',
                   '--hard-negative-k', '2', '--negative-loss-type', 'absolute',
                   '--dtw-normalization', 'aligned_mean',
                   '--negative-curriculum-epochs', '0'])
    history = json.loads((output / 'history.json').read_text())
    assert history[0]['train']['absolute_negative_loss'] is not None
    assert history[0]['train']['negative_loss_type'] == 'absolute'
    assert history[0]['validation']['negative_target_success_rate'] is not None
    _, _, loaded, saved = load_checkpoint(output / 'checkpoint_best.pt')
    assert loaded.negative_loss_type == 'absolute' and loaded.dtw_normalization == 'aligned_mean'
    assert saved['config']['hard_negative_k'] == 2
    legacy = dict(saved)
    legacy['config'] = {k: v for k, v in saved['config'].items() if k not in {
        'negative_loss_type', 'dtw_normalization', 'hard_negative_k',
        'negative_target_min', 'negative_target_max', 'negative_softness',
        'ranking_aux_weight', 'wrong_letter_unlikelihood_weight'}}
    old_path = tmp_path / 'old_config.pt'; torch.save(legacy, old_path)
    _, _, old_config, _ = load_checkpoint(old_path)
    assert old_config.negative_loss_type == 'ranking' and old_config.dtw_normalization == 'legacy'
    console = capsys.readouterr()
    assert 'Negative loss type: absolute' in console.out
    assert 'absNeg=' in console.err and 'negCost=' in console.err

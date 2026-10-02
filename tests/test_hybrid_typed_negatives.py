"""Type-specific negative generation and supervision contracts."""
from collections import Counter
from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F

from dataset import generate_typed_negative_transcripts
from losses import (absolute_rejection_loss, compute_loss, order_rejection_loss,
                    typed_negative_objective, weighted_wrong_letter_unlikelihood)
from parameters import Config, negatives_enabled, validate_objective
from text_embedding import ARABIC_LETTERS, OrthogonalCharEmbedding, clean_letters
from test_dataset import make_synthetic
from train import main, load_checkpoint


def settings(**changes):
    return replace(Config(embedding_dim=8, negative_count=4,
                          negative_loss_type='hybrid_typed', negative_dtw_weight=0.,
                          negative_curriculum_epochs=0, negative_target_mode='uniform_ratio',
                          dtw_normalization='aligned_mean', sigreg_weight=0.), **changes)


def text_encoder(*sources):
    encoder = OrthogonalCharEmbedding(8, 4096)
    encoder.letter_evidence_prior = {
        'vocabulary': list(dict.fromkeys(ARABIC_LETTERS + ''.join(
            ''.join(clean_letters(source)) for source in sources)))}
    return encoder


def test_four_typed_negatives_are_distinct_supported_and_reproducible():
    source = 'ابتثجحخ'
    result = generate_typed_negative_transcripts(source, sample_id='line-1', epoch=3)
    texts, info, entries = result
    assert result == generate_typed_negative_transcripts(source, sample_id='line-1', epoch=3)
    assert (info['requested'], info['generated'], info['shortfall']) == (4, 4, 0)
    assert [entry['negative_type'] for entry in entries] == [
        'strong_global', 'strong_global', 'local_substitution', 'order_negative']
    assert len(set(texts)) == 4 and source not in texts
    for candidate, entry in zip(texts, entries):
        assert candidate == ''.join(clean_letters(candidate)) == entry['text']
    for entry in entries[:2]:
        assert entry['corruption_ratio'] >= .70
        assert Counter(entry['text']) != Counter(source)
        assert entry['operation'] in {'strong_substitute', 'mixed_substitute',
                                      'multi_block_substitute'}
    local = entries[2]
    assert local['changed_positions'] == [i for i, (a, b) in enumerate(zip(source, local['text'])) if a != b]
    assert local['original_letters'] == [source[i] for i in local['changed_positions']]
    assert local['replacement_letters'] == [local['text'][i] for i in local['changed_positions']]
    order = entries[3]
    assert Counter(order['text']) == Counter(source)
    assert order['corruption_ratio'] == 0 and order['changed_positions'] is None
    assert order['order_change_ratio'] > 0


def test_short_and_duration_only_lines_report_finite_shortfall():
    for source in ('ا', 'اااا'):
        texts, info, entries = generate_typed_negative_transcripts(source, sample_id=source)
        assert len(texts) == len(entries) == info['generated']
        assert info['requested'] == info['generated'] + info['shortfall']
        assert info['by_type']['order_negative']['shortfall'] == 1
        assert info['attempts'] < 200


def test_local_metadata_and_probability_gradient_only_for_changed_column(monkeypatch):
    source, candidate = 'ابجد', 'ابسد'
    metadata = dict(text=candidate, negative_type='local_substitution',
                    operation='local_substitute', changed_positions=[2],
                    original_letters=['ج'], replacement_letters=['س'], corruption_ratio=.25)
    cfg = settings(strong_negative_weight=0., order_negative_weight=0.)
    encoder = text_encoder(source)
    visual = torch.randn(1, 5, 8, requires_grad=True)
    occupancy = torch.zeros(5, 4)
    occupancy[2, 2] = 1.
    def no_whole_transcript(*args, **kwargs):
        raise AssertionError('Local substitution must never align/reject the complete negative transcript')
    monkeypatch.setattr('losses._line_loss', no_whole_transcript)
    losses, info = typed_negative_objective(visual, [source], [[candidate]], [[metadata]],
        encoder, cfg, torch.ones(1, 5, dtype=torch.bool), [visual.sum() * 0], [occupancy])
    assert info['strong_count'] == info['order_count'] == 0
    assert info['changed_positions_count'] == info['wrong_count'] == 1
    losses['wrong'].backward()
    assert torch.isfinite(visual.grad).all()
    assert torch.count_nonzero(visual.grad[0, [0, 1, 3, 4]]) == 0
    assert visual.grad[0, 2].abs().sum() > 0

    logits = torch.tensor([[0., .5, -.3]], requires_grad=True)
    logp = F.log_softmax(logits, dim=-1)
    wrong_loss, wrong_p, _ = weighted_wrong_letter_unlikelihood(
        logp, torch.ones(1), wrong_index=1, correct_index=0)
    wrong_loss.backward()
    assert wrong_p.item() > 0 and logits.grad[0, 1] > 0
    p = torch.tensor(.4, requires_grad=True)
    (-torch.log1p(-p)).backward()
    assert p.grad > 0


def test_energy_gradient_directions_and_legacy_ablation():
    positive = torch.tensor(1., requires_grad=True)
    strong = torch.tensor([1.], requires_grad=True)
    order = torch.tensor(1.1, requires_grad=True)
    rejection, _, _ = absolute_rejection_loss(strong, [.75], target_min=2.,
        target_max=3., softness=.1, hard_k=1)
    loss = positive + .5 * rejection + .25 * order_rejection_loss(positive, order, .2)
    loss.backward()
    assert positive.grad.item() == pytest.approx(1.)
    assert strong.grad.item() < 0
    assert order.grad.item() < 0
    assert negatives_enabled(settings())
    assert not negatives_enabled(settings(strong_negative_weight=0., wrong_letter_weight=0.,
                                           order_negative_weight=0.))
    validate_objective(settings())
    with pytest.raises(ValueError, match='Typed negative counts'):
        validate_objective(settings(negative_count=3))
    with pytest.raises(ValueError, match='legacy negative weights'):
        validate_objective(settings(negative_dtw_weight=.5))


def test_mixed_lengths_masked_padding_and_component_ablation():
    sources = ['ابتثجحخ', 'بسمٱلله']
    generated = [generate_typed_negative_transcripts(source, sample_id=str(i))
                 for i, source in enumerate(sources)]
    assert all(item[1]['generated'] == 4 for item in generated)
    encoder = text_encoder(*sources)
    cfg = settings()
    valid = torch.tensor([[True] * 12, [True] * 9 + [False] * 3])
    initial = torch.randn(2, 12, 8)
    negative_texts = [item[0] for item in generated]
    negative_metadata = [item[2] for item in generated]
    def run(raw, config=cfg):
        vectors = raw.detach().clone().requires_grad_()
        loss, stats = compute_loss(dict(fused=vectors, fused_pre_l2=vectors,
            token_valid=valid), sources, encoder, config,
            negative_texts=negative_texts, negative_metadata=negative_metadata)
        loss.backward()
        return loss.detach(), stats, vectors.grad
    first, stats, gradient = run(initial)
    changed = initial.clone(); changed[1, 9:] = float('nan')
    second, _, changed_gradient = run(changed)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(gradient[valid], changed_gradient[valid])
    assert torch.count_nonzero(gradient[~valid]) == 0
    assert torch.isfinite(first) and torch.isfinite(gradient).all()
    assert (stats['strong_count'], stats['wrong_count'], stats['order_count']) == (4, 4, 2)
    positive_only = settings(strong_negative_weight=0., wrong_letter_weight=0.,
                             order_negative_weight=0.)
    _, disabled, _ = run(initial, positive_only)
    assert disabled['strong_count'] == disabled['wrong_count'] == disabled['order_count'] == 0
    no_negatives, disabled_without_candidates = compute_loss(
        dict(fused=initial, fused_pre_l2=initial, token_valid=valid), sources,
        encoder, positive_only)
    assert torch.isfinite(no_negatives)
    assert disabled_without_candidates['global_ranking_candidates'] == 0


def test_typed_cpu_training_smoke_logs_and_checkpoint(tmp_path, capsys):
    root = make_synthetic(tmp_path / 'data')
    output = main(['--dataset', str(root), '--dataset-type', 'synthetic',
                   '--run-name', 'typed_smoke', '--output-root', str(tmp_path / 'weights'),
                   '--epochs', '1', '--max-batches', '1', '--batch-size', '2',
                   '--num-workers', '0', '--cnn-type', 'simple', '--no-cnn-pretrained',
                   '--image-height', '32', '--image-width', '64', '--embedding-dim', '16',
                   '--transformer-layers', '1', '--transformer-heads', '1',
                   '--negative-loss-type', 'hybrid_typed', '--negative-count', '4',
                   '--negative-curriculum-epochs', '0', '--negative-target-mode', 'uniform_ratio',
                   '--dtw-normalization', 'aligned_mean'])
    history = json.loads((output / 'history.json').read_text())
    train = history[0]['train']; validation = history[0]['validation']
    assert train['strong_negative_evaluated'] > 0
    assert train['changed_positions_count'] > 0
    assert train['order_negative_evaluated'] > 0
    assert train['strong_neg_loss'] is not None
    assert validation['strong_neg_loss'] is not None
    _, _, saved_config, saved = load_checkpoint(output / 'checkpoint_best.pt')
    assert saved_config.negative_loss_type == 'hybrid_typed'
    assert saved_config.negative_dtw_weight == 0.
    assert saved['negative_target_reference']['effective_alphabet_size'] > 0
    stdout, stderr = capsys.readouterr()
    assert 'Typed negatives: 2 strong + 1 local substitution + 1 order' in stdout
    assert 'STRONG NEGATIVES' in stdout and 'LOCAL SUBSTITUTION' in stdout
    assert 'ORDER NEGATIVE' in stdout and 'GENERATION' in stdout
    assert 'sNeg=' in stderr and 'wrongL=' in stderr and 'ordL=' in stderr


def _typed_ddp_worker(rank, rendezvous, result_path):
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        source = ['ابتثجحخ', 'اااا'][rank]
        values, _, metadata = generate_typed_negative_transcripts(source, sample_id=str(rank))
        torch.manual_seed(17)
        raw = torch.randn(1, 9, 8, requires_grad=True)
        loss, _ = compute_loss(dict(fused=raw, fused_pre_l2=raw,
            token_valid=torch.ones(1, 9, dtype=torch.bool)), [source],
            text_encoder('ابتثجحخ', 'اااا'), settings(), negative_texts=[values],
            negative_metadata=[metadata], distributed_statistics=True)
        loss.backward()
        dist.all_reduce(raw.grad); raw.grad.div_(2)
        if rank == 0:
            torch.save(raw.grad, result_path)
    finally:
        dist.destroy_process_group()


def test_two_rank_typed_gradient_matches_single_population(tmp_path):
    torch.multiprocessing.spawn(_typed_ddp_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'gradient.pt')),
        nprocs=2, join=True)
    sources = ['ابتثجحخ', 'اااا']
    generated = [generate_typed_negative_transcripts(source, sample_id=str(i))
                 for i, source in enumerate(sources)]
    torch.manual_seed(17)
    raw = torch.randn(1, 9, 8, requires_grad=True)
    vectors = raw.expand(2, -1, -1)
    loss, _ = compute_loss(dict(fused=vectors, fused_pre_l2=vectors,
        token_valid=torch.ones(2, 9, dtype=torch.bool)), sources,
        text_encoder(*sources), settings(), negative_texts=[item[0] for item in generated],
        negative_metadata=[item[2] for item in generated])
    loss.backward()
    torch.testing.assert_close(torch.load(tmp_path / 'gradient.pt', weights_only=True),
                               raw.grad, atol=1e-5, rtol=1e-4)

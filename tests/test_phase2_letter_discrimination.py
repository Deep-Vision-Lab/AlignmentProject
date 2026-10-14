"""Stage 2–4 regression tests; no full training or learned classifier head."""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from dataset import generate_typed_negative_transcripts
from evaluation_utils import EvaluationSession
from losses import compute_loss
from parameters import Config, validate_objective
from text_embedding import ARABIC_LETTERS, OrthogonalCharEmbedding, clean_letters
from train import build_model, load_checkpoint, main
from test_evaluation_notebook import evaluation_fixture, notebook_source
from test_dataset import make_synthetic


def test_zero_local_weight_matches_previous_positive_objective():
    torch.manual_seed(2)
    fused = torch.nn.functional.normalize(torch.randn(2, 5, 128, requires_grad=True), dim=-1)
    output = dict(fused=fused, fused_pre_l2=fused,
                  token_valid=torch.tensor([[True, True, True, False, False],
                                            [True, True, True, True, True]]))
    text = OrthogonalCharEmbedding(vocab_size=4096)
    config = Config(sigreg_weight=0, local_positive_dtw_weight=0)
    baseline, old_stats = compute_loss(output, ['باب', 'سلام'], text, config)
    output['local'] = torch.randn_like(fused)
    current, new_stats = compute_loss(output, ['باب', 'سلام'], text, config)
    torch.testing.assert_close(current, baseline, rtol=0, atol=0)
    assert old_stats['total'] == new_stats['total']
    assert new_stats['local_positive_dtw'] is None


def test_local_auxiliary_reaches_cnn_but_not_transformer():
    torch.manual_seed(3)
    config = Config(cnn_type='simple', cnn_pretrained=False, image_height=32, image_width=64,
                    embedding_dim=32, positive_dtw_weight=0, local_positive_dtw_weight=.3,
                    sigreg_weight=0)
    model = build_model(config)
    output = model(torch.randn(2, 1, 32, 64))
    text = OrthogonalCharEmbedding(32, 4096)
    loss, stats = compute_loss(output, ['باب', 'سلام'], text, config)
    assert stats['local_positive_dtw'] is not None
    assert stats['weighted_local_positive_dtw'] == pytest.approx(.3 * stats['local_positive_dtw'])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cnn.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.transformer.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.fusion.parameters())


def test_local_loss_excludes_padded_windows_and_gradients():
    torch.manual_seed(4)
    config = Config(sigreg_weight=0, local_positive_dtw_weight=.3)
    text = OrthogonalCharEmbedding(vocab_size=4096)
    valid = torch.tensor([[True, True, True, False, False],
                          [True, True, True, True, False]])
    local = torch.randn(2, 5, 128, requires_grad=True)
    fused = torch.randn(2, 5, 128, requires_grad=True)
    output = dict(local=local, fused=fused, fused_pre_l2=fused, token_valid=valid)
    loss, stats = compute_loss(output, ['باب', 'سلام'], text, config)
    changed = dict(output, local=local.detach().clone().masked_fill(~valid[..., None], 1e6).requires_grad_(),
                   fused=fused.detach().clone().masked_fill(~valid[..., None], -1e6).requires_grad_())
    changed_loss, changed_stats = compute_loss(changed, ['باب', 'سلام'], text, config)
    torch.testing.assert_close(loss, changed_loss)
    assert stats['evaluated'] == stats['local_evaluated'] == 2
    assert changed_stats['local_positive_dtw'] == pytest.approx(stats['local_positive_dtw'])
    loss.backward()
    assert torch.count_nonzero(local.grad[~valid]) == 0
    assert torch.count_nonzero(fused.grad[~valid]) == 0


def test_typed_counts_and_metadata_are_reused():
    positive = 'ابتثجحخدذرزسشصضطظعغ'
    candidates, stats, metadata = generate_typed_negative_transcripts(
        positive, strong_count=2, local_count=1, order_count=1,
        seed=42, sample_id='phase2', epoch=1)
    assert stats['generated'] == len(candidates) == 4
    assert [item['negative_type'] for item in metadata] == [
        'strong_global', 'strong_global', 'local_substitution', 'order_negative']
    original = clean_letters(positive)
    local = metadata[2]
    assert len(clean_letters(local['text'])) == len(original)
    assert all(original[position] == letter for position, letter in
               zip(local['changed_positions'], local['original_letters']))
    assert all(clean_letters(local['text'])[position] == letter for position, letter in
               zip(local['changed_positions'], local['replacement_letters']))
    assert candidates[3] != positive
    assert sorted(clean_letters(candidates[3])) == sorted(original)


def test_typed_negatives_supervise_fused_not_local():
    source = 'ابتثجحخدذر'
    negatives, stats, metadata = generate_typed_negative_transcripts(
        source, strong_count=2, local_count=1, order_count=1, sample_id='fused-only')
    assert stats['generated'] == 4
    config = Config(embedding_dim=8, alphabet_inventory=ARABIC_LETTERS,
                    negative_loss_type='hybrid_typed', negative_count=4,
                    negative_target_mode='uniform_ratio', negative_curriculum_epochs=0,
                    dtw_normalization='aligned_mean', positive_dtw_weight=0,
                    local_positive_dtw_weight=0)
    fused = torch.randn(1, 12, 8, requires_grad=True)
    local = torch.randn(1, 12, 8, requires_grad=True)
    loss, batch = compute_loss(dict(fused=fused, local=local, fused_pre_l2=fused,
                                    token_valid=torch.ones(1, 12, dtype=torch.bool)),
                               [source], OrthogonalCharEmbedding(8, 4096), config,
                               negative_texts=[negatives], negative_metadata=[metadata])
    assert torch.isfinite(loss)
    assert batch['strong_count'] == 2 and batch['local_count'] == batch['order_count'] == 1
    loss.backward()
    assert fused.grad is not None and fused.grad.abs().sum() > 0
    assert local.grad is None or local.grad.abs().sum() == 0


def test_letter_space_probabilities_and_soft_occupancy(evaluation_fixture):
    from evaluation_utils import letter_space_diagnostics, plot_letter_probability_heatmaps
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False)
    side = session.pairs[0]['sides'][0]
    report = letter_space_diagnostics(session, side)
    assert report['fused_soft_occupancy'].shape == (
        len(report['physical_window_indices']), len(report['transcript_letters']))
    assert (report['fused_soft_occupancy'] >= 0).all()
    for name in ('local', 'context', 'fused'):
        row = report['representations'][name]
        assert row['probabilities'].shape == (len(report['physical_window_indices']), len(report['alphabet']))
        np.testing.assert_allclose(row['probabilities'].sum(axis=1), 1, atol=1e-5)
        assert len(row['windows']) == len(report['physical_window_indices'])
        assert 0 <= row['top1_correct_fraction'] <= 1
        assert np.isfinite(row['positive_dtw_cost'])
    fig = plot_letter_probability_heatmaps(report)
    assert len(fig.axes) == 4  # 3 representations plus shared colorbar
    plt.close(fig)


def test_stage2_notebook_cell_reuses_phase1_fixed_pairs(evaluation_fixture, monkeypatch):
    import IPython.display
    monkeypatch.setattr(IPython.display, 'display', lambda *args, **kwargs: None)
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False)
    source = notebook_source('# STAGE 2 CONFIGURATION')
    scope = dict(phase1_session=session, phase1_positive=[session.pairs[0]])
    try:
        exec(compile(source, 'stage2-notebook-cell', 'exec'), scope)
        assert len(scope['stage2_reports']) == 2
        assert len(scope['stage2_rows']) == 6
        assert {row['representation'] for row in scope['stage2_rows']} == {'local', 'context', 'fused'}
    finally:
        plt.close('all')


def test_old_checkpoint_defaults_local_weight_to_zero(evaluation_fixture, tmp_path):
    root, checkpoint = evaluation_fixture
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved['config'].pop('local_positive_dtw_weight', None)
    legacy = tmp_path / 'old.pt'
    torch.save(saved, legacy)
    _, _, config, _ = load_checkpoint(legacy, 'cpu')
    assert config.local_positive_dtw_weight == 0
    with pytest.raises(ValueError, match='local_positive_dtw_weight'):
        validate_objective(replace(config, local_positive_dtw_weight=-.1))


@pytest.mark.parametrize('typed', [False, True])
def test_one_batch_stage3_stage4_smoke_and_saved_test_evaluation(tmp_path, typed):
    from evaluation_utils import letter_space_diagnostics
    root = make_synthetic(tmp_path / 'data')
    arguments = ['--dataset', str(root), '--dataset-type', 'synthetic',
                 '--run-name', 'typed' if typed else 'local',
                 '--output-root', str(tmp_path / 'weights'), '--epochs', '1',
                 '--max-batches', '1', '--batch-size', '2', '--num-workers', '0',
                 '--cnn-type', 'simple', '--no-cnn-pretrained',
                 '--image-height', '32', '--image-width', '64',
                 '--local-positive-dtw-weight', '0.30']
    if typed:
        arguments += ['--negative-loss-type', 'hybrid_typed', '--negative-count', '4',
                      '--strong-negative-count', '2', '--local-substitution-count', '1',
                      '--order-negative-count', '1', '--negative-dtw-weight', '0',
                      '--ranking-aux-weight', '0', '--wrong-letter-unlikelihood-weight', '0',
                      '--dtw-normalization', 'aligned_mean',
                      '--negative-target-mode', 'uniform_ratio',
                      '--negative-curriculum-epochs', '0']
    output = main(arguments)
    assert (output / 'checkpoint_best.pt').exists()
    model, text, config, saved = load_checkpoint(output / 'checkpoint_best.pt')
    assert config.local_positive_dtw_weight == .3
    train_stats = saved['metrics']['train']
    assert train_stats['local_positive_dtw'] is not None
    if typed:
        assert train_stats['corruption']['strong_global_generated'] > 0
        assert train_stats['corruption']['local_substitution_generated'] > 0
        assert train_stats['corruption']['order_negative_generated'] > 0
    session = EvaluationSession(output / 'checkpoint_best.pt', root, device='cpu',
                                construct_negatives=False)
    side = session.view.records[0]['sides'][0]
    report = letter_space_diagnostics(session, side)
    assert all(name in report['representations'] for name in ('local', 'context', 'fused'))


def _local_ddp_worker(rank, rendezvous, result_path):
    from datetime import timedelta
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(91)
        fused = torch.randn(1, 5, 8, requires_grad=True)
        local = torch.randn(1, 5, 8, requires_grad=True)
        config = Config(embedding_dim=8, local_positive_dtw_weight=.3)
        text = OrthogonalCharEmbedding(8, 4096)
        loss, _ = compute_loss(dict(fused=fused, local=local, fused_pre_l2=fused,
                                    token_valid=torch.ones(1, 5, dtype=torch.bool)),
                               ['سلام' if rank == 0 else ''], text, config,
                               distributed_statistics=True)
        loss.backward()
        dist.all_reduce(fused.grad); fused.grad.div_(2)
        dist.all_reduce(local.grad); local.grad.div_(2)
        if rank == 0:
            torch.save(dict(fused=fused.grad, local=local.grad), result_path)
    finally:
        dist.destroy_process_group()


def test_local_auxiliary_ddp_scaling_with_unequal_valid_lines(tmp_path):
    torch.multiprocessing.spawn(_local_ddp_worker,
        args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'gradient.pt')),
        nprocs=2, join=True)
    torch.manual_seed(91)
    fused = torch.randn(1, 5, 8, requires_grad=True)
    local = torch.randn(1, 5, 8, requires_grad=True)
    config = Config(embedding_dim=8, local_positive_dtw_weight=.3)
    loss, _ = compute_loss(dict(fused=fused, local=local, fused_pre_l2=fused,
                                token_valid=torch.ones(1, 5, dtype=torch.bool)),
                           ['سلام'], OrthogonalCharEmbedding(8, 4096), config)
    loss.backward()
    distributed = torch.load(tmp_path / 'gradient.pt', weights_only=True)
    torch.testing.assert_close(distributed['fused'], fused.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(distributed['local'], local.grad, rtol=1e-5, atol=1e-6)

"""Offline submission, resolved negative settings, and valid-window loss checks."""
import os
import json
from pathlib import Path
import subprocess

import pytest
import torch

from dataset import generate_negative_transcripts
from losses import compute_loss
from parameters import Config
from text_embedding import OrthogonalCharEmbedding, clean_letters
from train import _batch_postfix, main
from test_dataset import make_synthetic


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = 'substitute,adjacent,blocks,words,shift,shuffle'


def test_offline_wrapper_forwards_all_negative_flags(tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    sbatch = fake_bin / 'sbatch'
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=f'{fake_bin}:{os.environ["PATH"]}',
               RUN_NAME='real_sum_gate_neg4_w05', FUSION_MODE='sum', USE_GATED_FUSION='1')
    flags = ['--negative-dtw-weight', '0.5', '--negative-count', '4',
             '--negative-margin', '0.2', '--negative-severity', '0.35',
             '--negative-operations', OPERATIONS, '--negative-warmup-epochs', '0',
             '--negative-curriculum-epochs', '5', '--negative-seed', '42']
    completed = subprocess.run(['bash', 'scripts/train/train.sh', 'real', *flags],
                               cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    submitted = completed.stdout.splitlines()
    assert '--job-name=real_sum_gate_neg4_w05' in submitted
    assert 'scripts/train/train.sbatch' in submitted
    assert submitted[-len(flags):] == flags
    assert submitted[-len(flags)-4:-len(flags)] == [
        '--dataset', str(ROOT / 'DataSet/ArabicDataset'), '--dataset-type', 'real']
    assert (ROOT / 'scripts/train/train.sbatch').read_text().count('"$@"') == 1


def test_invalid_concat_gate_fails_before_submission(tmp_path):
    env = dict(os.environ, FUSION_MODE='concat', USE_GATED_FUSION='1')
    completed = subprocess.run(['bash', 'scripts/train/train.sh', 'real'],
                               cwd=ROOT, env=env, text=True, capture_output=True)
    assert completed.returncode != 0
    assert 'concat + gated fusion is unsupported' in completed.stderr


def test_negative_environment_variable_is_rejected_instead_of_ignored():
    completed = subprocess.run(['bash', 'scripts/train/train.sh', 'real'],
                               cwd=ROOT, env=dict(os.environ, NEGATIVE_DTW_WEIGHT='0.5'),
                               text=True, capture_output=True)
    assert completed.returncode != 0
    assert 'pass --negative-dtw-weight instead' in completed.stderr


def test_cli_resolves_negative_controls_and_rejects_silent_disable(monkeypatch):
    def inspect(config):
        assert config.negative_dtw_weight == 0.5
        assert config.negative_count == 4
        assert config.negative_margin == 0.2
        assert config.negative_severity == 0.35
        assert config.negative_operations == OPERATIONS
        assert (config.negative_warmup_epochs, config.negative_curriculum_epochs,
                config.negative_seed) == (0, 5, 42)
        raise RuntimeError('parsed')
    monkeypatch.setattr('train.validate_objective', inspect)
    with pytest.raises(RuntimeError, match='parsed'):
        main(['--dataset', 'unused', '--run-name', 'unused', '--negative-dtw-weight', '0.5',
              '--negative-count', '4', '--negative-margin', '0.2',
              '--negative-severity', '0.35', '--negative-operations', OPERATIONS,
              '--negative-warmup-epochs', '0', '--negative-curriculum-epochs', '5',
              '--negative-seed', '42'])
    with pytest.raises(ValueError, match='negative DTW weight is 0'):
        main(['--dataset', 'unused', '--run-name', 'unused', '--negative-count', '4'])


def test_four_negatives_margin_and_padding_exclusion():
    positive = 'السَّلام عليكم ورحمة الله'
    negatives, details = generate_negative_transcripts(
        positive, count=4, operations=OPERATIONS, severity=0.35,
        seed=42, sample_id='line-a', epoch=1)
    assert details['generated'] == 4 and details['shortfall'] == 0
    assert len(set(negatives)) == 4
    assert all(value != ''.join(clean_letters(positive)) for value in negatives)

    config = Config(embedding_dim=8, negative_dtw_weight=0.5,
                    negative_count=4, negative_margin=10.0)
    text = OrthogonalCharEmbedding(8, 4096)
    base = torch.randn(1, 18, 8)
    valid = torch.tensor([[True] * 16 + [False] * 2])

    def calculate(padded):
        vectors = padded.detach().clone().requires_grad_()
        output = dict(fused=vectors, fused_pre_l2=vectors, token_valid=valid)
        loss, stats = compute_loss(output, [positive], text, config,
                                   negative_texts=[negatives])
        loss.backward()
        return loss, stats, vectors.grad

    first_loss, first_stats, first_grad = calculate(base)
    changed = base.clone()
    changed[:, 16:] = float('nan')
    second_loss, second_stats, second_grad = calculate(changed)
    torch.testing.assert_close(first_loss, second_loss)
    torch.testing.assert_close(first_grad[:, :16], second_grad[:, :16])
    assert torch.count_nonzero(first_grad[:, 16:]) == 0
    assert torch.isfinite(first_loss) and torch.isfinite(first_grad).all()
    assert first_grad[:, :16].abs().sum() > 0
    assert first_stats['ranking_candidates'] == 4
    assert first_stats['hard_negatives'] == 4
    assert first_stats['negative_cost_min'] <= first_stats['negative_cost_max']
    assert first_stats['negative_dtw'] is not None
    assert _batch_postfix(first_stats)['negCost'] != 'n/a'
    assert _batch_postfix(first_stats)['rank'] != 'n/a'
    assert second_stats['negative_cost_sum'] == pytest.approx(first_stats['negative_cost_sum'])


def test_one_batch_cpu_run_logs_resolved_negative_settings(tmp_path, capsys):
    root = make_synthetic(tmp_path / 'data')
    output = main(['--dataset', str(root), '--dataset-type', 'synthetic',
                   '--run-name', 'neg4_smoke', '--output-root', str(tmp_path / 'weights'),
                   '--epochs', '1', '--max-batches', '1', '--batch-size', '2',
                   '--num-workers', '0', '--cnn-type', 'simple', '--no-cnn-pretrained',
                   '--image-height', '32', '--image-width', '64', '--embedding-dim', '16',
                   '--transformer-layers', '1', '--transformer-heads', '1',
                   '--fusion-mode', 'sum', '--use-gated-fusion', '1',
                   '--negative-dtw-weight', '0.5', '--negative-count', '4',
                   '--negative-margin', '0.2', '--negative-severity', '0.35',
                   '--negative-operations', OPERATIONS, '--negative-warmup-epochs', '0'])
    console = capsys.readouterr()
    assert 'Negative DTW weight: 0.5' in console.out
    assert 'Negative transcripts: ENABLED' in console.out
    assert 'Negative count: 4' in console.out
    assert 'Negative margin: 0.2' in console.out
    assert f'Negative operations: {OPERATIONS}' in console.out
    assert 'Fusion mode: sum' in console.out and 'Gated fusion: enabled' in console.out
    assert 'neg=n/a' not in console.err
    history = json.loads((output / 'history.json').read_text())
    train = history[0]['train']
    assert train['negative_weight'] == 0.5 and train['ranking_candidates'] > 0
    assert train['negative_objective'] is not None and train['negative_cost_mean'] is not None
    assert train['negative_cost_min'] <= train['negative_cost_max']
    assert (output / 'checkpoint_latest.pt').exists()

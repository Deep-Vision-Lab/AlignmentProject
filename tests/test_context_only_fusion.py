"""CONTEXT uses the existing FUSED loss route without any external fusion."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch.nn import functional as F

from dataset import generate_typed_negative_transcripts
from losses import compute_loss
from parameters import Config, validate_objective
from text_embedding import ARABIC_LETTERS, OrthogonalCharEmbedding, clean_letters
from train import build_model, load_checkpoint, main
from test_dataset import make_synthetic


ROOT = Path(__file__).resolve().parents[1]


def compact(**changes):
    return replace(Config(cnn_type='simple', cnn_pretrained=False, cnn_layers=2,
                          image_height=32, image_width=64, embedding_dim=8,
                          transformer_layers=1, transformer_heads=1, fusion_mode='context',
                          local_positive_dtw_weight=0.), **changes)


def test_context_equals_normalized_transformer_and_receives_gradients():
    model = build_model(compact())
    assert model.fusion is None
    assert not any(name.startswith('fusion.') for name, _ in model.named_parameters())
    image = torch.randn(2, 1, 32, 64)
    output = model(image)
    torch.testing.assert_close(output['fused'], F.normalize(output['context'].float(), dim=-1))
    assert output['fusion_gate'] is None
    encoder = OrthogonalCharEmbedding(8, 4096)
    loss, stats = compute_loss(output, ['باب', 'سلام'], encoder, compact())
    loss.backward()
    assert torch.isfinite(loss) and stats['alignment_representation'] == 'context'
    assert stats['fused_positive_dtw'] is None and stats['context_positive_dtw'] is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cnn.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.parameters())


def test_typed_negatives_use_context_and_exclude_invalid_windows():
    config = compact(negative_loss_type='hybrid_typed', negative_count=4,
                     negative_curriculum_epochs=0, negative_target_mode='uniform_ratio',
                     dtw_normalization='aligned_mean')
    validate_objective(config)
    candidates, info, metadata = generate_typed_negative_transcripts('ابتثجحخ', sample_id='context')
    assert info['generated'] == 4
    encoder = OrthogonalCharEmbedding(8, 4096)
    encoder.letter_evidence_prior = {
        'vocabulary': list(dict.fromkeys(ARABIC_LETTERS + ''.join(
            ''.join(clean_letters(text)) for text in ['ابتثجحخ', *candidates])))}
    valid = torch.tensor([[True] * 9 + [False] * 3])
    source = torch.randn(1, 12, 8)

    def calculate(values):
        vectors = values.detach().clone().requires_grad_()
        output = dict(fused=vectors, fused_pre_l2=vectors, token_valid=valid)
        loss, stats = compute_loss(output, ['ابتثجحخ'], encoder, config,
                                   negative_texts=[candidates], negative_metadata=[metadata])
        loss.backward()
        return loss.detach(), stats, vectors.grad

    first, stats, grad = calculate(source)
    changed = source.clone(); changed[:, 9:] = float('nan')
    second, _, changed_grad = calculate(changed)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(grad[:, :9], changed_grad[:, :9])
    assert torch.count_nonzero(grad[:, 9:]) == 0
    assert stats['strong_count'] and stats['wrong_count'] and stats['order_count']


def test_context_checkpoint_smoke_and_old_fused_compatibility(tmp_path):
    root = make_synthetic(tmp_path / 'data')
    args = ['--dataset', str(root), '--dataset-type', 'synthetic', '--output-root', str(tmp_path / 'weights'),
            '--epochs', '1', '--max-batches', '1', '--batch-size', '2', '--num-workers', '0',
            '--cnn-type', 'simple', '--no-cnn-pretrained', '--cnn-layers', '2',
            '--image-height', '32', '--image-width', '64', '--embedding-dim', '8',
            '--transformer-layers', '1', '--transformer-heads', '1']
    output = main([*args, '--run-name', 'context-smoke', '--fusion-mode', 'context',
                   '--local-positive-dtw-weight', '0'])
    restored, _, config, saved = load_checkpoint(output / 'checkpoint_latest.pt')
    assert config.fusion_mode == 'context' and restored.fusion is None
    assert saved['metrics']['train']['alignment_representation'] == 'context'
    assert saved['metrics']['train']['local_positive_dtw'] is None
    assert (output / 'checkpoint_best.pt').exists()
    fused_output = main([*args, '--run-name', 'fused-smoke'])
    old, _, old_config, _ = load_checkpoint(fused_output / 'checkpoint_latest.pt')
    assert old_config.fusion_mode == 'concat' and old.fusion is not None


def test_offline_launcher_accepts_context_and_forwards_flags(tmp_path):
    fake = tmp_path / 'bin'; fake.mkdir()
    sbatch = fake / 'sbatch'
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=f'{fake}:{os.environ["PATH"]}',
               RUN_NAME='real_context_typed_local0', FUSION_MODE='context', USE_GATED_FUSION='0')
    flags = ['--local-positive-dtw-weight', '0', '--negative-loss-type', 'hybrid_typed',
             '--negative-count', '4', '--negative-target-mode', 'uniform_ratio',
             '--negative-curriculum-epochs', '0', '--dtw-normalization', 'aligned_mean']
    result = subprocess.run(['bash', 'scripts/train/train.sh', 'real', *flags], cwd=ROOT,
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[-len(flags):] == flags
    assert '--job-name=real_context_typed_local0' in result.stdout


def test_invalid_context_gate_and_local_auxiliary():
    with pytest.raises(ValueError, match='local_positive_dtw_weight=0'):
        validate_objective(compact(local_positive_dtw_weight=.3))
    with pytest.raises(ValueError, match='gate must be OFF'):
        build_model(compact(use_gated_fusion=1))

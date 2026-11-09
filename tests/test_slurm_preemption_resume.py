"""A SLURM requeue must continue the run rather than overwrite its directory."""
import os
from pathlib import Path
import subprocess
import json

import pytest

from train import main
from test_dataset import make_synthetic


ROOT = Path(__file__).resolve().parents[1]


def run_batch(tmp_path, restart_count, checkpoint=False):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    conda_base = tmp_path / 'conda'
    (conda_base / 'etc/profile.d').mkdir(parents=True)
    (conda_base / 'etc/profile.d/conda.sh').write_text(
        'conda() { export CONDA_DEFAULT_ENV="$2"; }\n')
    fake_conda = bin_dir / 'conda'
    fake_conda.write_text('#!/bin/bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n')
    fake_gpu = bin_dir / 'nvidia-smi'
    fake_gpu.write_text('#!/bin/bash\nexit 0\n')
    fake_python = bin_dir / 'python'
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$FAKE_CALL_LOG"\n')
    for path in (fake_conda, fake_gpu, fake_python):
        path.chmod(0o755)
    if checkpoint:
        target = tmp_path / 'Weights/requeued/checkpoint_latest.pt'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'complete checkpoint')
    log = tmp_path / 'args.log'
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
               FAKE_CONDA_BASE=str(conda_base), FAKE_CALL_LOG=str(log),
               SLURM_JOB_NAME='requeued', SLURM_RESTART_COUNT=str(restart_count))
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/train/train.sbatch'), '--dataset', 'data',
         '--space-mode', 'optional'], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True)
    return log.read_text().splitlines(), result.stdout


def test_requeued_batch_resumes_latest_checkpoint(tmp_path):
    args, output = run_batch(tmp_path, 1, checkpoint=True)
    assert args.count('--resume') == 1
    assert args[args.index('--resume') + 1] == 'Weights/requeued/checkpoint_latest.pt'
    assert '--restart-after-preemption' not in args
    assert 'SLURM restart: resuming' in output


def test_requeue_before_first_checkpoint_retries_safely(tmp_path):
    args, output = run_batch(tmp_path, 1)
    assert '--restart-after-preemption' in args
    assert '--resume' not in args
    assert 'retrying the first epoch' in output


def test_fresh_batch_does_not_resume_an_old_run(tmp_path):
    args, _ = run_batch(tmp_path, 0, checkpoint=True)
    assert '--resume' not in args and '--restart-after-preemption' not in args


def test_first_epoch_retry_requires_slurm_requeue(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_RESTART_COUNT', raising=False)
    with pytest.raises(ValueError, match='requires a requeued SLURM job'):
        main(['--dataset', 'unused', '--run-name', 'unused',
              '--output-root', str(tmp_path), '--restart-after-preemption'])


def test_first_epoch_retry_rejects_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_RESTART_COUNT', '1')
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'checkpoint_latest.pt').write_bytes(b'complete checkpoint')
    with pytest.raises(FileExistsError, match='use --resume'):
        main(['--dataset', 'unused', '--run-name', 'run',
              '--output-root', str(tmp_path), '--restart-after-preemption'])


def test_first_epoch_retry_and_history_reconcile(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_RESTART_COUNT', '1')
    data = make_synthetic(tmp_path / 'data', count=10)
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'split_manifest.json').write_text('{}')
    args = ['--dataset', str(data), '--dataset-type', 'synthetic',
            '--run-name', 'run', '--output-root', str(tmp_path),
            '--cnn-type', 'simple', '--no-cnn-pretrained',
            '--image-height', '32', '--image-width', '64', '--embedding-dim', '16',
            '--transformer-layers', '1', '--transformer-heads', '1',
            '--batch-size', '2', '--num-workers', '0', '--epochs', '1',
            '--max-batches', '1']
    main(args + ['--restart-after-preemption'])
    checkpoint = output / 'checkpoint_latest.pt'
    assert checkpoint.exists()
    history = json.loads((output / 'history.json').read_text())
    (output / 'history.json').write_text(json.dumps(history + [dict(epoch=2)]))
    main(args + ['--resume', str(checkpoint)])
    assert [entry['epoch'] for entry in json.loads((output / 'history.json').read_text())] == [1]

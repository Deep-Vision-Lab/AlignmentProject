"""Run single-process Optuna alignment search with an Excel save after every trial.

Example:
    python scripts/train/run_optuna.py --dataset DataSet/ArabicDataset --n-trials 30
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import optuna

from optuna_tracker import DEFAULT_TRACKER, initialize, ranking_score, save_trial
from parameters import Config
from train import main as train_main


DEFAULT_SEARCH_SPACE = {
    'fusion': ['concat', 'sum', 'gated_sum'],
    'dropout': [0.0, 0.1, 0.3],
    'vector_dimension': [64, 128, 256],
    'sigreg_weight': [0.0, 0.3],
    'dtw_gamma': [0.02, 0.05, 0.1],
    'transformer_layers': [1, 3, 5],
    'transformer_heads': [1, 2, 4],
    'window_size': [32, 64, 128],
    'stride_ratio': [0.5, 0.75, 1.0],
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _load_json(path, default):
    return json.loads(Path(path).read_text()) if path else default


def _suggest_config(trial, base, search_space):
    required = set(DEFAULT_SEARCH_SPACE)
    if set(search_space) != required:
        raise ValueError(f'Search space needs exactly these keys: {sorted(required)}')
    vector_dim = trial.suggest_categorical('vector_dimension', search_space['vector_dimension'])
    valid_heads = [head for head in search_space['transformer_heads']
                   if vector_dim % head == 0]
    transformer_heads = trial.suggest_categorical('transformer_heads', valid_heads)
    choices = {'vector_dimension': vector_dim, 'transformer_heads': transformer_heads}
    choices.update({name: trial.suggest_categorical(name, values)
                    for name, values in search_space.items()
                    if name not in ('vector_dimension', 'transformer_heads')})
    config = dict(base)
    fusion = choices['fusion']
    config.update(fusion_mode='sum' if fusion == 'gated_sum' else fusion,
                  use_gated_fusion=int(fusion == 'gated_sum'),
                  local_dropout=choices['dropout'],
                  transformer_dropout=choices['dropout'],
                  embedding_dim=choices['vector_dimension'],
                  sigreg_weight=choices['sigreg_weight'],
                  dtw_gamma=choices['dtw_gamma'],
                  transformer_layers=choices['transformer_layers'],
                  transformer_heads=choices['transformer_heads'],
                  window_width=choices['window_size'],
                  window_stride=max(1, round(choices['window_size'] * choices['stride_ratio'])))
    return config


def _valid_head_choices(search_space):
    dimensions = search_space['vector_dimension']
    heads = search_space['transformer_heads']
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in dimensions + heads):
        raise ValueError('Vector dimensions and Transformer heads must be positive integers')
    choices = {dimension: tuple(head for head in heads if dimension % head == 0)
               for dimension in dimensions}
    if any(not valid for valid in choices.values()):
        raise ValueError('Every vector dimension needs at least one divisible head choice')
    if len(set(choices.values())) != 1:
        raise ValueError('This Optuna storage requires the same valid Transformer-head '
                         'choices for every vector dimension')
    return choices


def _check_existing_study(study, valid_heads):
    """Reject older categorical distributions before running or writing trials."""
    expected = tuple(next(iter(valid_heads.values())))
    for trial in study.get_trials(deepcopy=False):
        distribution = trial.distributions.get('transformer_heads')
        if distribution is not None and tuple(distribution.choices) != expected:
            raise RuntimeError(
                f'Study {study.study_name!r} has Transformer-head choices '
                f'{tuple(distribution.choices)!r}; corrected search requires {expected!r}. '
                'Use a new --study-name and --storage database.')


def _config_args(config, args, trial_name):
    argv = ['--dataset', str(args.dataset), '--run-name', trial_name,
            '--output-root', str(args.checkpoint_root), '--device', args.device,
            '--max-batches', str(args.max_batches)]
    for field in fields(Config):
        name = '--' + field.name.replace('_', '-')
        value = config[field.name]
        if isinstance(value, bool):
            argv.append(name if value else '--no-' + field.name.replace('_', '-'))
        else:
            argv.extend((name, str(value)))
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--base-config', type=Path, help='JSON overrides for Config defaults')
    parser.add_argument('--search-space', type=Path, help='JSON categorical choices')
    parser.add_argument('--tracker', type=Path, default=DEFAULT_TRACKER)
    parser.add_argument('--checkpoint-root', type=Path, default=Path('results/optuna/checkpoints'))
    parser.add_argument('--log-root', type=Path, default=Path('results/optuna/logs'))
    parser.add_argument('--storage', help='Optuna SQLAlchemy URL; defaults beside the workbook')
    parser.add_argument('--study-name', default='alignment_no_pruning_v2')
    args = parser.parse_args(argv)
    if args.n_trials < 1 or args.epochs < 1 or args.max_batches < 0:
        parser.error('n-trials and epochs must be positive; max-batches must be nonnegative')
    search_space = _load_json(args.search_space, DEFAULT_SEARCH_SPACE)
    if set(search_space) != set(DEFAULT_SEARCH_SPACE) or any(
            not isinstance(values, list) or not values for values in search_space.values()):
        parser.error('Search space must contain a nonempty choice list for each default parameter')
    try:
        valid_heads = _valid_head_choices(search_space)
    except ValueError as exc:
        parser.error(str(exc))
    base = asdict(Config())
    overrides = _load_json(args.base_config, {})
    unknown = set(overrides) - set(base)
    if unknown:
        parser.error(f'Unknown Config fields: {sorted(unknown)}')
    base.update(overrides)
    base.update(epochs=args.epochs, seed=args.seed)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.tracker.parent.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f'sqlite:///{(args.tracker.parent.resolve() / "study_no_pruning_v2.db")}'
    study = optuna.create_study(study_name=args.study_name, storage=storage,
        direction='maximize', load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.NopPruner())
    _check_existing_study(study, valid_heads)
    study_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', study.study_name)
    study_tag += '_' + hashlib.sha256(storage.encode()).hexdigest()[:8]
    args.checkpoint_root = args.checkpoint_root / study_tag
    args.log_root = args.log_root / study_tag
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    initialize(args.tracker, search_space)

    def run_trial(trial):
        config = _suggest_config(trial, base, search_space)
        run_name = f'trial_{trial.number:05d}'
        output_dir = args.checkpoint_root / run_name
        log_path = args.log_root / f'{run_name}.log'
        trial.set_user_attr('config', config)
        trial.set_user_attr('output_dir', str(output_dir))
        trial.set_user_attr('log_path', str(log_path))

        with log_path.open('w') as log, redirect_stdout(_Tee(sys.stdout, log)), \
                redirect_stderr(_Tee(sys.stderr, log)):
            try:
                train_main(_config_args(config, args, run_name))
                history = json.loads((output_dir / 'history.json').read_text())
                score, metric = ranking_score(history[-1]['validation'])
                if score is None:
                    raise ValueError('Final validation has no ranking metric')
                print(f'Final validation objective ({metric}): {score}', flush=True)
                return score
            except Exception as exc:
                trial.set_user_attr('error', f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
                raise

    def save_finished(study, frozen):
        output_dir = Path(frozen.user_attrs.get('output_dir', ''))
        history_path = output_dir / 'history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        status = frozen.state.name
        if status not in ('COMPLETE', 'PRUNED', 'FAIL'):
            status = 'FAILED'
        elif status == 'FAIL':
            status = 'FAILED'
        save_trial(args.tracker, trial_id=f'{study.study_name}:{frozen.number}', status=status,
                   config=frozen.user_attrs.get('config', base), history=history,
                   checkpoint=output_dir / 'checkpoint_latest.pt',
                   log_path=frozen.user_attrs.get('log_path'),
                   error=frozen.user_attrs.get('error'), search_space=search_space)
        print(f'Excel saved after trial {frozen.number}: {args.tracker}', flush=True)

    study.optimize(run_trial, n_trials=args.n_trials, callbacks=[save_finished],
                   catch=(Exception,))
    return args.tracker


if __name__ == '__main__':
    main()

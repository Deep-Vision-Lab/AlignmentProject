"""Small Optuna/SQLite/Excel integration test without model training."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openpyxl import load_workbook
import optuna

from scripts.train import run_optuna


class _ChoiceTrial:
    def __init__(self, dimension):
        self.dimension = dimension
        self.seen = {}

    def suggest_categorical(self, name, choices):
        self.seen[name] = tuple(choices)
        return self.dimension if name == 'vector_dimension' else choices[0]


class OptunaSmokeTest(unittest.TestCase):
    def test_conditional_heads_for_every_dimension(self):
        for dimension in (64, 128, 256):
            with self.subTest(dimension=dimension):
                trial = _ChoiceTrial(dimension)
                config = run_optuna._suggest_config(
                    trial, {}, run_optuna.DEFAULT_SEARCH_SPACE)
                self.assertEqual(trial.seen['transformer_heads'], (1, 2))
                self.assertNotIn(3, trial.seen['transformer_heads'])
                self.assertEqual(config['embedding_dim'], dimension)
                self.assertEqual(dimension % config['transformer_heads'], 0)
                self.assertTrue(all(dimension % head == 0
                                    for head in trial.seen['transformer_heads']))
                self.assertEqual(set(trial.seen), set(run_optuna.DEFAULT_SEARCH_SPACE))

    def test_full_epoch_failure_excel_and_sqlite(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracker = root / 'tracker.xlsx'
            database = root / 'study.db'
            checkpoints = root / 'checkpoints'
            logs = root / 'logs'
            study_name = 'no_prune_smoke'
            trained_epochs = {}

            def fake_training(argv):
                def argument(name):
                    return argv[argv.index(name) + 1]

                run_name = argument('--run-name')
                configured_epochs = int(argument('--epochs'))
                output = Path(argument('--output-root')) / run_name
                output.mkdir(parents=True)
                if run_name == 'trial_00001':
                    self.assertTrue(tracker.exists())
                    prior = load_workbook(tracker, read_only=True, data_only=True)
                    self.assertEqual(prior['Dashboard']['B7'].value, 1)
                    prior.close()
                history = []
                for epoch in range(1, configured_epochs + 1):
                    history.append(dict(epoch=epoch, model_parameter_count=100,
                        train=dict(total=2.0 / epoch, positive_dtw=2.0 / epoch,
                                   seconds=.1, gradients={}),
                        validation=dict(total=1.0 / epoch,
                                        positive_dtw=1.0 / epoch, seconds=.1)))
                    (output / 'history.json').write_text(json.dumps(history))
                    trained_epochs[run_name] = epoch
                    if run_name == 'trial_00001':
                        raise RuntimeError('simulated real training error')
                (output / 'checkpoint_latest.pt').write_bytes(b'smoke checkpoint')

            arguments = ['--dataset', str(root), '--n-trials', '2', '--epochs', '3',
                         '--tracker', str(tracker), '--checkpoint-root', str(checkpoints),
                         '--log-root', str(logs), '--storage', f'sqlite:///{database}',
                         '--study-name', study_name]
            with patch.object(run_optuna, 'train_main', fake_training):
                run_optuna.main(arguments)

            study = optuna.load_study(study_name=study_name,
                                      storage=f'sqlite:///{database}')
            self.assertEqual([trial.state.name for trial in study.trials],
                             ['COMPLETE', 'FAIL'])
            self.assertEqual(trained_epochs, {'trial_00000': 3, 'trial_00001': 1})
            for trial in study.trials:
                dimension = trial.params['vector_dimension']
                heads = trial.params['transformer_heads']
                self.assertEqual(dimension % heads, 0)
                self.assertNotEqual(heads, 3)
                self.assertEqual(tuple(trial.distributions['transformer_heads'].choices),
                                 (1, 2))
                self.assertFalse(trial.intermediate_values)
            self.assertIn('simulated real training error',
                          study.trials[1].user_attrs['error'])

            workbook = load_workbook(tracker, data_only=True)
            self.assertEqual(workbook.sheetnames,
                ['RankedResults', 'Trials', 'EpochMetrics', 'Dashboard', 'SearchSpace'])
            ranked = list(workbook['RankedResults'].iter_rows(values_only=True))
            self.assertEqual(ranked[1][0:3], (1, f'{study_name}:0', 'COMPLETE'))
            self.assertEqual(ranked[2][0:3], (None, f'{study_name}:1', 'FAILED'))
            epochs = list(workbook['EpochMetrics'].iter_rows(min_row=2, values_only=True))
            self.assertEqual(len(epochs), 8)
            self.assertEqual({(row[0], row[1], row[2]) for row in epochs},
                {(f'{study_name}:0', epoch, split)
                 for epoch in (1, 2, 3) for split in ('train', 'val')} |
                {(f'{study_name}:1', 1, split) for split in ('train', 'val')})
            self.assertEqual(workbook['Dashboard']['B7'].value, 1)
            self.assertEqual(workbook['Dashboard']['B8'].value, 0)
            self.assertEqual(workbook['Dashboard']['B9'].value, 1)
            workbook.close()
            self.assertTrue(database.exists())

    def test_legacy_head_distribution_is_rejected(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / 'legacy.db'
            storage = f'sqlite:///{database}'
            legacy = optuna.create_study(study_name='alignment', storage=storage,
                                         direction='maximize')
            legacy.optimize(lambda trial: float(trial.suggest_categorical(
                'transformer_heads', [1, 2, 3])), n_trials=1)
            with self.assertRaisesRegex(RuntimeError, 'new --study-name and --storage'):
                run_optuna.main(['--dataset', str(root), '--n-trials', '1',
                    '--tracker', str(root / 'tracker.xlsx'), '--storage', storage,
                    '--study-name', 'alignment', '--checkpoint-root', str(root / 'weights'),
                    '--log-root', str(root / 'logs')])


if __name__ == '__main__':
    unittest.main()

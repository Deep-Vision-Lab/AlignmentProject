import json

from openpyxl import load_workbook

from optuna_tracker import initialize, save_trial


def _config():
    return dict(cnn_type='simple', cnn_layers=4,
                fusion_mode='concat', use_gated_fusion=0, local_dropout=.1,
                embedding_dim=128, sigreg_weight=0., dtw_gamma=.05,
                transformer_layers=5, transformer_heads=1,
                window_width=32, window_stride=16, seed=42)


def _history(losses, f1=None):
    return [dict(epoch=index, train=dict(total=loss + .5, positive_dtw=loss + .5,
                   seconds=2., gradients=dict(cnn_mean=.1, transformer_mean=.2,
                   fusion_mean=.3)), validation=dict(total=loss, positive_dtw=loss,
                   alignment_f1=f1, seconds=1.), model_parameter_count=123)
            for index, loss in enumerate(losses, 1)]


def test_cnn_type_and_layers_are_recorded(tmp_path):
    path = tmp_path / 'tracker.xlsx'
    save_trial(path, trial_id=0, status='COMPLETE', config=_config(),
               history=_history([1.]))
    resnet = dict(_config(), cnn_type='resnet18', cnn_layers=3)
    save_trial(path, trial_id=1, status='COMPLETE', config=resnet,
               history=_history([2.]))
    book = load_workbook(path, data_only=True)
    for sheet in ('RankedResults', 'Trials'):
        rows = list(book[sheet].iter_rows(values_only=True))
        head = {value: index for index, value in enumerate(rows[0])}
        assert 'CNN type' in head and 'CNN layers' in head
        values = {row[head['Trial ID']]: (row[head['CNN type']], row[head['CNN layers']])
                  for row in rows[1:]}
        assert values[0] == ('simple', 4)
        assert values[1] == ('resnet18', 'fixed')
    labels = [cell.value for cell in book['Dashboard'][13]]
    assert 'CNN type' in labels and 'CNN layers' in labels
    top_row = [cell.value for cell in book['Dashboard'][14]]
    assert top_row[labels.index('CNN type')] == 'simple'
    assert top_row[labels.index('CNN layers')] == 4
    book.close()


def test_legacy_workbook_without_cnn_columns_still_loads(tmp_path):
    from openpyxl import Workbook

    from optuna_tracker import METRICS, SUMMARY_HEADERS

    path = tmp_path / 'legacy.xlsx'
    old_config_headers = ('Trial ID', 'Status', 'Fusion', 'Dropout', 'Vector dimension',
                          'SIGReg ON/OFF', 'DTW gamma', 'Transformer layers',
                          'Transformer heads', 'Window size', 'Stride ratio',
                          'Actual stride pixels')
    old_headers = old_config_headers + SUMMARY_HEADERS + ('Config JSON', 'Error',
                                                          'Updated UTC')
    book = Workbook()
    trials_sheet = book.active
    trials_sheet.title = 'Trials'
    trials_sheet.append(old_headers)
    legacy_config = _config()
    del legacy_config['cnn_type'], legacy_config['cnn_layers']
    values = {header: None for header in old_headers}
    values.update({'Trial ID': 7, 'Status': 'COMPLETE', 'Model parameter count': 999,
                   'Config JSON': json.dumps(legacy_config)})
    trials_sheet.append(tuple(values[header] for header in old_headers))
    epoch_sheet = book.create_sheet('EpochMetrics')
    epoch_sheet.append(('Trial_ID', 'Epoch', 'Split') + METRICS)
    for split in ('train', 'val'):
        epoch_sheet.append((7, 1, split) + tuple(
            1.0 if metric == 'Total loss' else .1 if metric == 'Epoch time (s)' else None
            for metric in METRICS))
    book.save(path)
    book.close()

    save_trial(path, trial_id='new:0', status='COMPLETE', config=_config(),
               history=_history([1.]))
    book = load_workbook(path, data_only=True)
    rows = list(book['Trials'].iter_rows(values_only=True))
    head = {value: index for index, value in enumerate(rows[0])}
    old = next(row for row in rows[1:] if row[head['Trial ID']] == 7)
    assert old[head['Model parameter count']] == 999  # read via legacy header position
    assert old[head['CNN type']] is None and old[head['CNN layers']] is None
    new = next(row for row in rows[1:] if row[head['Trial ID']] == 'new:0')
    assert new[head['CNN type']] == 'simple' and new[head['CNN layers']] == 4
    book.close()


def test_fixed_parameters_are_marked_not_searched(tmp_path):
    from scripts.train.run_optuna import DEFAULT_SEARCH_SPACE, FIXED_SEARCH_PARAMETERS

    path = tmp_path / 'tracker.xlsx'
    initialize(path, DEFAULT_SEARCH_SPACE, fixed=FIXED_SEARCH_PARAMETERS)
    save_trial(path, trial_id=0, status='COMPLETE', config=_config(),
               history=_history([1.]))  # fixed values persist across later saves
    book = load_workbook(path, data_only=True)
    space = {row[0]: (row[1], row[2]) for row in
             book['SearchSpace'].iter_rows(min_row=2, values_only=True)}
    assert set(space) == set(DEFAULT_SEARCH_SPACE)
    assert 'cnn_layers' in space and 'sigreg_weight' in space
    for removed in ('dropout', 'dtw_gamma', 'stride_ratio'):
        assert removed not in space  # not displayed as Optuna variables
    fixed = dict(book['FixedParameters'].iter_rows(min_row=2, values_only=True))
    assert fixed['local_dropout'] == 0.1
    assert fixed['transformer_dropout'] == 0.1
    assert fixed['dtw_gamma'] == 0.5
    assert fixed['stride_ratio'] == 0.5
    assert fixed['cnn_type'] == 'simple'
    assert bool(fixed['cnn_pretrained']) is False
    dashboard = book['Dashboard']
    fixed_labels = {dashboard.cell(row, 1).value for row in range(36, 42)}
    assert fixed_labels == {f'{key} (FIXED)' for key in FIXED_SEARCH_PARAMETERS}
    fixed_row = {dashboard.cell(row, 1).value: dashboard.cell(row, 2).value
                 for row in range(36, 42)}
    assert fixed_row['dtw_gamma (FIXED)'] == 0.5
    assert fixed_row['local_dropout (FIXED)'] == 0.1
    assert fixed_row['stride_ratio (FIXED)'] == 0.5
    book.close()


def test_tracker_saves_every_epoch_and_physically_sorts(tmp_path):
    path = tmp_path / 'tracker.xlsx'
    initialize(path, {'fusion': ['concat', 'sum']})
    save_trial(path, trial_id=0, status='COMPLETE', config=_config(),
               history=_history([2., 1.]), search_space={'fusion': ['concat', 'sum']})
    save_trial(path, trial_id=1, status='COMPLETE', config=_config(),
               history=_history([3., .8]), search_space={'fusion': ['concat', 'sum']})
    save_trial(path, trial_id=2, status='PRUNED', config=_config(),
               history=_history([.1]), search_space={'fusion': ['concat', 'sum']})
    book = load_workbook(path, data_only=True)
    assert book.sheetnames == ['RankedResults', 'Trials', 'EpochMetrics', 'Dashboard',
                               'SearchSpace', 'FixedParameters']
    rows = list(book['RankedResults'].iter_rows(values_only=True))
    head = {value: index for index, value in enumerate(rows[0])}
    assert [row[head['Trial ID']] for row in rows[1:]] == [1, 0, 2]
    assert [row[head['Rank']] for row in rows[1:]] == [1, 2, None]
    assert rows[1][head['Learning_Improvement']] == 2.2
    epoch_rows = list(book['EpochMetrics'].iter_rows(values_only=True))
    assert len(epoch_rows) == 11  # Header + (2 + 2 + 1) epochs x 2 splits.
    assert {(row[0], row[1], row[2]) for row in epoch_rows[1:]} == {
        (0, 1, 'train'), (0, 1, 'val'), (0, 2, 'train'), (0, 2, 'val'),
        (1, 1, 'train'), (1, 1, 'val'), (1, 2, 'train'), (1, 2, 'val'),
        (2, 1, 'train'), (2, 1, 'val')}
    assert book['Dashboard']['B3'].value == 1
    assert book['Dashboard']['B7'].value == 2
    assert book['Dashboard']['B8'].value == 1
    assert book['SearchSpace']['B2'].value == json.dumps(['concat', 'sum'])
    book.close()


def test_f1_priority_and_improvement_tie_break(tmp_path):
    path = tmp_path / 'tracker.xlsx'
    for trial_id, losses, f1 in ((0, [2., .1], None), (1, [2., 1.], .6),
                                 (2, [2., 1.], .6)):
        history = _history(losses, f1)
        if trial_id == 1:
            history[0]['validation']['alignment_f1'] = .3
        if trial_id == 2:
            history[0]['validation']['alignment_f1'] = .5
        save_trial(path, trial_id=trial_id, status='COMPLETE', config=_config(),
                   history=history)
    book = load_workbook(path, data_only=True)
    assert [row[1] for row in book['RankedResults'].iter_rows(min_row=2, values_only=True)] == [1, 2, 0]
    book.close()


def test_old_and_new_study_trial_ids_do_not_collide(tmp_path):
    path = tmp_path / 'tracker.xlsx'
    save_trial(path, trial_id=0, status='COMPLETE', config=_config(),
               history=_history([2., 1.]))
    save_trial(path, trial_id='alignment_no_pruning_v2:0', status='COMPLETE',
               config=_config(), history=_history([2., .5]))
    book = load_workbook(path, data_only=True)
    assert [row[1] for row in book['RankedResults'].iter_rows(min_row=2, values_only=True)] == [
        'alignment_no_pruning_v2:0', 0]
    assert book['Trials'].max_row == 3
    book.close()

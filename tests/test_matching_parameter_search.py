"""Validate selection semantics, matcher parity, and a real resumable Optuna run."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
from PIL import Image
import pytest
import torch

from evaluate import resolved_match_settings
from evaluation_utils import EvaluationSession, predict_cached_pair
from scripts.eval import tune_matching as tuning
from test_evaluation_notebook import evaluation_fixture


def prepared_pair(sample_id, truth_widths, size=(60, 10)):
    width, height = size
    geometry = dict(source_size=[width, height], crop=[0, 0, width, height], scale_x=1)
    lines = [dict(features={'fused': torch.eye(6)}, physical=np.arange(6), geometry=geometry)
             for _ in range(2)]
    masks = []
    for end in truth_widths:
        mask = np.zeros((height, width), dtype=bool)
        mask[:, :end] = True
        masks.append(mask)
    return tuning.PreparedPair(dict(sample_id=sample_id), lines, np.eye(6) * .95,
                               tuple(masks), tuning.category_for_masks(masks))


def test_objective_weights_both_sides_and_pairs_equally_and_excludes_black_pairs():
    session = SimpleNamespace(config=SimpleNamespace(window_width=10, window_stride=10), text_encoder=None)
    first = prepared_pair('positive1', (30, 60))  # A=.5, B=1, pair=.75
    second = prepared_pair('positive2', (10, 10), size=(60, 30))  # Larger image: pair=1/6
    negative = prepared_pair('negative', (0, 0))
    partial = prepared_pair('partial', (0, 30))
    settings = resolved_match_settings(dict(threshold=.5, score_mode='raw',
        enable_path_extension=False, enable_region_merging=False, allow_short_regions=False))
    metrics, rows = tuning.evaluate_settings(session, [first, second, negative, partial], 'fused', settings)
    assert metrics['mean_mask_iou'] == pytest.approx((.75 + 1 / 6) / 2)
    assert metrics['a_mean_iou'] == pytest.approx((.5 + 1 / 6) / 2)
    assert metrics['b_mean_iou'] == pytest.approx((1 + 1 / 6) / 2)
    assert metrics['negative_false_positive_rate'] == 1
    assert metrics['positive_pairs'] == 2 and metrics['partial_pairs'] == 1
    assert rows[2]['pair_mean_iou'] == 0
    empty, _ = tuning.evaluate_settings(session, [first, negative], 'fused', dict(settings, threshold=1.))
    assert empty['mean_mask_iou'] == 0
    assert empty['negative_false_positive_rate'] == 0
    assert tuning.mask_iou(np.zeros((3, 3)), np.zeros((3, 3), bool)) == 1


def test_folder_masks_missing_black_and_source_size_validation(tmp_path):
    sides = []
    for side in 'AB':
        image = tmp_path / ('images' + side) / 'line.png'
        image.parent.mkdir()
        Image.new('L', (60, 10)).save(image)
        sides.append(dict(image=str(image)))
    pair = dict(sides=sides)
    masks, sources = tuning.load_masks(pair, 'folders')
    assert all(not mask.any() for mask in masks)
    assert not any(source['exists'] for source in sources)
    path = tmp_path / 'masksA' / 'line.png'
    path.parent.mkdir()
    Image.new('L', (60, 10), 255).save(path)
    masks, _ = tuning.load_masks(pair, 'folders')
    assert tuning.category_for_masks(masks) == 'partial'
    Image.new('L', (59, 10)).save(path)
    with pytest.raises(ValueError, match='Mask/source size mismatch'):
        tuning.load_masks(pair, 'folders')


def test_saved_settings_match_notebook_prediction_and_short_class_is_reachable():
    cached = prepared_pair('positive', (60, 60))
    session = SimpleNamespace(config=SimpleNamespace(window_width=10, window_stride=10), text_encoder=None)
    settings = tuning.suggest_settings(optuna.trial.FixedTrial(tuning.BASELINE_PARAMS))
    session.settings = settings
    cached.pair.update(sides=[{}, {}])
    session.split = 'val'
    result = predict_cached_pair(session, cached.pair, cached.lines, cosine=cached.cosine)
    metrics, rows = tuning.evaluate_settings(session, [cached], 'fused', settings)
    assert rows[0]['a_iou'] == tuning.mask_iou(result['masks'][0], cached.ground_truth[0])
    assert rows[0]['b_iou'] == tuning.mask_iou(result['masks'][1], cached.ground_truth[1])
    assert metrics['mean_mask_iou'] == 1
    # A compact 3x3 repeat path can be a short region without passing the normal floor.
    c = np.zeros((6, 6))
    for pair in ((0, 0), (0, 1), (1, 1), (2, 2)):
        c[pair] = .95
    from evaluate import local_repeat_regions
    match = local_repeat_regions(c, np.arange(6), np.arange(6),
        **{k: settings[k] for k in ('normal_min_distinct_a', 'normal_min_distinct_b',
                                    'normal_min_matched_pairs')},
        score_mode='raw', enable_path_extension=False, enable_region_merging=False)
    assert any(region['candidate_type'] == 'short' for region in match['regions'])


def test_real_sqlite_search_resume_replay_and_changed_masks_rejected(evaluation_fixture, tmp_path):
    root, checkpoint = evaluation_fixture
    manifest_path = root / 'dataset_manifest.jsonl'
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    for row in records:
        if row.get('split') != 'val':
            continue
        for name in 'AB':
            path = root / (name + row['pair_id'] + '_mask.png')
            mask = np.zeros((32, 160), np.uint8)
            mask[:, 30:120] = 255
            Image.fromarray(mask).save(path)
            row[name]['alignment_mask_path'] = path.name
    manifest_path.write_text('\n'.join(json.dumps(row) for row in records))
    output = tmp_path / 'search'
    checkpoint_hash = tuning.file_hash(checkpoint)
    args = ['--checkpoint', str(checkpoint), '--dataset', str(root), '--output', str(output),
            '--n-trials', '28', '--device', 'cpu', '--crop', 'checkpoint', '--negative-pairs', '0']
    study = tuning.main(args)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == 28  # Goes beyond TPE's 25 startup trials.
    assert study.best_value == max(trial.value for trial in complete)
    report = json.loads((output / 'best_result.json').read_text())
    assert report['value'] == study.best_value
    assert report['run_manifest']['split'] == 'val'
    assert len(report['run_manifest']['pairs']) == 2
    assert all('pair6' in row['sample_id'] or 'pair7' in row['sample_id']
               for row in report['run_manifest']['pairs'])
    assert (output / 'trials.csv').is_file()
    assert len(list(output.glob('trial_*_pairs.csv'))) == 28
    session = EvaluationSession(checkpoint, root, split='val', device='cpu', construct_negatives=False)
    prepared, _, _ = tuning.prepare_pairs(session, 'fused', 'manifest')
    settings = json.loads((output / 'best_settings.json').read_text())
    metrics, _ = tuning.evaluate_settings(session, prepared, 'fused', settings)
    assert metrics['mean_mask_iou'] == report['value']
    assert all(trial.user_attrs['settings']['similarity_mode'] ==
               trial.user_attrs['settings']['candidate_similarity_mode'] for trial in complete)
    resume_args = list(args)
    resume_args[resume_args.index('--n-trials') + 1] = '2'
    resumed = tuning.main([*resume_args, '--resume'])
    assert len(resumed.trials) == 30 and resumed.best_value >= study.best_value
    assert tuning.file_hash(checkpoint) == checkpoint_hash
    # A changed mask must not silently mix incomparable trials in the same study.
    mask_path = root / records[6]['A']['alignment_mask_path']
    Image.new('L', (160, 32), 255).save(mask_path)
    with pytest.raises(ValueError, match='different data, masks, checkpoint, code or search settings'):
        tuning.main([*args, '--resume'])


def test_missing_positive_validation_masks_fail_without_split_fallback(evaluation_fixture, tmp_path):
    root, checkpoint = evaluation_fixture
    with pytest.raises(ValueError, match='No val pairs have foreground'):
        tuning.main(['--checkpoint', str(checkpoint), '--dataset', str(root),
                     '--output', str(tmp_path / 'missing_masks'), '--n-trials', '1', '--device', 'cpu'])

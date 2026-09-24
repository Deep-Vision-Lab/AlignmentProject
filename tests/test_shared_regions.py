"""Partial image alignment: reject unsupported paths and preserve source geometry."""
import hashlib
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from Evaluation.sw_core import smith_waterman_affine
from Evaluation.shared_regions import (RegionSettings, affine_trace_score, extract_regions,
    intervals_mask, match_rewards, region_source_intervals, valid_window_geometry)
from Evaluation.eval_shared_regions import parse_args, prepare_image, score_annotations, select_pair
from Evaluation.checkpoint_contract import resolve_evaluation_contract


def planted(pairs, shape=(24, 24)):
    matrix = np.full(shape, .1, dtype=np.float32)
    for i, j in pairs:
        matrix[i, j] = .95
    return matrix


def run(matrix, **settings):
    return extract_regions(matrix, np.arange(matrix.shape[0]), np.arange(matrix.shape[1]), RegionSettings(**settings))


def test_off_diagonal_shared_passage_and_raw_cosines_unchanged():
    pairs = [(i+3, i+12) for i in range(7)]
    c = planted(pairs)
    before = c.copy()
    result = run(c)
    assert len(result['regions']) == 1
    assert result['regions'][0]['pairs'] == pairs
    assert result['regions'][0]['support'] == [7, 7]
    np.testing.assert_array_equal(c, before)


@pytest.mark.parametrize('matrix', [np.zeros((20, 20)), np.full((20, 20), .99),
    np.random.default_rng(0).uniform(-.5, .5, (20, 20)), np.empty((0, 6))])
def test_unrelated_uniform_background_and_empty_inputs(matrix):
    result = run(matrix)
    assert result['regions'] == []
    assert result['rejected'][0]['reason'] == 'no_positive_local_alignment'


def test_raw_mode_documents_uniform_background_false_positive_risk():
    assert len(run(np.full((20, 20), .99), score_mode='raw')['regions']) == 1
    c = planted([(4, 10)])
    np.testing.assert_allclose(match_rewards(c, RegionSettings(score_mode='raw')), c.astype(float)-.6)


def test_short_isolated_peaks_do_not_make_masks():
    r = run(planted([(i, i+8) for i in range(4)]))
    assert not r['regions']
    assert r['rejected'][0]['reason'] == 'insufficient_distinct_positive_windows'
    assert len(run(planted([(i, i+8) for i in range(4)]), min_windows=4)['regions']) == 1


def test_one_internal_gap_and_strict_consecutive_mode():
    pairs = [(0, 5), (1, 6), (2, 7), (4, 8), (5, 9), (6, 10)]
    result = run(planted(pairs))
    region = result['regions'][0]
    assert region['support'] == [6, 6]
    assert region['filled_physical'] == [[3], []]
    assert (3, None) in region['steps']
    assert not run(planted(pairs), max_internal_gap=0)['regions']


def test_large_internal_gap_splits_and_requires_support_on_each_piece():
    pairs = [(i, i) for i in range(3)] + [(i+10, i+10) for i in range(3)]
    assert not run(planted(pairs))['regions']
    pairs = [(i, i+2) for i in range(5)] + [(i+14, i+15) for i in range(5)]
    result = run(planted(pairs))
    assert len(result['regions']) == 2
    assert [r['support'] for r in result['regions']] == [[5, 5], [5, 5]]
    assert all(r['filled_physical'] == [[], []] for r in result['regions'])


@pytest.mark.parametrize('second', [[(i+12, i) for i in range(5)], [(i+12, i+12) for i in range(5)]])
def test_crossing_and_reused_windows_are_excluded(second):
    first = [(i, i+12) for i in range(5)]
    regions = run(planted(first + second))['regions']
    assert len(regions) == 1


def test_invalid_physical_windows_cannot_be_filled_or_counted():
    c = planted([(i, i) for i in range(6)], shape=(6, 6))
    r = extract_regions(c, [0, 1, 2, 4, 5, 6], np.arange(6))
    assert not r['regions']
    with pytest.raises(ValueError, match='unique'):
        extract_regions(c, [0, 0, 1, 2, 3, 4], np.arange(6))


def test_affine_open_extension_and_local_endpoints():
    pairs = [(1, 2), (2, 3), (3, 4), (7, 5), (8, 6), (9, 7)]
    rewards = np.full((12, 12), -5.)
    for i, j in pairs:
        rewards[i, j] = 1.
    path = smith_waterman_affine(rewards, .4, .1)
    assert [(i,j) for i,j in path.steps if i is not None and j is not None] == pairs
    assert path.score == pytest.approx(6 - .4 - 2*.1)
    assert affine_trace_score(path.steps, rewards, .4, .1) == pytest.approx(path.score)
    transposed = smith_waterman_affine(rewards.T, .4, .1)
    assert transposed.score == pytest.approx(path.score)
    assert transposed.steps == [(j, i) for i, j in path.steps]


def test_affine_optimum_matches_exhaustive_tiny_path_enumeration():
    # Independent enumeration of all locally started paths, including either gap.
    rng = np.random.default_rng(17)
    for _ in range(5):
        rewards = rng.uniform(-1, 1, (3, 3))
        totals = [0.]
        def visit(i, j, score, previous):
            totals.append(score)
            if i < 3 and j < 3:
                visit(i+1, j+1, score+rewards[i, j], None)
            if i < 3:
                visit(i+1, j, score-(.1 if previous == 'up' else .4), 'up')
            if j < 3:
                visit(i, j+1, score-(.1 if previous == 'left' else .4), 'left')
        for i in range(3):
            for j in range(3):
                visit(i, j, 0., None)
        actual = smith_waterman_affine(rewards, .4, .1)
        assert actual.score == pytest.approx(max(totals))
        assert affine_trace_score(actual.steps, rewards, .4, .1) == pytest.approx(actual.score)


def gray_contract():
    return resolve_evaluation_contract(dict(architecture_family='restoration-positive-dtw-window-encoder',
        visual_input_channels=1, visual_grayscale=True, line_geometry_mode='xml-bbox-gray-full-resize',
        window_size=32, stride=16, line_width=1024, line_height=128, real_bbox_crop=True,
        zero_shot_foreground_crop=False, zero_shot_preserve_aspect=False, full_image_no_padding=True))


def features(count, physical=None, valid=None):
    return SimpleNamespace(contextual=torch.zeros(count, 128),
        token_valid=torch.ones(count, dtype=torch.bool) if valid is None else torch.tensor(valid),
        physical_window_indices=None if physical is None else torch.tensor(physical))


def test_rtl_exact_physical_and_inverse_crop_resize_coordinates():
    geometry = dict(canvas_width=1024, crop_left=100, crop_width=2048, offset_x=0,
                    scale_x=.5, source_width=2400, source_height=210)
    entries = valid_window_geometry(features(63), geometry, gray_contract(), True)
    expected = {0: (62, 2084, 2148), 1: (61, 2052, 2116), 2: (60, 2020, 2084),
                60: (2, 164, 228), 61: (1, 132, 196), 62: (0, 100, 164)}
    for logical, (physical, x0, x1) in expected.items():
        assert entries[logical]['physical_index'] == physical
        assert entries[logical]['source_interval'] == [x0, x1]
    packed = valid_window_geometry(features(3, [62, 60, 0]), geometry, gray_contract(), True)
    assert [e['physical_index'] for e in packed] == [62, 60, 0]
    assert packed[1]['source_interval'] == [2020, 2084]


def test_padding_clipped_and_excluded():
    geometry = dict(canvas_width=128, crop_left=100, crop_width=80, offset_x=40,
                    scale_x=.5, source_width=300, source_height=50)
    entries = valid_window_geometry(features(7), geometry, gray_contract(), False)
    assert [e['physical_index'] for e in entries] == [1, 2, 3, 4]
    assert entries[0]['source_interval'] == [100, 116]
    assert entries[-1]['source_interval'] == [148, 180]


def test_masks_are_source_size_full_height_with_separated_support():
    geometry = dict(canvas_width=1024, crop_left=100, crop_width=2048, offset_x=0, scale_x=.5)
    region = dict(supported_physical=[[1,2,3,4,5], []], filled_physical=[[], []])
    intervals = region_source_intervals(region, 0, geometry, gray_contract())
    mask = np.asarray(intervals_mask(intervals+[[1000, 1040]], (2400, 210)))
    assert mask.shape == (210, 2400)
    assert np.all(mask[:, 132:324] == 255)
    assert np.all(mask[:, 324:1000] == 0)
    assert np.array_equal(mask[0], mask[-1])
    assert not np.asarray(intervals_mask([], (2400, 210))).any()


def test_gt_scoring_geometry_empty_and_missing(tmp_path):
    geometry = dict(canvas_width=1024, crop_left=0, crop_width=100, offset_x=0,
                    scale_x=10.24, source_width=100, source_height=30)
    mask = intervals_mask([[20,40]], (100, 30))
    gt = tmp_path/'gt.png'; mask.save(gt)
    r = score_annotations(mask, gt, geometry, gray_contract(), RegionSettings(), 'positive')
    assert r['pixel_metrics']['iou'] == 1
    assert r['column_region_metrics']['iou'] == 1
    assert r['boundary_error_px'] == 0
    assert score_annotations(mask, None, geometry, gray_contract(), RegionSettings(), 'unknown')['status'] == 'unavailable'
    empty = intervals_mask([], (100,30)); empty.save(gt)
    r = score_annotations(mask, gt, geometry, gray_contract(), RegionSettings(), 'negative')
    assert r['negative_pair_false_positive_coverage'] == .2
    Image.new('L', (99,30)).save(gt)
    with pytest.raises(ValueError, match='resizing is forbidden'):
        score_annotations(mask, gt, geometry, gray_contract(), RegionSettings(), 'unknown')


def test_external_images_require_explicit_no_xml_override(tmp_path):
    image = tmp_path/'external.png'; Image.new('L', (200, 40), 100).save(image)
    models = SimpleNamespace(contract=gray_contract())
    with pytest.raises(ValueError, match='explicit --preprocessing no-xml'):
        prepare_image(models, image, 'training')
    prepared, geometry = prepare_image(models, image, 'no-xml')
    assert prepared.mode == 'L' and prepared.size == (1024,128)
    assert geometry['explicit_preprocessing_override']


def manifest_fixture(tmp_path):
    rows=[]
    for i in range(4):
        image=tmp_path/f'image{i}.png'; Image.new('L', (20,10)).save(image)
        rows.append(dict(_root=str(tmp_path), record_id=str(i), line_image_path=image.name, pair_id=f'page{i}'))
    data=json.dumps(dict(train_eval=rows[:2], val_eval=rows[2:]), sort_keys=True).encode()
    path=tmp_path/'split_manifest.json'; path.write_bytes(data)
    return path, dict(split_manifest_sha256=hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize('split,expected', [('train',['0','1']), ('validation',['2','3'])])
def test_saved_membership_and_checkpoint_identity(tmp_path, split, expected):
    manifest, config=manifest_fixture(tmp_path)
    args=parse_args(['--weights','unused','--manifest',str(manifest),'--split',split,'--record-indices','0','1'])
    paths, selection=select_pair(args, config)
    assert selection['record_ids'] == expected
    assert selection['manifest_identity'] == 'verified'
    with pytest.raises(ValueError,match='SHA256'):
        select_pair(args,dict(split_manifest_sha256='wrong'))
    args.record_indices=[0,2]
    with pytest.raises(ValueError,match='indices'):
        select_pair(args,config)
    with pytest.raises(SystemExit):
        parse_args(['--weights','unused','--split','test'])


def test_membership_leak_fails(tmp_path):
    manifest, _=manifest_fixture(tmp_path)
    data=json.loads(manifest.read_text()); data['val_eval'][0]=data['train_eval'][0]
    manifest.write_text(json.dumps(data))
    args=parse_args(['--weights','unused','--manifest',str(manifest),'--split','train','--record-indices','0','1'])
    with pytest.raises(ValueError,match='leakage'):
        select_pair(args,{})


def test_public_outputs_csv_empty_and_gt_cannot_affect_predictions(tmp_path, monkeypatch):
    import Evaluation.eval_shared_regions as public
    images = [tmp_path/'one.png', tmp_path/'two.png']
    for path in images:
        Image.new('L', (200, 40), 150).save(path)
    weights = tmp_path/'mock.pth'; weights.write_bytes(b'mock checkpoint')
    model = torch.nn.Linear(1, 1); model.use_flip = True
    contract = gray_contract()
    models = SimpleNamespace(image_model=model, contract=contract,
        config=contract.config, checkpoint={'epoch': 1})
    def loader(path, device, *, load_text_model):
        assert not load_text_model
        return models
    monkeypatch.setattr(public, 'load_evaluation_models', loader)
    def encode(models, first, second, mode):
        assert not models.image_model.training
        assert Image.open(first).mode == Image.open(second).mode == 'L'
        values = torch.eye(63, 128)
        return [SimpleNamespace(contextual=values, token_valid=torch.ones(63, dtype=torch.bool),
                  physical_window_indices=torch.arange(62, -1, -1)) for _ in range(2)]
    monkeypatch.setattr(public, 'point2_pair_features', encode)
    monkeypatch.setattr(public, 'save_figures', lambda *a: None)
    base = ['--weights', str(weights), '--image1', str(images[0]), '--image2', str(images[1]),
            '--preprocessing', 'no-xml', '--device', 'cpu']
    output = tmp_path/'prediction'
    args = parse_args(base + ['--output-dir', str(output)])
    summary = public.evaluate(args)
    assert summary['status'] == 'accepted_regions'
    assert summary['metrics'][0]['status'] == 'unavailable'
    assert summary['valid_tokens'] == [63, 63]
    rows = list(csv.DictReader((output/'correspondences.csv').open()))
    assert rows[0]['line1_logical_index'] == '0'
    assert rows[0]['line1_physical_index'] == '62'
    assert float(rows[0]['line1_source_x0']) == pytest.approx(193.75)
    np.testing.assert_array_equal(np.load(output/'cosine.npy'), np.eye(63, dtype=np.float32))
    for side in (1, 2):
        mask = np.asarray(Image.open(output/f'line{side}_mask.png'))
        assert mask.shape == (40, 200) and (mask == 255).all()
    with pytest.raises(FileExistsError, match='overwrite'):
        public.evaluate(args)
    gt = tmp_path/'bad_gt.png'; Image.new('L', (199,40)).save(gt)
    annotated = tmp_path/'annotated'
    with pytest.raises(ValueError, match='resizing is forbidden'):
        public.evaluate(parse_args(base + ['--output-dir', str(annotated), '--gt-mask1', str(gt)]))
    assert json.loads((annotated/'summary.json').read_text())['annotation_error']
    assert (output/'line1_mask.png').read_bytes() == (annotated/'line1_mask.png').read_bytes()
    empty = tmp_path/'empty'
    public.evaluate(parse_args(base + ['--output-dir', str(empty), '--cosine-threshold', '1']))
    assert not np.asarray(Image.open(empty/'line1_mask.png')).any()
    assert len(list(csv.DictReader((empty/'correspondences.csv').open()))) == 0


def test_local_checkpoint_real_inference_is_repeatable_and_read_only(tmp_path):
    """Optional local-assets acceptance; never downloads models or datasets."""
    from Evaluation._eval_utils import load_evaluation_models
    from Evaluation.point2_runtime import point2_pair_features
    root = Path(__file__).resolve().parents[1]
    checkpoint = root/'Weights/resnet18_128d_5l_1h_no_pos_21634461/model_best_validation_dtw.pth'
    manifest = root/'Results/Monitoring/resnet18_128d_5l_1h_no_pos_21634461/split_manifest.json'
    if not checkpoint.is_file() or not manifest.is_file():
        pytest.skip('Local real checkpoint and saved split membership are unavailable')
    models = load_evaluation_models(checkpoint, 'cpu', load_text_model=False)
    assert models.text_model is None
    models.image_model.eval()
    args = parse_args(['--weights',str(checkpoint),'--manifest',str(manifest),
                       '--split','validation','--record-indices','14','4'])
    paths, selection = select_pair(args, models.config)
    assert selection['manifest_identity'] == 'verified'
    before = {k: v.detach().clone() for k,v in models.image_model.state_dict().items()}
    prepared = []
    for k, path in enumerate(paths):
        image, _ = prepare_image(models, path, 'training')
        assert image.mode == 'L' and image.size == (1024,128)
        target = tmp_path/f'input{k}.png'; image.save(target); prepared.append(target)
    for mode in ('fused', 'local', 'context'):
        first = point2_pair_features(models, *prepared, mode)
        repeat = point2_pair_features(models, *prepared, mode)
        for a,b in zip(first, repeat):
            assert a.contextual.shape == (63,128)
            assert a.token_valid.sum() == 63
            assert a.physical_window_indices.tolist() == list(range(62,-1,-1))
            assert torch.isfinite(a.contextual).all()
            torch.testing.assert_close(a.contextual.norm(dim=-1), torch.ones(63), atol=1e-6, rtol=0)
            torch.testing.assert_close(a.contextual, b.contextual, atol=0, rtol=0)
    assert all(not module.training for module in models.image_model.modules())
    assert all(parameter.grad is None for parameter in models.image_model.parameters())
    for k, value in models.image_model.state_dict().items():
        assert torch.equal(value, before[k]), k  # Includes BatchNorm running buffers.

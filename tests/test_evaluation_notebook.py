"""Evaluation-only fixtures: random checkpoint creation, never optimization."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch

from dataset import prepare_image
from evaluate import region_mask, shared_regions
from evaluation_utils import (EvaluationSession, sample_random_pairs, localization_metrics,
    plot_pair_alignment, plot_representation_comparison, discrimination_metrics,
    retrieval_metrics, feature_diagnostics, evaluate_population, aggregate_summary,
    save_results, save_pair_artifacts, _constructed_negatives)
from parameters import Config
from train import build_loaders, build_model, save_checkpoint
from text_embedding import OrthogonalCharEmbedding


def notebook_source(first_line):
    book = json.loads((Path(__file__).parents[1] / 'notebooks/model_evaluation.ipynb').read_text())
    matches = [''.join(cell['source']) for cell in book['cells']
               if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith(first_line)]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture
def evaluation_fixture(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    rows = []
    for i in range(10):
        sides = {}
        split = 'train' if i < 6 else 'val' if i < 8 else 'test'
        for name in ('A', 'B'):
            image, text = f'{name}{i}.png', f'{name}{i}.txt'
            pixels = np.full((32, 160), 230, np.uint8)
            pixels[10:25, 20 + i:90 + i] = 40
            Image.fromarray(pixels).save(root / image)
            (root / text).write_text('سلام', encoding='utf-8')
            sides[name] = dict(line_image_path=image, text_original_path=text,
                               group_id=f'group{i}', bbox=[10, 2, 150, 30])
        rows.append(dict(pair_id=f'pair{i}', split=split, label_type='high_match', **sides))
    rows.append(dict(pair_id='negative', split='test', label_type='no_shared_content',
                     A=rows[8]['A'], B=rows[9]['B']))
    (root / 'dataset_manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    cfg = Config(dataset_type='real', split_mode='predefined', cnn_type='simple', cnn_pretrained=False,
                 image_height=32, image_width=128, num_workers=0, batch_size=2)
    loaders = build_loaders(root, cfg)
    ids = {name: [r['sample_id'] for r in loader.dataset.records]
           for name, loader in zip(('train', 'val', 'test'), loaders)}
    model = build_model(cfg)
    text = OrthogonalCharEmbedding(cfg.embedding_dim, cfg.text_vocab_size, cfg.text_embedding_seed)
    checkpoint = tmp_path / 'checkpoint_best.pt'
    save_checkpoint(checkpoint, model, None, 1, 1., cfg, text, ids)
    return root, checkpoint


def test_saved_membership_pair_labels_sampling_and_summary(evaluation_fixture, capsys):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu')
    assert len(session.view) == 4 and len(session.pairs) == 3
    assert sum(p['target'] == 1 for p in session.pairs) == 2
    assert session.provided_negative_count == 1
    assert all(not p['constructed_negative'] for p in session.pairs)
    for pair in session.pairs:
        assert all(Path(s['image']).stem in ('A8', 'B8', 'A9', 'B9') for s in pair['sides'])
    session.summary()
    assert str(root.resolve()) in capsys.readouterr().out
    for n in (0, 1, 2, 10):
        a = sample_random_pairs(session.pairs, n, seed=42)
        assert a == sample_random_pairs(session.pairs, n, seed=42)
        assert len(a) == min(n, 2)
        assert len(sample_random_pairs(session.pairs, n, aligned=False)) == min(n, 1)
    fake = [dict(sample_id=str(i), target=1) for i in range(20)]
    assert sample_random_pairs(fake, 5, seed=42) != sample_random_pairs(fake, 5, seed=43)


def test_image_only_cache_and_checkpoint_geometry(evaluation_fixture):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu')
    pair = session.pairs[0]
    before = {k: v.clone() for k, v in session.model.state_dict().items()}
    predictions = [session.evaluate_pair(pair, rep) for rep in ('local', 'context', 'fused')]
    for result in predictions:
        assert result['cosine'].shape == (7, 7)
        assert all(m.shape == (32, 160) for m in result['masks'])
        assert result['lines'][0]['physical'].tolist() == list(range(6, -1, -1))
        assert result['lines'][0]['geometry']['crop'] == [10, 2, 150, 30]
        assert all(not m[:, :10].any() and not m[:, 150:].any() for m in result['masks'])
    assert len(session.feature_cache) == 2
    for key, value in session.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    side = pair['sides'][0]
    _, tensor, _ = prepare_image(side['image'], (32, 128), bbox=side['bbox'])
    with torch.no_grad():
        expected = session.model(tensor[None])['fused'][0]
    torch.testing.assert_close(expected, predictions[-1]['lines'][0]['features']['fused'])
    # Prediction does not read annotations, labels or text contents.
    Path(side['text']).write_text('DIFFERENT', encoding='utf-8')
    hostile = dict(pair, label='no_shared_content', target=0,
                   annotations=[dict(mask='does-not-exist.png')] * 2)
    np.testing.assert_array_equal(session.predict_pair(hostile)['cosine'], predictions[-1]['cosine'])
    with pytest.raises(ValueError, match='outside saved'):
        session.get_line_features(dict(side, image=str(root / 'A0.png')))


def test_supported_masks_and_ground_truth_metrics(evaluation_fixture, tmp_path):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu')
    line = session.get_line_features(session.pairs[0]['sides'][0])
    cosine = np.zeros((7, 7))
    for i in range(4): cosine[i, i+2] = .95
    match = shared_regions(cosine, line['physical'], line['physical'], min_windows=4)
    assert len(match['regions']) == 1
    mask = region_mask(match['regions'], 0, line['geometry'], 32, 16)
    assert mask.any() and not mask.all() and np.array_equal(mask[0], mask[-1])
    path = tmp_path / 'gt.png'; Image.fromarray(mask).save(path)
    metrics, gt = localization_metrics(mask, {'mask': str(path)}, line, session.config, match['regions'], 0)
    assert metrics['iou'] == metrics['dice'] == 1.
    assert metrics['left_boundary_error_px'] == 0
    assert metrics['window_iou'] <= 1.
    Image.new('L', (5, 5)).save(path)
    with pytest.raises(ValueError, match='geometry mismatch'):
        localization_metrics(mask, {'mask': str(path)}, line, session.config, [], 0)
    with pytest.raises(ValueError, match='exact source_size'):
        localization_metrics(mask, {'intervals': [[10, 20]]}, line, session.config, [], 0)
    empty = region_mask([], 0, line['geometry'], 32, 16)
    assert not empty.any()
    assert localization_metrics(empty, {}, line, session.config, [], 0)[0]['status'] == 'unavailable'


def test_constructed_negatives_are_test_only_and_group_safe():
    records = [dict(side=dict(image=f'/tmp/im{i}', text=f'/tmp/t{i}', bbox=None),
                    group_id=str(i // 2), pair_ids=set(), anchors=set()) for i in range(6)]
    rows = _constructed_negatives([], records, 6, 42)
    assert rows == _constructed_negatives([], records, 6, 42)
    assert len(rows) == 6
    assert all(r['constructed_negative'] and r['group_ids'][0] != r['group_ids'][1] for r in rows)
    assert all(not any(r['annotations']) for r in rows)
    records[0]['pair_ids'] = records[4]['pair_ids'] = {'known-page-pair'}
    records[1]['anchors'] = records[5]['anchors'] = {'known-anchor'}
    pairs = _constructed_negatives([], records, 100, 42)
    assert not any({s['image'] for s in p['sides']} in ({'/tmp/im0', '/tmp/im4'}, {'/tmp/im1', '/tmp/im5'}) for p in pairs)


def test_optional_annotations_and_gt_figure(evaluation_fixture, tmp_path):
    root, checkpoint = evaluation_fixture
    with Image.open(root / 'A8.png') as source:
        mask = np.zeros((source.height, source.width), np.uint8)
        mask[:, 40:90] = 255
    Image.fromarray(mask).save(root / 'gt.png')
    annotations = tmp_path / 'annotations.jsonl'
    annotations.write_text(json.dumps(dict(A={'image': 'A8.png', 'alignment_mask_path': 'gt.png'},
        B={'image': 'B8.png'}, alignment_mask_meta={'method': 'unit-test annotation',
        'B': {'intervals_x': [[40, 90]], 'mask_size': [160, 32]}})))
    session = EvaluationSession(checkpoint, root, device='cpu', annotation_manifest=annotations)
    pair = next(p for p in session.pairs if p['sides'][0]['image'].endswith('A8.png'))
    result = session.evaluate_pair(pair)
    assert result['metrics']['a_status'] == result['metrics']['b_status'] == 'available'
    figure = plot_pair_alignment(session, result)
    figure.canvas.draw(); plt.close(figure)


def test_no_saved_split_fallback(evaluation_fixture):
    root, checkpoint = evaluation_fixture
    saved = torch.load(checkpoint, map_location='cpu')
    saved['split_ids']['test'].append('not-a-sample')
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='split identity'):
        EvaluationSession(checkpoint, root, device='cpu')


def test_aggregate_discrimination_retrieval_ties_and_render(evaluation_fixture, tmp_path, monkeypatch):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu', settings={'threshold': 1.})
    population = evaluate_population(session)
    assert population['evaluated_pairs'] == 3
    assert population['evaluated_unique_lines'] == 4
    assert all(r['pair_score'] == 0 for r in population['rows'])
    assert discrimination_metrics(population['rows'])['roc_auc'] == .5
    retrieved = retrieval_metrics(session, negatives=20)
    assert retrieved['queries'] == 2 and retrieved['top1'] == 0.
    assert retrieved['mrr'] == .5 and retrieved['top5'] is None
    session.split = 'train'
    assert retrieval_metrics(session)['status'] == 'unavailable'
    session.split = 'test'
    summary = aggregate_summary(population, retrieved)
    assert summary['localization'] == {}
    result = session.evaluate_pair(session.pairs[0])
    figure = plot_pair_alignment(session, result)
    figure.savefig(tmp_path / 'figure.png', dpi=30); plt.close(figure)
    figure, scores = plot_representation_comparison(session, session.pairs[0])
    figure.canvas.draw(); plt.close(figure)
    assert set(scores) == {'local', 'context', 'fused'}
    diagnostics = feature_diagnostics(session, max_lines=4, max_windows=20)
    assert diagnostics['representations']['fused']['norm_mean'] == pytest.approx(1., abs=1e-5)
    directory = save_results(session, population, summary, [], [], tmp_path / 'results', diagnostics)
    assert json.loads((directory / 'metrics.json').read_text())['split'] == 'test'
    save_pair_artifacts(session, result, directory / 'pair')
    with pytest.raises(FileExistsError): save_pair_artifacts(session, result, directory / 'pair')
    assert (directory / 'pair/correspondences.csv').read_text().startswith('region,cosine,reward')


def test_notebook_structure_and_python_cells():
    path = Path(__file__).parents[1] / 'notebooks/model_evaluation.ipynb'
    book = json.loads(path.read_text())
    assert book['nbformat'] == 4
    code = [c for c in book['cells'] if c['cell_type'] == 'code']
    assert 'NUM_SAMPLES = 5' in ''.join(code[0]['source'])
    # Existing saved review figures are user results; notebook edits retain them.
    assert all(isinstance(c['outputs'], list) for c in code)
    for i, cell in enumerate(code):
        compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec')


def test_window_recognition_uses_fused_full_alphabet_and_position_dtw(evaluation_fixture):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu')
    pair = next(p for p in session.pairs if p['target'] == 1)
    side = pair['sides'][0]
    Path(side['text']).write_text('مم', encoding='utf-8')
    result = session.evaluate_pair(pair)
    namespace = dict(session=session, evaluations=[], TOP_K=5, np=np, pd=__import__('pandas'),
                     torch=torch, Path=Path)
    from text_embedding import ARABIC_LETTERS, clean_letters
    from dtw import cosine_similarity_matrix, letter_cost_matrix, hard_dtw_path
    from evaluate import source_interval
    namespace.update(ARABIC_LETTERS=ARABIC_LETTERS, clean_letters=clean_letters,
                     cosine_similarity_matrix=cosine_similarity_matrix,
                     letter_cost_matrix=letter_cost_matrix, hard_dtw_path=hard_dtw_path,
                     source_interval=source_interval)
    exec(notebook_source('def ensure_recognition_alphabet():'), namespace)
    namespace['ensure_recognition_alphabet']()
    exec(notebook_source('def cosine_checked('), namespace)
    diagnostic = namespace['recognize_line'](session, side, result['lines'][0], 5)
    frame = diagnostic['frame']
    assert len(frame) == len(result['lines'][0]['features']['fused'])
    assert frame['physical_window_index'].tolist() == list(range(len(frame)))
    assert frame['model_sequence_index'].tolist() == list(range(len(frame) - 1, -1, -1))
    assert diagnostic['transcript_cosine'].shape == (len(frame), 2)
    np.testing.assert_allclose(diagnostic['transcript_cosine'][:, 0],
                               diagnostic['transcript_cosine'][:, 1])
    assert all(set(positions) <= {0, 1} for positions in frame['dtw_all_text_positions'])
    assert all(-1.00001 <= score <= 1.00001 for score in frame['top1_score'])
    assert all(frame['top1_score'] >= frame['top2_score'])

    # The complete alphabet score is independent of the transcript, even for
    # characters absent from that transcript.
    Path(side['text']).write_text('با', encoding='utf-8')
    changed = namespace['recognize_line'](session, side, result['lines'][0], 5)
    np.testing.assert_allclose(changed['alphabet_cosine'], diagnostic['alphabet_cosine'])
    assert changed['frame']['cos_top1_letter'].tolist() == frame['cos_top1_letter'].tolist()

    # Actual source-pixel character boxes make accuracy and position metrics
    # available; one box can overlap several windows without inventing others.
    first = frame.iloc[0]
    side['character_boxes'] = [dict(text_position=0, character='ب',
                                    x_start=first.source_x_start, x_end=first.source_x_end)]
    annotated = namespace['recognize_line'](session, side, result['lines'][0], 5)
    exec(notebook_source('def recognition_metrics(').split('if ENABLE_RECOGNITION_METRICS', 1)[0], namespace)
    metrics, categories = namespace['recognition_metrics'](annotated)
    assert metrics['gt_evaluated_windows'] >= 1
    assert metrics['dtw_position_aware_accuracy'] is not None
    assert sum(v['count'] for v in categories.values()) == metrics['gt_evaluated_windows']


def test_cached_samples_threshold_sweeps_and_independent_slots(evaluation_fixture, tmp_path, monkeypatch):
    import random
    import time
    from collections import Counter
    import pandas as pd
    from dtw import cosine_similarity_matrix
    from evaluate import shared_regions, local_repeat_regions, region_mask
    from evaluation_utils import load_pair_ground_truth, compute_pair_metrics, shuffled_positive_pairs

    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu')
    positives = [pair for pair in session.pairs if pair['target'] == 1]
    assert len(positives) == 2
    gt_path = tmp_path / 'gt.png'
    gt_pixels = np.zeros((32, 160), dtype=np.uint8)
    gt_pixels[8:23, 20:90] = 255
    Image.fromarray(gt_pixels).save(gt_path)
    for pair in positives:
        pair['annotations'] = [dict(mask=str(gt_path)), dict(mask=str(gt_path))]

    shown = []
    namespace = dict(
        session=session, RANDOM_SEED=42, REQUIRE_GROUND_TRUTH=True,
        REPRESENTATION='fused', NUM_SAMPLES=5, np=np, pd=pd, torch=torch,
        random=random, time=time, Counter=Counter,
        shuffled_positive_pairs=shuffled_positive_pairs,
        cosine_similarity_matrix=cosine_similarity_matrix,
        shared_regions=shared_regions, local_repeat_regions=local_repeat_regions,
        region_mask=region_mask,
        load_pair_ground_truth=load_pair_ground_truth,
        compute_pair_metrics=compute_pair_metrics,
        plot_pair_alignment=lambda *_: plt.figure(),
        display=lambda value: shown.append(value), plt=plt,
    )
    exec(notebook_source('pool = shuffled_positive_pairs('), namespace)
    exec(notebook_source('# EVALUATION-ONLY MATCHING SETTINGS.'), namespace)
    exec(notebook_source('KEY_METRICS ='), namespace)
    sample_slots = namespace['sample_slots']

    namespace['show_random_sample'](1)
    first_id = sample_slots[1]['pair']['sample_id']
    namespace['show_random_sample'](2)
    second_id = sample_slots[2]['pair']['sample_id']
    assert first_id != second_id
    first_base = sample_slots[1]['base']
    first_gt = sample_slots[1]['ground_truth']
    reference = session.evaluate_pair(sample_slots[1]['pair'])
    cached_reference = namespace['evaluate_cached_sample'](
        1, session.settings['threshold'], alignment_mode='smith_waterman',
        contrast_margin=session.settings['contrast_margin'])
    np.testing.assert_array_equal(cached_reference['masks'][0], reference['masks'][0])
    np.testing.assert_array_equal(cached_reference['masks'][1], reference['masks'][1])
    assert cached_reference['metrics']['pair_mean_iou'] == pytest.approx(
        reference['metrics']['pair_mean_iou'])
    before = len(session.feature_cache)
    assert before == 4  # two model forwards per selected pair

    # Any attempt to reenter model inference or GT loading during tuning fails.
    monkeypatch.setattr(session, 'get_line_features', lambda *_: pytest.fail('feature lookup during tuning'))
    monkeypatch.setattr(session, 'evaluate_pair', lambda *_: pytest.fail('evaluate_pair during tuning'))
    monkeypatch.setattr(session, 'predict_pair', lambda *_: pytest.fail('predict_pair during tuning'))
    monkeypatch.setattr(session.model, 'forward', lambda *_: pytest.fail('model forward during tuning'))
    namespace['load_pair_ground_truth'] = lambda *_: pytest.fail('GT reload during tuning')

    results = [namespace['evaluate_cached_sample'](1, value) for value in (.2, .4, .6)]
    assert all(result['pair']['sample_id'] == first_id for result in results)
    assert all(result['cosine'] is first_base['cosine'] for result in results)
    assert all(result['lines'] is first_base['lines'] for result in results)
    assert all(result['ground_truth'] is first_gt for result in results)
    assert [result['settings']['threshold'] for result in results] == [.2, .4, .6]
    assert len(session.feature_cache) == before
    assert namespace['evaluate_cached_sample'](1, .4) is results[1]
    penalty_a = namespace['evaluate_cached_sample'](1, .4, repeat_penalty=.03)
    penalty_b = namespace['evaluate_cached_sample'](1, .4, repeat_penalty=.09)
    assert penalty_a is not penalty_b
    assert penalty_a['cosine'] is penalty_b['cosine'] is first_base['cosine']
    assert len(session.feature_cache) == before

    exec(notebook_source('TUNE_SLOT ='), namespace)
    single = namespace['sweep_thresholds'](1, [.2, .4])
    multi = namespace['sweep_cached_slots']([1, 2, 3], [.2, .4])
    assert list(single['threshold']) == [.2, .4]
    assert list(multi['samples']) == [2, 2]
    exec(notebook_source('COMPARE_SLOT = 1'), namespace)
    compared, comparison_table = namespace['compare_alignment_modes'](1, show_paths=True)
    assert compared['smith_waterman']['cosine'] is compared['local_repeat_dtw']['cosine']
    assert comparison_table['mode'].tolist() == ['smith_waterman', 'local_repeat_dtw']
    assert pd.isna(comparison_table.iloc[0]['horizontal_repeats'])
    exec(notebook_source('EXPERIMENT_SLOTS = [1]'), namespace)
    experiment = namespace['experiment_repeat_settings']([1, 2], [.4], [.03, .09], [2])
    assert len(experiment) == 2 and set(experiment['samples']) == {2}
    assert len(session.feature_cache) == before

    # Rerunning slot 1 replaces only that slot. Both pair identities are
    # genuine manifest positives, and cached features prevent new forwards.
    namespace['load_pair_ground_truth'] = load_pair_ground_truth
    monkeypatch.undo()
    namespace['show_random_sample'](1)
    assert sample_slots[1]['pair']['sample_id'] != first_id
    assert sample_slots[2]['pair']['sample_id'] == second_id
    assert len(session.feature_cache) == before
    assert all(slot['pair'] in positives for slot in sample_slots.values())


def test_sample_count_controls_only_its_own_cells():
    book = json.loads((Path(__file__).parents[1] / 'notebooks/model_evaluation.ipynb').read_text())
    sample_cells = [''.join(book['cells'][i]['source']) for i in range(13, 32, 2)]
    assert len(sample_cells) == 10
    for count in (1, 3, 5):
        called = []
        for source in sample_cells:
            exec(source, dict(NUM_SAMPLES=count, show_random_sample=called.append))
        assert called == list(range(1, count + 1))

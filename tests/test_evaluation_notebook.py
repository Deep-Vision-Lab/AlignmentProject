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


@pytest.fixture
def arabic_folder_fixture(tmp_path):
    import shutil
    from test_dataset import make_book_lines
    root = make_book_lines(tmp_path / 'ArabicData', pages=6)
    for folder in ('alignedPairs', 'Unalignedlines'):
        for side in 'AB':
            for kind in ('images', 'texts', 'BoundingBoxes'):
                (root / folder / (kind + side)).mkdir(parents=True)
    ids = {split: [] for split in ('train', 'val', 'test')}
    for side in 'AB':
        rows = [json.loads(line) for line in (root / f'Book{side}/lines.jsonl').read_text().splitlines()]
        for i, row in enumerate(rows):
            Image.new('L', (80, 32), 80 + i + (60 if side == 'B' else 0)).save(root / row['image_path'])
            page, number = map(int, row['line_id'].split('_')[1:])
            transcript = 'رحيم' if side == 'B' and number == 1 else 'سلام'
            (root / row['text_path']).write_text(transcript, encoding='utf-8')
            ids[('train', 'val', 'test')[page // 2]].append(f'Book{side}:' + row['line_id'])
            if number > 1:
                continue
            folder = 'alignedPairs' if number == 0 else 'Unalignedlines'
            stem = f'pair_{page:06d}__A_line_001__B_line_001' if number == 0 else row['line_id']
            for key, kind, ext in (('image_path', 'images', '.png'), ('text_path', 'texts', '.txt')):
                shutil.copyfile(root / row[key], root / folder / (kind + side) / (stem + ext))
            payload = dict(line_id=row['line_id'], image_size=[80, 32],
                crop_status='VERIFIED_SOURCE_CROP_PIXELS', subwords=[dict(text=transcript,
                    line_bbox_xyxy=[10, 4, 60, 28], status='VERIFIED_SOURCE_CROP_PIXELS')])
            (root / folder / ('BoundingBoxes' + side) / (stem + '.json')).write_text(json.dumps(payload))
    (root / 'alignedPairs/masksA').mkdir()
    mask = np.zeros((32, 80), np.uint8); mask[:, 20:50] = 255
    Image.fromarray(mask).save(root / 'alignedPairs/masksA/pair_000004__A_line_001__B_line_001.png')
    cfg = Config(dataset_type='real', cnn_type='simple', cnn_pretrained=False,
                 image_height=32, image_width=64, num_workers=0, batch_size=2)
    checkpoint = tmp_path / 'folder.pt'
    save_checkpoint(checkpoint, build_model(cfg), None, 1, 1., cfg,
                    OrthogonalCharEmbedding(cfg.embedding_dim, cfg.text_vocab_size,
                                            cfg.text_embedding_seed), ids)
    return root, checkpoint


def test_arabic_folder_partners_unmatched_sampling_split_and_masks(arabic_folder_fixture):
    from evaluation_utils import positive_pairs, _side_key
    root, checkpoint = arabic_folder_fixture
    session = EvaluationSession(checkpoint, root, device='cpu', negative_count=10)
    assert session.pair_source == 'folders'
    positives = positive_pairs(session)
    assert len(positives) == 2 and len(session.pairs) == 4
    assert session.folder_summary['excluded_aligned_pairs'] == 4
    assert session.folder_summary['eligible_unmatched_A'] == session.folder_summary['eligible_unmatched_B'] == 2
    assert len(session.allowed) == 10 and session.provided_negative_count == 0
    assert all('/alignedPairs/' in side['image'] for p in positives for side in p['sides'])
    negatives = [p for p in session.pairs if p['target'] == 0]
    assert all(p['constructed_negative'] and '/Unalignedlines/' in side['image']
               for p in negatives for side in p['sides'])
    assert all(p['group_ids'][0] != p['group_ids'][1] for p in negatives)
    assert all('_04_' in side['image'] or '_05_' in side['image'] for p in negatives for side in p['sides'])
    other = EvaluationSession(checkpoint, root, device='cpu', negative_count=10)
    assert [p['sample_id'] for p in session.pairs] == [p['sample_id'] for p in other.pairs]
    result = session.evaluate_pair(positives[0])
    assert result['ground_truth'][0][:, 20:50].all() and not result['ground_truth'][0][:, :20].any()
    assert result['ground_truth'][1][:, 10:60].all()
    assert 'supplied mask' in result['metrics']['ground_truth_provenance']
    assert 'derived' in result['metrics']['ground_truth_provenance']
    side = positives[0]['sides'][0]
    canonical = session.allowed[session.line_aliases[_side_key(side)]]['side']
    assert session.get_line_features(side) is session.get_line_features(canonical)
    assert len(session.feature_cache) == 2
    with pytest.raises(ValueError, match='outside saved'):
        session.get_line_features(dict(side, image=str(root / 'alignedPairs/imagesA/pair_000000__A_line_001__B_line_001.png')))


def test_arabic_folder_review_flags_copy_verification_and_negatives_off(arabic_folder_fixture):
    from evaluation_utils import positive_pairs
    root, checkpoint = arabic_folder_fixture
    box = root / 'alignedPairs/BoundingBoxesB/pair_000005__A_line_001__B_line_001.json'
    payload = json.loads(box.read_text()); payload['crop_status'] = 'CROP_GEOMETRY_MISMATCH'
    box.write_text(json.dumps(payload))
    session = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False)
    assert len(positive_pairs(session)) == 1
    assert len(positive_pairs(session, require_ground_truth=False)) == 2
    assert not any(p['constructed_negative'] for p in session.pairs)
    (root / 'alignedPairs/textsA/pair_000004__A_line_001__B_line_001.txt').write_text('changed')
    with pytest.raises(ValueError, match='Folder copy differs'):
        EvaluationSession(checkpoint, root, device='cpu')


def test_notebook_explicitly_selects_arabic_data_folders():
    setup = notebook_source('# MODEL/DATA SETUP.')
    assert 'DATASET = "DataSet/ArabicData"' in setup
    assert 'PAIR_SOURCE = "folders"' in setup
    assert 'pair_source=PAIR_SOURCE' in setup
    assert 'ArabicDataset' not in setup


def test_content_crop_reaches_model_input_and_all_representations(evaluation_fixture, monkeypatch):
    from evaluation_utils import indexing_sanity
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False,
                                crop_override='content')
    pair = session.pairs[0]
    captured = []
    forward = session.model.forward
    def record_input(tensor, *args, **kwargs):
        captured.append(tensor.detach().cpu().clone())
        return forward(tensor, *args, **kwargs)
    monkeypatch.setattr(session.model, 'forward', record_input)
    allowed, aliases = dict(session.allowed), dict(session.line_aliases)
    results = [session.evaluate_pair(pair, rep) for rep in ('local', 'fused', 'context')]
    assert len(captured) == len(session.feature_cache) == 2
    # The toy line is solid ink within these bounds. A crop applied only to the
    # figure would still send white margins into this model input.
    assert results[0]['lines'][0]['geometry']['crop'] == [28, 10, 98, 25]
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], (40/255-.449)/.226))
    assert session.view.crop == session.config.crop == 'content'
    assert all(r['lines'][0] is results[0]['lines'][0] for r in results)
    assert session.allowed == allowed and session.line_aliases == aliases
    for result in results:
        indexing_sanity(result, session.config)
        assert result['masks'][0].shape == (32, 160)
    figure = plot_pair_alignment(session, results[0], diagnostic=True, show_input_crop=True)
    mask_axis = next(ax for ax in figure.axes if ax.get_title() == 'A: predicted mask')
    assert mask_axis.images[0].get_array().shape == (15, 70)
    assert tuple(mask_axis.images[0].get_extent()) == (27.5, 97.5, 24.5, 9.5)
    assert 'heatmap rows' in mask_axis.get_xlabel()
    assert mask_axis.get_xticklabels()[0].get_text() == '0'
    heatmap_axis = figure.axes[0]
    assert [label.get_text() for label in mask_axis.get_xticklabels()] == [
        label.get_text() for label in heatmap_axis.get_yticklabels()]
    plt.close(figure)
    with pytest.raises(ValueError, match='outside saved'):
        session.get_line_features(dict(pair['sides'][0], image=str(root / 'A0.png')))
    setup = notebook_source('# MODEL/DATA SETUP.')
    assert 'EVALUATION_CROP = "content"' in setup
    assert 'crop_override=EVALUATION_CROP' in setup
    assert 'NotebookBoundingBoxSession' not in setup


@pytest.mark.parametrize('physical, expected_centers, rtl', [
    ([0, 1, 3], [46, 62, 86], False),
    ([3, 1, 0], [86, 62, 46], True),
])
def test_image_window_ticks_follow_filtered_heatmap_order(physical, expected_centers, rtl):
    from evaluation_utils import set_line_window_axis
    # Source crop starts at x=30, resizing doubles source-space stride, and
    # physical window 3 is clipped by the crop's right edge. Window 2 was
    # excluded: the heatmap therefore has compact indices 0, 1, 2.
    pixels = np.zeros((24, 120), dtype=np.uint8)
    line = dict(physical=np.array(physical), logical=np.array([0, 2, 3]),
                geometry=dict(crop=[30, 4, 94, 20], source_size=[120, 24], scale_x=.5))
    figure, axis = plt.subplots()
    axis.imshow(pixels, cmap='gray')
    axis.set_xlim(29.5, 93.5)
    extent = axis.images[0].get_extent()
    set_line_window_axis(axis, line, 16, 8, side='B', rtl=rtl)
    np.testing.assert_allclose(axis.get_xticks(), expected_centers)
    assert [label.get_text() for label in axis.get_xticklabels()] == ['0', '1', '2']
    assert 'heatmap columns' in axis.get_xlabel()
    assert ('RTL' if rtl else 'LTR') in axis.get_xlabel()
    assert axis.fmt_xdata(expected_centers[0]) == 'window 0'
    assert axis.get_xlim() == (29.5, 93.5)
    assert axis.images[0].get_extent() == extent
    np.testing.assert_array_equal(axis.images[0].get_array(), pixels)
    plt.close(figure)


def test_rtl_image_axis_matches_heatmap_decade_ticks():
    from evaluation_utils import set_line_window_axis, _window_tick_indices
    line = dict(physical=np.arange(62, -1, -1),
                geometry=dict(crop=[20, 0, 544, 24], source_size=[560, 24], scale_x=2))
    figure, axis = plt.subplots()
    axis.imshow(np.zeros((24, 560)), cmap='gray')
    set_line_window_axis(axis, line, 32, 16, side='B', rtl=True)
    # Heatmap column 0 is source window 62; column 60 is source window 2.
    np.testing.assert_allclose(axis.get_xticks(), [524, 444, 364, 284, 204, 124, 44])
    ticks = sorted(zip(axis.get_xticks(), axis.get_xticklabels()))
    assert [label.get_text() for _, label in ticks] == ['60', '50', '40', '30', '20', '10', '0']
    assert _window_tick_indices(63).tolist() == [0, 10, 20, 30, 40, 50, 60]
    plt.close(figure)


def test_notebook_folder_mask_missing_black_and_accuracy(tmp_path):
    from evaluation_utils import compute_pair_metrics
    namespace = dict(np=np, Path=Path, json=json, EvaluationSession=EvaluationSession,
                     compute_pair_metrics=compute_pair_metrics, plot_pair_alignment=plot_pair_alignment)
    exec(notebook_source('# FOLDER MASK GROUND TRUTH'), namespace)
    sides = []
    for name, size in [('A', (4, 3)), ('B', (5, 2))]:
        folder = tmp_path / ('images' + name)
        folder.mkdir()
        image = folder / 'sample.png'
        Image.new('L', size, 255).save(image)
        sides.append(dict(image=str(image)))
    (tmp_path / 'masksA').mkdir()
    truth_a = np.zeros((3, 4), np.uint8)
    truth_a[1, 1:3] = 255
    mask_path = tmp_path / 'masksA/sample.png'
    Image.fromarray(truth_a).save(mask_path)
    pair = dict(sample_id='sample', sides=sides)
    truth, provenance = namespace['load_folder_masks'](pair)
    assert truth[0].sum() == 2 and truth[1].shape == (2, 5) and not truth[1].any()
    assert 'missing folder mask -> black' in provenance
    pred = [np.zeros((3, 4), bool), np.zeros((2, 5), bool)]
    pred[0][1, 1] = True
    pred[1][0, 0] = True
    result = dict(pair=pair, masks=pred, ground_truth=truth, metrics={})
    namespace['add_folder_mask_accuracy'](result)
    m = result['metrics']
    assert m['a_accuracy'] == pytest.approx(11/12)
    assert m['b_accuracy'] == pytest.approx(9/10)
    assert m['pair_pixel_accuracy'] == pytest.approx(20/22)
    assert (m['pair_tp'], m['pair_tn'], m['pair_fp'], m['pair_fn']) == (1, 19, 1, 1)
    assert not m['pair_alignment_correct']
    summary = namespace['mask_accuracy_summary']([m])
    assert summary['missing_masks_black'] == 1 and summary['pixel_accuracy'] == pytest.approx(20/22)
    mask_path.unlink()
    black, _ = namespace['load_folder_masks'](pair)
    for predicted, expected in [(black, 1.), ([np.ones_like(g) for g in black], 0.)]:
        scored = dict(pair=pair, masks=predicted, ground_truth=black, metrics={})
        namespace['add_folder_mask_accuracy'](scored)
        assert scored['metrics']['pair_pixel_accuracy'] == expected
        assert scored['metrics']['pair_alignment_correct'] == bool(expected)
    Image.new('L', (1, 1), 0).save(mask_path)
    with pytest.raises(ValueError, match='size mismatch'):
        namespace['load_folder_masks'](pair)


def test_notebook_mask_session_never_derives_gt_from_boxes(evaluation_fixture, monkeypatch):
    from evaluation_utils import compute_pair_metrics
    import evaluation_utils
    namespace = dict(np=np, Path=Path, json=json, EvaluationSession=EvaluationSession,
                     compute_pair_metrics=compute_pair_metrics, plot_pair_alignment=plot_pair_alignment)
    exec(notebook_source('# FOLDER MASK GROUND TRUTH'), namespace)
    root, checkpoint = evaluation_fixture
    for name in 'AB':
        folder = root.parent / ('masks' + name)
        folder.mkdir()
        pixels = np.zeros((32, 160), np.uint8)
        pixels[:, 30:90] = 255
        Image.fromarray(pixels).save(folder / (name + '8.png'))
    monkeypatch.setattr(evaluation_utils, 'boxes_ground_truth', lambda *_: pytest.fail('derived GT used'))
    session = namespace['FolderMaskEvaluationSession'](checkpoint, root, device='cpu', construct_negatives=False)
    positive = next(p for p in session.pairs if [Path(s['image']).stem for s in p['sides']] == ['A8', 'B8'])
    negative = next(p for p in session.pairs if [Path(s['image']).stem for s in p['sides']] == ['A9', 'B9'])
    assert positive['target'] == 1 and negative['target'] == 0
    for rep in ('local', 'fused', 'context'):
        result = session.evaluate_pair(negative, rep)
        assert not any(g.any() for g in result['ground_truth'])
        assert result['metrics']['evaluation_category'] == 'true_no_overlap'
        for index, name in enumerate('ab'):
            assert result['metrics'][name + '_accuracy'] == pytest.approx(
                np.mean((result['masks'][index] > 0) == result['ground_truth'][index]))
    prediction = session.predict_pair(positive)
    hostile = dict(positive, target=0, annotations=[dict(mask='nonexistent')] * 2)
    other = session.predict_pair(hostile)
    np.testing.assert_array_equal(prediction['cosine'], other['cosine'])
    for a, b in zip(prediction['masks'], other['masks']):
        np.testing.assert_array_equal(a, b)


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


def test_notebook_setup_imports_project_dependencies():
    import subprocess
    import sys
    setup = notebook_source('# MODEL/DATA SETUP.')
    script = '''
namespace = {}
exec(SETUP, namespace)
import dataset, dataloader, text_embedding, evaluation_utils, train, parameters, model
assert text_embedding.read_side_text({'raw_text': ' سلام '}) == 'سلام'
assert dataset.read_side_text is text_embedding.read_side_text
assert evaluation_utils.read_side_text is text_embedding.read_side_text
assert dataloader.AlignmentDataset is dataset.AlignmentDataset
assert train.Config is parameters.Config
assert train.AlignmentModel is model.AlignmentModel
assert namespace['DATASET'] == 'DataSet/ArabicData'
'''
    subprocess.run([sys.executable, '-c', 'SETUP = ' + repr(setup) + '\n' + script],
                   cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True)


def test_old_notebook_setup_recovers_cached_missing_transcript_reader():
    import subprocess
    import sys
    script = '''
import sys, text_embedding
# Reproduce the cached module from before read_side_text was added.
del text_embedding.read_side_text
for name in ('evaluation_utils', 'evaluate', 'alignment_candidates', 'train',
             'losses', 'dtw', 'dataloader', 'dataset'):
    sys.modules.pop(name, None)
import evaluation_utils, dataset
assert text_embedding.read_side_text({'raw_text': ' سلام '}) == 'سلام'
assert dataset.read_side_text is text_embedding.read_side_text
assert evaluation_utils.read_side_text is text_embedding.read_side_text
'''
    subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parents[1],
                   check=True, capture_output=True, text=True)


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
    session = EvaluationSession(checkpoint, root, split='val', device='cpu')
    positives = [pair for pair in session.pairs if pair['target'] == 1]
    assert len(positives) == 2
    gt_path = tmp_path / 'gt.png'
    gt_pixels = np.zeros((32, 160), dtype=np.uint8)
    gt_pixels[8:23, 20:90] = 255
    Image.fromarray(gt_pixels).save(gt_path)
    for pair in positives:
        pair['annotations'] = [dict(mask=str(gt_path)), dict(mask=str(gt_path))]

    shown = []
    plotted = []
    def capture_plot(_, result, **kwargs):
        plotted.append(result)
        return plt.figure()
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
        plot_pair_alignment=capture_plot,
        display=lambda value: shown.append(value), plt=plt,
    )
    exec(notebook_source('pool = shuffled_positive_pairs('), namespace)
    exec(notebook_source('# EVALUATION-ONLY MATCHING SETTINGS.'), namespace)
    exec(notebook_source('KEY_METRICS ='), namespace)
    sample_slots = namespace['sample_slots']

    # Select once, then execute three distinct representation cells.
    namespace['N_POSITIVE_SAMPLES'] = 1
    exec(notebook_source('# Sample 1 — Select pair'), namespace)
    assert not plotted
    for rep in ('LOCAL', 'FUSED', 'CONTEXT'):
        exec(notebook_source(f'# Sample 1 — {rep}:'), namespace)
    first_id = sample_slots[1]['pair']['sample_id']
    namespace['show_random_sample'](2)
    second_id = sample_slots[2]['pair']['sample_id']
    assert first_id != second_id
    # A single sample-cell run displays all representations, in vertical order,
    # for one pair, with identical matcher settings and shared line features.
    assert [r['representation'] for r in plotted] == ['local', 'fused', 'context',
                                                   'local', 'context', 'fused']
    assert [r['pair']['sample_id'] for r in plotted] == [first_id] * 3 + [second_id] * 3
    for slot in (1, 2):
        sample = sample_slots[slot]
        assert list(sample['results']) == ['local', 'context', 'fused']
        assert all(r['settings'] == sample['results']['local']['settings']
                   for r in sample['results'].values())
        assert all(r['lines'][0] is sample['base']['lines'][0]
                   for r in sample['results'].values())
        assert sample['result'] is sample['results']['fused']
    first_base = sample_slots[1]['base']
    first_gt = sample_slots[1]['ground_truth']
    reference = session.evaluate_pair(sample_slots[1]['pair'])
    cached_reference = namespace['evaluate_cached_sample'](
        1, session.settings['threshold'],
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

    # A single representation cell can be rerun without selecting a new pair
    # or displaying the other two representations.
    count = len(plotted)
    exec(notebook_source('# Sample 1 — CONTEXT:'), namespace)
    assert len(plotted) == count + 1
    assert plotted[-1]['representation'] == 'context'
    assert plotted[-1]['pair']['sample_id'] == first_id

    # Re-displaying the saved slot also emits all three, without selection,
    # feature retrieval, model inference, or annotation reload.
    repeated = namespace['show_existing_sample'](1)
    assert [r['representation'] for r in plotted[-3:]] == ['local', 'context', 'fused']
    assert all(r['pair']['sample_id'] == first_id for r in repeated.values())
    assert len(session.feature_cache) == before

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
    assert sample_slots[1]['pair']['sample_id'] == first_id  # exhausted pool never wraps
    assert sample_slots[2]['pair']['sample_id'] == second_id
    assert len(session.feature_cache) == before
    assert all(slot['pair'] in positives for slot in sample_slots.values())


def test_sample_count_controls_only_its_own_cells():
    book = json.loads((Path(__file__).parents[1] / 'notebooks/model_evaluation.ipynb').read_text())
    sample_cells = [''.join(cell['source']) for cell in book['cells']
                    if cell.get('id', '').startswith('align-sample-')]
    assert len(sample_cells) == 10
    for count in (1, 3, 5):
        called = []
        for source in sample_cells:
            exec(source, dict(N_POSITIVE_SAMPLES=count, select_random_sample=called.append))
        assert called == list(range(1, count + 1))

    for slot in range(1, 11):
        cells = [cell for cell in book['cells']
                 if cell.get('id', '').startswith(f'align-view-{slot:02d}-')]
        assert len(cells) == 3
        calls = []
        for cell in cells:
            exec(''.join(cell['source']), dict(N_POSITIVE_SAMPLES=10,
                 show_sample_representation=lambda s, r: calls.append((s, r))))
        assert calls == [(slot, rep) for rep in ('local', 'fused', 'context')]

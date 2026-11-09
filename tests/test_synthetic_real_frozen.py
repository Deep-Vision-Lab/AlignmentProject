"""Rendering geometry, source-label validity and the image-only boundary."""
import json
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.eval import synthetic_real_frozen as frozen


@pytest.fixture
def source(tmp_path):
    image = tmp_path / 'source.png'
    Image.new('RGB', (180, 40), 'white').save(image)
    groups = [dict(text=text, line_bbox_xyxy=[150 - i * 35, 4, 175 - i * 35, 35],
                   status='SOURCE_LINE_COORDINATES')
              for i, text in enumerate(['خريف', 'حجره', 'سعدت', 'بثوق'])]
    annotation = dict(image_sha256=frozen.sha256(image), image_size=[180, 40], subwords=groups)
    path = tmp_path / 'BookC/BoundingBoxes/example.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(annotation))
    record = dict(sample_id='BookC:example', sides=[dict(image=str(image), raw_text=''.join(g['text'] for g in groups))])
    return tmp_path, record, annotation, path


def test_source_annotations_require_matching_image_text_and_geometry(source):
    root, record, annotation, path = source
    assert frozen.validate_annotation(record, root)['size'] == [180, 40]
    annotation['image_sha256'] = 'stale'
    path.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match='hash/size'):
        frozen.validate_annotation(record, root)
    annotation['image_sha256'] = frozen.sha256(record['sides'][0]['image'])
    annotation['subwords'][0]['line_bbox_xyxy'][2] = 181
    path.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match='outside'):
        frozen.validate_annotation(record, root)
    annotation['subwords'][0]['line_bbox_xyxy'][2] = 175
    annotation['subwords'][0]['text'] = 'غير'
    path.write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match='sequence'):
        frozen.validate_annotation(record, root)


def test_rtl_reversal_moves_both_shared_spans_and_keeps_distractor_unmasked(tmp_path):
    font = frozen.ROOT / 'Fonts/Amiri-Regular.ttf'
    if not font.is_file():
        pytest.skip('Optional local font unavailable')
    segments = [dict(kind='shared', text='خريف'), dict(kind='distractor', text='حجره'), dict(kind='shared', text='سعدت')]
    rendered, _ = frozen.render_segmented_positive(segments, font, tmp_path/'line.png', tmp_path/'mask.png',
        width=512, height=96, font_size=50, padding=8, gap_min_px=16, gap_max_px=16, rng=random.Random(2))
    boxes = [s['bbox_px'] for s in rendered]
    assert boxes[2][2] < boxes[1][0] < boxes[1][2] < boxes[0][0]
    mask = np.asarray(Image.open(tmp_path/'mask.png')) > 0
    assert mask[:, boxes[0][0]:boxes[0][2]].all()
    assert mask[:, boxes[2][0]:boxes[2][2]].all()
    assert not mask[:, boxes[1][0]:boxes[1][2]].any()
    reversed_rendered, _ = frozen.render_segmented_positive(segments[::-1], font, tmp_path/'reversed.png', tmp_path/'reversed_mask.png',
        width=512, height=96, font_size=50, padding=8, gap_min_px=16, gap_max_px=16, rng=random.Random(2))
    assert reversed_rendered[0]['text'] == segments[2]['text']
    assert reversed_rendered[2]['text'] == segments[0]['text']
    assert frozen.sha256(tmp_path/'line.png') != frozen.sha256(tmp_path/'reversed.png')


def test_negative_guard_detects_short_words_inside_unspaced_c():
    anchor = dict(record=dict(sides=[dict(raw_text='منالسماء')]), annotation=dict(subwords=[dict(text='من'), dict(text='السماء')]))
    vocabulary = {'من': 2, 'منطق': 1, 'خريف': 1, 'حجره': 1, 'سعدت': 1, 'بثوق': 1}
    phrase = frozen.unrelated_phrase(anchor, vocabulary, random.Random(42))
    assert 'منطق' not in phrase
    assert 'من' not in frozen.canonical(phrase)
    assert frozen.safe_negative('منالسماء', phrase, negative_ngram=3, min_overlap_word_chars=2)


def test_predictor_receives_only_images_not_annotations(monkeypatch):
    seen = {}
    features = dict(features={}, physical=np.arange(2), geometry={})
    session = SimpleNamespace(get_line_features=lambda side: features,
        allowed={'member': dict(side=dict(image='real.png', text='private_reference.txt', raw_text='never forwarded'))})
    monkeypatch.setattr(frozen, 'synthetic_features', lambda session, path: features)
    def predictor(session, pair, lines):
        seen.update(pair)
        return 'prediction'
    monkeypatch.setattr(frozen, 'predict_cached_pair', predictor)
    assert frozen.predict_images(session, 'real.png', 'synthetic.png') == 'prediction'
    assert set(seen) == {'sample_id', 'sides'}
    assert seen['sides'] == [dict(image='real.png', bbox=None), dict(image='synthetic.png', bbox=None)]
    with pytest.raises(ValueError, match='saved-test'):
        frozen.predict_images(session, 'outside.png', 'synthetic.png')


def test_state_fingerprint_detects_buffer_adaptation_without_parameter_updates():
    model = torch.nn.BatchNorm1d(3).eval().requires_grad_(False)
    before = frozen.state_sha256(model)
    with torch.inference_mode():
        model(torch.ones(2, 3))
    assert frozen.state_sha256(model) == before
    model.running_mean.add_(1)
    assert frozen.state_sha256(model) != before


def test_positive_metrics_and_negative_pair_fpr_have_separate_denominators():
    rows = []
    for target, detected, score in [(1, True, .8), (1, False, 0), (0, True, 0), (0, False, 0)]:
        rows.append(dict(anchor_id='same-anchor', target=target, detected=detected,
            pair_iou=score, pair_dice=score, pair_precision=score, pair_recall=score,
            real_foreground_fraction=.1 if detected else 0, synthetic_foreground_fraction=.2 if detected else 0))
    result = frozen.summarize(rows).iloc[0]
    assert result['iou'] == .4  # Empty negative masks cannot improve this.
    assert result['positive_detection_rate'] == .5
    assert result['negative_pair_fpr'] == .5
    assert result['negative_false_positives'] == 1
    assert result['unique_anchors'] == 1


def test_masks_reject_out_of_bounds_coordinates():
    with pytest.raises(ValueError, match='outside'):
        frozen.column_mask([20, 10], [[-1, 5]])
    mask = frozen.column_mask([20, 10], [[2, 5], [11, 15]])
    assert mask[:, 2:5].all() and not mask[:, 5:11].any() and mask[:, 11:15].all()


def test_examples_embed_pngs_for_both_fonts_even_with_agg_backend(tmp_path, monkeypatch):
    import IPython.display
    import matplotlib.pyplot as plt
    outputs = []
    monkeypatch.setattr(IPython.display, 'display', outputs.append)
    monkeypatch.setattr(frozen, 'predict_images', lambda *args, **kwargs: None)
    monkeypatch.setattr(frozen, 'score_prediction', lambda prediction, case: dict(region_count=0, pair_iou=0))
    monkeypatch.setattr(frozen, 'plot_diagnostic', lambda *args: plt.subplots(figsize=(1, 1))[0])
    cases = [dict(sample_id=font, book='A', font=font, case='full_line',
                  real_image='real.png', synthetic_image='synthetic.png') for font in ('font_a', 'font_b')]
    frozen.run_cases(None, dict(cases=cases), tmp_path)
    assert len(outputs) == 2
    assert all(isinstance(value, IPython.display.Image) and value.data.startswith(b'\x89PNG') for value in outputs)

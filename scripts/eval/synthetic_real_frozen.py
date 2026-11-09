"""Frozen-checkpoint real/rendered diagnostics; references never enter prediction.

Arabic shaping uses the existing arabic_reshaper + python-bidi Bridge renderer.
This module deliberately does not import any training or corpus-building routine.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from dataset import prepare_image
from evaluation_utils import EvaluationSession, predict_cached_pair
from scripts.data.build_real_conditioned_synthetic_bridge import (
    clean_render_text, render_segmented_positive, safe_negative,
)
from scripts.data.build_real_conditioned_synthetic_bridge_v3 import font_supports_text
from scripts.eval.mask_quantitative import binary_metrics
from text_embedding import clean_letters
from word_alignment import normalize_word

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt'
SETTINGS = ROOT / 'docs/evaluation/word_alignment_20261109/settings.json'
CASES = ('full_line', 'separated_spans', 'reversed_spans', 'wide_gaps', 'unrelated')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model):
    """Include parameters AND buffers, detecting accidental BatchNorm adaptation."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def canonical(text):
    return normalize_word(''.join(clean_letters(text)))


def load_frozen(checkpoint=CHECKPOINT, dataset=None, settings=SETTINGS, device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    options = json.loads(Path(settings).read_text())
    session = EvaluationSession(checkpoint, dataset or ROOT / 'DataSet/ArabicData',
        device=device, split='test', settings=options, construct_negatives=False,
        pair_source='folders', crop_override='none')
    session.model.eval().requires_grad_(False)
    session.text_encoder.eval().requires_grad_(False)
    session.frozen_provenance = dict(checkpoint_sha256=sha256(checkpoint),
        model_state_sha256=state_sha256(session.model),
        text_encoder_sha256=state_sha256(session.text_encoder),
        settings_sha256=sha256(settings),
        vocabulary_sha256=sha256(ROOT / options['word_vocabulary_path']),
        settings=dict(session.settings), device=str(session.device), split='test',
        space_mode=session.config.space_mode, training_steps=0)
    return session


def verify_frozen(session):
    before = session.frozen_provenance
    checks = dict(checkpoint_unchanged=sha256(session.checkpoint) == before['checkpoint_sha256'],
        model_state_unchanged=state_sha256(session.model) == before['model_state_sha256'],
        text_encoder_unchanged=state_sha256(session.text_encoder) == before['text_encoder_sha256'],
        settings_unchanged=session.settings == before['settings'],
        vocabulary_unchanged=sha256(ROOT / before['settings']['word_vocabulary_path']) == before['vocabulary_sha256'],
        evaluation_mode=not session.model.training and not session.text_encoder.training,
        gradients_disabled=not any(p.requires_grad for module in (session.model, session.text_encoder) for p in module.parameters()))
    if not all(checks.values()):
        raise RuntimeError(f'Frozen evaluation contract violated: {checks}')
    return dict(**before, checks=checks)


def validate_annotation(record, dataset):
    """Reject stale, mismatched or out-of-bounds source annotations."""
    book, line = record['sample_id'].split(':', 1)
    path = Path(dataset) / book / 'BoundingBoxes' / f'{line}.json'
    annotation = json.loads(path.read_text())
    side = record['sides'][0]
    with Image.open(side['image']) as image:
        width, height = image.size
    if annotation['image_sha256'] != sha256(side['image']) or annotation['image_size'] != [width, height]:
        raise ValueError('Annotation image hash/size mismatch')
    groups = annotation['subwords']
    if not groups or canonical(''.join(g['text'] for g in groups)) != canonical(side['raw_text']):
        raise ValueError('Annotation/transcript character sequence mismatch')
    for group in groups:
        x0, y0, x1, y1 = group['line_bbox_xyxy']
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Source box outside line image')
        if group['status'] not in ('VERIFIED_SOURCE_CROP_PIXELS', 'SOURCE_LINE_COORDINATES'):
            raise ValueError('Unverified source coordinates')
    return dict(record=record, annotation=annotation, annotation_path=str(path), size=[width, height])


def envelope(groups):
    return [min(g['line_bbox_xyxy'][0] for g in groups), max(g['line_bbox_xyxy'][2] for g in groups)]


def span_blocks(anchor):
    """Choose two disjoint, separated annotated blocks before seeing predictions.

    Subword boxes can overlap (e.g. separately annotated dots); require that no
    unselected group intersects either selected column interval.
    """
    groups = anchor['annotation']['subwords']
    blocks = []
    reference = canonical(anchor['record']['sides'][0]['raw_text'])
    for start in range(len(groups)):
        for end in range(start + 1, min(len(groups), start + 6) + 1):
            text = ''.join(g['text'] for g in groups[start:end])
            value = canonical(text)
            if not 5 <= len(value) <= 14 or reference.count(value) != 1:
                continue
            interval = envelope(groups[start:end])
            if any(min(interval[1], g['line_bbox_xyxy'][2]) > max(interval[0], g['line_bbox_xyxy'][0])
                   for i, g in enumerate(groups) if not start <= i < end):
                continue
            blocks.append(dict(start=start, end=end, text=clean_render_text(text), interval=interval))
    candidates = [(a, b) for a in blocks for b in blocks
                  if a['end'] < b['start'] and b['interval'][1] + 8 < a['interval'][0]]
    if not candidates:
        raise ValueError('No two unambiguous separated source spans')
    return max(candidates, key=lambda pair: (min(len(canonical(b['text'])) for b in pair),
        sum(len(canonical(b['text'])) for b in pair)))


def unrelated_phrase(anchor, vocabulary, rng):
    """Conservative constructed negative, NOT human-reviewed semantic truth.

    Reject shared 3-grams and every known >=2-letter word/group that occurs in
    the anchor, including unspaced C. Single letters are allowed to overlap.
    """
    reference = anchor['record']['sides'][0]['raw_text']
    normalized = canonical(reference)
    known = set(vocabulary) | {g['text'] for g in anchor['annotation']['subwords']} | set(reference.split())
    forbidden = {canonical(w) for w in known if len(canonical(w)) >= 2 and canonical(w) in normalized}
    pool = sorted(w for w in vocabulary if 4 <= len(canonical(w)) <= 9)
    rng.shuffle(pool)
    selected = []
    for word in pool:
        candidate = ' '.join(selected + [word])
        if any(w in canonical(candidate) for w in forbidden):
            continue
        if not safe_negative(reference, candidate, negative_ngram=3, min_overlap_word_chars=2):
            continue
        selected.append(word)
        if len(selected) == 3:
            return ' '.join(selected)
    raise ValueError('Cannot construct a conservative unrelated phrase')


def column_mask(size, intervals):
    width, height = size
    mask = np.zeros((height, width), np.uint8)
    for lo, hi in intervals:
        lo, hi = int(lo), int(hi)
        if not 0 <= lo < hi <= width:
            raise ValueError('Mask interval outside source image')
        mask[:, lo:hi] = 255
    return mask


def generate_cases(session, output, *, anchors_per_book=2, seed=42, fonts=None):
    """References/boxes used ONLY here, for controlled stimulus and GT creation."""
    if anchors_per_book < 1:
        raise ValueError('anchors_per_book must be positive')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fonts = [Path(p) for p in (fonts or [ROOT / 'Fonts/Arslan_Wessam.ttf', ROOT / 'Fonts/Amiri-Regular.ttf'])]
    vocabulary = json.loads((ROOT / session.settings['word_vocabulary_path']).read_text())['words']
    rng = random.Random(seed)
    anchors, excluded = [], []
    for book in ('A', 'B', 'C'):
        candidates = sorted((r for r in session.view.records if r['book_id'] == book), key=lambda r: r['sample_id'])
        rng.shuffle(candidates)
        selected = 0
        for record in candidates:
            try:
                anchor = validate_annotation(record, session.dataset_path)
                anchor['blocks'] = span_blocks(anchor)
                anchor['unrelated'] = unrelated_phrase(anchor, vocabulary, rng)
                texts = [record['sides'][0]['raw_text'], anchor['unrelated']]
                texts += [g['text'] for g in anchor['annotation']['subwords']]
                texts += [b['text'] for b in anchor['blocks']]
                if not all(font_supports_text(font, t) for font in fonts for t in texts):
                    raise ValueError('Font missing shaped glyphs')
            except (ValueError, KeyError, OSError) as error:
                excluded.append(dict(sample_id=record['sample_id'], reason=str(error)))
                continue
            anchor['book'] = book
            anchors.append(anchor); selected += 1
            if selected == anchors_per_book:
                break
        if selected != anchors_per_book:
            raise ValueError(f'Only {selected} eligible Book {book} anchors')

    cases = []
    for index, anchor in enumerate(anchors):
        record = anchor['record']; groups = anchor['annotation']['subwords']
        for font in fonts:
            for kind in CASES:
                sample_id = f'{anchor["book"]}_{index:02d}_{font.stem}_{kind}'
                directory = output / 'samples' / sample_id
                directory.mkdir(parents=True, exist_ok=True)
                blocks = list(anchor['blocks'])
                if kind == 'reversed_spans':
                    blocks.reverse()
                if kind in ('separated_spans', 'reversed_spans'):
                    segments = [dict(kind='shared', text=blocks[0]['text']),
                                dict(kind='distractor', text=anchor['unrelated']),
                                dict(kind='shared', text=blocks[1]['text'])]
                    real_intervals = [b['interval'] for b in blocks]
                elif kind == 'full_line':
                    segments = [dict(kind='shared', text=record['sides'][0]['raw_text'])]
                    real_intervals = [envelope(groups)]
                elif kind == 'wide_gaps':
                    segments = [dict(kind='shared', text=g['text']) for g in groups]
                    real_intervals = [envelope(groups)]
                else:
                    segments = [dict(kind='distractor', text=anchor['unrelated'])]
                    real_intervals = []
                # Same seed for paired ordered/reversed layouts: reversal changes
                # logical text order without changing sampled gap/jitter policy.
                render_rng = random.Random(f'{seed}:{index}:{font.stem}')
                image_path, synthetic_mask = directory / 'synthetic.png', directory / 'synthetic_mask.png'
                rendered, size = render_segmented_positive(segments, font, image_path, synthetic_mask,
                    width=1024, height=128, font_size=72, padding=8,
                    gap_min_px=20 if kind == 'wide_gaps' else 8,
                    gap_max_px=20 if kind == 'wide_gaps' else 8, rng=render_rng)
                if kind in ('full_line', 'wide_gaps'):
                    # Full-line extent includes internal spaces, excluding margins.
                    synthetic_intervals = [[min(g['bbox_px'][0] for g in rendered), max(g['bbox_px'][2] for g in rendered)]]
                else:
                    synthetic_intervals = [[g['bbox_px'][0], g['bbox_px'][2]] for g in rendered if g['kind'] == 'shared']
                Image.fromarray(column_mask([1024, 128], synthetic_intervals)).save(synthetic_mask)
                real_mask = directory / 'real_mask.png'
                Image.fromarray(column_mask(anchor['size'], real_intervals)).save(real_mask)
                cases.append(dict(sample_id=sample_id, anchor_id=record['sample_id'], book=anchor['book'],
                    font=font.stem, case=kind, target=int(kind != 'unrelated'),
                    real_image=record['sides'][0]['image'], synthetic_image=str(image_path.resolve()),
                    real_mask=str(real_mask.resolve()), synthetic_mask=str(synthetic_mask.resolve()),
                    real_image_sha256=sha256(record['sides'][0]['image']), synthetic_image_sha256=sha256(image_path),
                    source_annotation=str(Path(anchor['annotation_path']).relative_to(ROOT)),
                    source_annotation_sha256=sha256(anchor['annotation_path']),
                    shared_blocks=blocks if kind in ('separated_spans', 'reversed_spans') else [],
                    rendered_segments=rendered, font_size=size, font_sha256=sha256(font),
                    negative_policy='no shared normalized 3-gram or known >=2-letter word/group',
                    real_intervals=real_intervals, synthetic_intervals=synthetic_intervals))
    manifest = dict(schema_version=1, seed=seed, cases=cases, excluded=excluded,
        anchor_ids=[a['record']['sample_id'] for a in anchors],
        selection='Seeded saved-test order; annotation, glyph and span eligibility only; no model predictions',
        limitations='Source subword blocks are annotated spans, not certified whole words. Wide gaps are between annotated groups. Negatives are constructed, not human reviewed.')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def synthetic_features(session, image):
    """Explicit external-image inference; never alter saved-test membership."""
    config = session.config
    _, tensor, geometry = prepare_image(image, (config.image_height, config.image_width),
        config.grayscale, config.crop, augment=False, binarize=config.binarize)
    with torch.inference_mode():
        output = session.model(tensor[None].to(session.device))
    valid = output['token_valid'][0]
    features = {key: output[key][0][valid].detach().float().cpu() for key in ('local', 'context', 'fused')}
    if any(not torch.isfinite(value).all() for value in features.values()):
        raise ValueError('Nonfinite synthetic features')
    return dict(features=features, geometry=geometry,
        physical=output['physical_window_indices'][0][valid].cpu().numpy())


def predict_images(session, real_image, synthetic_image, *, sample_id='external'):
    """Only image paths/features enter the matcher: no text, boxes, GT or labels."""
    sides = [dict(image=str(p), bbox=None) for p in (real_image, synthetic_image)]
    # The regular session's membership key includes its transcript FILE PATH.
    # Resolve that saved record for its image-only feature loader, then discard
    # all reference metadata before constructing the prediction pair.
    members = [entry['side'] for entry in session.allowed.values()
               if Path(entry['side']['image']).resolve() == Path(real_image).resolve()
               and not entry['side'].get('bbox')]
    if len(members) != 1:
        raise ValueError(f'Expected one saved-test real image record: {real_image}')
    real = session.get_line_features(members[0])
    synthetic = synthetic_features(session, sides[1]['image'])
    with torch.inference_mode():
        return predict_cached_pair(session, dict(sample_id=sample_id, sides=sides), [real, synthetic])


def score_prediction(prediction, case):
    """Score AFTER prediction; do not put source annotation in the predictor."""
    row = {k: case[k] for k in ('sample_id', 'anchor_id', 'book', 'font', 'case', 'target')}
    row['region_count'] = len(prediction['match']['regions'])
    row['detected'] = row['region_count'] > 0
    for side, name in enumerate(('real', 'synthetic')):
        with Image.open(case[f'{name}_mask']) as image:
            truth = np.asarray(image)
        row.update({f'{name}_{k}': v for k, v in binary_metrics(prediction['masks'][side], truth).items()})
    for metric in ('iou', 'dice', 'precision', 'recall'):
        row[f'pair_{metric}'] = (row[f'real_{metric}'] + row[f'synthetic_{metric}']) / 2
    return row


def summarize(rows, groupby=()):
    """Macro per-pair localization on positives; pair FPR on negatives only."""
    frame = pd.DataFrame(rows)
    groups = [((), frame)] if not groupby else frame.groupby(list(groupby), sort=True)
    output = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        pos, neg = group[group.target == 1], group[group.target == 0]
        result = dict(zip(groupby, key))
        result.update(pairs=len(group), unique_anchors=group.anchor_id.nunique(), positives=len(pos), negatives=len(neg))
        result.update({metric: float(pos[f'pair_{metric}'].mean()) if len(pos) else None
                       for metric in ('iou', 'dice', 'precision', 'recall')})
        result.update(positive_detection_rate=float(pos.detected.mean()) if len(pos) else None,
            negative_false_positives=int(neg.detected.sum()),
            negative_pair_fpr=float(neg.detected.mean()) if len(neg) else None,
            negative_real_coverage=float(neg.real_foreground_fraction.mean()) if len(neg) else None,
            negative_synthetic_coverage=float(neg.synthetic_foreground_fraction.mean()) if len(neg) else None)
        output.append(result)
    return pd.DataFrame(output)


def plot_diagnostic(prediction, case, metrics):
    """Use explicit real/synthetic labels, independent of positional A/B names."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 1, 1, 2]})
    for side, name in enumerate(('real', 'synthetic')):
        image = np.asarray(Image.open(case[f'{name}_image']).convert('RGB'))
        truth = np.asarray(Image.open(case[f'{name}_mask'])) > 0
        mask = np.asarray(prediction['masks'][side]) > 0
        overlay = image.copy()
        overlay[mask] = (.5 * overlay[mask] + .5 * np.array([255, 40, 30])).astype(np.uint8)
        for index, (pixels, title) in enumerate(((image, name), (mask, 'predicted columns'),
                                               (overlay, 'prediction overlay'), (truth, 'reference columns'))):
            axes[index, side].imshow(pixels, cmap='gray', vmin=0 if pixels.ndim == 2 else None,
                vmax=1 if pixels.ndim == 2 else None, aspect='auto')
            axes[index, side].set_title(f'{"Real Book " + case["book"] if side == 0 else "Synthetic"}: {title}')
            axes[index, side].set_xticks([]); axes[index, side].set_yticks([])
    axes[4, 0].imshow(prediction['cosine'], aspect='auto', origin='upper', cmap='viridis', vmin=-1, vmax=1)
    axes[4, 0].set(title='Frozen image embedding cosine', xlabel='Synthetic RTL window', ylabel='Real RTL window')
    axes[4, 1].axis('off')
    axes[4, 1].text(.02, .95, '\n'.join([f'Regions: {metrics["region_count"]}',
        *[f'{name}: IoU {metrics[name + "_iou"]:.1%}, precision {metrics[name + "_precision"]:.1%}, recall {metrics[name + "_recall"]:.1%}' for name in ('real', 'synthetic')],
        'Constructed negative' if not case['target'] else 'Reference = annotated source spans / rendered spans',
        'Matching inputs: two images only', 'Frozen weights, vocabulary and thresholds']), va='top', fontsize=11)
    fig.suptitle(f'Real Book {case["book"]} / synthetic | {case["case"]} | {case["font"]}\n{case["anchor_id"]}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    return fig


def run_cases(session, manifest, output, *, display_examples=True):
    from IPython.display import Image as NotebookImage, display
    import matplotlib.pyplot as plt
    output = Path(output); rows = []; displayed = set()
    for index, case in enumerate(manifest['cases']):
        # Manifest annotations never enter this call.
        prediction = predict_images(session, case['real_image'], case['synthetic_image'], sample_id=case['sample_id'])
        metrics = score_prediction(prediction, case); rows.append(metrics)
        print(f'{index + 1}/{len(manifest["cases"])} {case["sample_id"]}: regions={metrics["region_count"]}, IoU={metrics["pair_iou"]:.3f}', flush=True)
        example_key = (case['book'], case['font'], case['case'])
        if display_examples and example_key not in displayed:
            displayed.add(example_key)
            figure = plot_diagnostic(prediction, case, metrics)
            image_path = output / f'{case["sample_id"]}.png'
            figure.savefig(image_path, dpi=100)
            plt.close(figure)
            # Explicit PNG MIME works even when the kernel uses the Agg backend.
            display(NotebookImage(filename=str(image_path)))
    pd.DataFrame(rows).to_csv(output / 'per_pair.csv', index=False)
    return rows


def save_report(session, manifest, rows, output):
    output = Path(output)
    provenance = verify_frozen(session)
    provenance.update(unique_real_anchors=len(manifest['anchor_ids']), synthetic_images=len(rows),
        matcher_inputs='Only real and synthetic image paths/features; no reference text/GT/boxes',
        checkpoint_training_books=sorted({r['book_id'] for r in session.views['train'].records}),
        limitation='Diagnostic font transfer on reused anchors, not independent real-real test improvement or a calibrated population FPR')
    (output / 'verification.json').write_text(json.dumps(provenance, indent=2))
    for name, grouping in [('overall', ()), ('by_book', ('book',)), ('by_case', ('case',)),
                           ('by_font', ('font',)), ('by_book_case', ('book', 'case'))]:
        summarize(rows, grouping).to_csv(output / f'{name}.csv', index=False)
    return provenance

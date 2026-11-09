"""Select and render a reproducible, frozen-checkpoint A/B/C qualitative review.

Masks establish the sampling strata and score predictions, never inference inputs.
Run --prepare-only after downloading pairs.jsonl, then run without that flag once
all images and masks have been synced. Original manifests remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GROUPS = ('A_B', 'B_C', 'A_C')


def training_book_counts(split_ids):
    """Book exposure and individual line membership are different facts."""
    from collections import Counter
    return dict(Counter(sid.split(':', 1)[0].removeprefix('Book')
                        for sid in split_ids['train']))


def membership_caption(book, line_split, book_counts):
    line = ('outside saved subset' if line_split == 'unseen' else line_split)
    count = book_counts.get(book, 0)
    exposure = f'{count:,} book training lines' if count else 'book training not recorded in this split'
    return f'{book}: line={line}; {exposure}'


def merged_intervals(intervals):
    result = []
    for left, right in sorted(intervals):
        if left >= right:
            raise ValueError('Empty or reversed mask interval')
        if result and left <= result[-1][1]:
            result[-1][1] = max(right, result[-1][1])
        else:
            result.append([left, right])
    return result


def topology(row, books):
    counts = [len(merged_intervals(row[book]['mask_regions_x'])) for book in books]
    # Require separated foreground in BOTH lines for an unambiguous stratum.
    return ('consecutive' if counts == [1, 1] else
            'non_consecutive' if min(counts) >= 2 else 'asymmetric_or_empty')


def prepare(dataset, split_manifest, output, seed=42, per_type=5):
    membership = {sid: name for name, ids in json.loads(split_manifest.read_text()).items() for sid in ids}
    saved_books = {sid.split(':', 1)[0] for sid in membership}
    chosen, inventory = [], {}
    for group in GROUPS:
        books = group.split('_')
        source = dataset / 'alignedPairs' / group / 'pairs.jsonl'
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len({row['pair_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate pair IDs in ' + group)
        pools = {name: [] for name in ('consecutive', 'non_consecutive', 'asymmetric_or_empty')}
        for row in rows:
            splits = {book: membership.get(f"Book{book}:" + row[book]['line_id'], 'unseen') for book in books}
            row = dict(row, book_pair=group, path_type=topology(row, books), saved_line_splits=splits)
            pools[row['path_type']].append(row)
        inventory[group] = dict(total=len(rows), types={name: len(pool) for name, pool in pools.items()})
        for name in ('consecutive', 'non_consecutive'):
            pool = pools[name]
            # Test members for every book represented in the saved split.
            eligible = [row for row in pool if all(
                row['saved_line_splits'][book] == ('test' if f'Book{book}' in saved_books else 'unseen') for book in books)]
            if len(eligible) < per_type:
                raise ValueError(f'{group}/{name}: need {per_type} held-out eligible pairs; found {len(eligible)}')
            rng = random.Random(f'{seed}:{group}:{name}')
            order = sorted(eligible, key=lambda row: row['pair_id'])
            rng.shuffle(order)
            selected, used_lines = [], set()
            for row in order:
                identities = {(book, row[book]['line_id']) for book in books}
                if identities & used_lines:
                    continue
                selected.append(row)
                used_lines.update(identities)
                if len(selected) == per_type:
                    break
            if len(selected) < per_type:
                raise ValueError(f'{group}/{name}: insufficient pairs with distinct lines')
            for index, row in enumerate(selected, 1):
                chosen.append(dict(row, review_id=f'{group}_{name}_{index:02d}'))
            inventory[group]['types'][name] = dict(total=len(pool), eligible=len(eligible), selected=len(selected))
    output.mkdir(parents=True, exist_ok=True)
    selection = dict(seed=seed, samples_per_type=per_type, inventory=inventory, samples=chosen,
        policy='Books present in the checkpoint split manifest use saved test lines; other books are absent from that manifest. '
               'Absence does not establish exclusion from earlier training. '
               'Consecutive = one connected foreground x interval in each supplied mask; '
               'non-consecutive = at least two separated intervals in each. '
               'Selection is independent of model predictions. Distinct lines within each stratum.',
        split_manifest_sha256=hashlib.sha256(split_manifest.read_bytes()).hexdigest())
    (output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    return selection


def reuse_selection(source, split_manifest, output):
    """Keep review identities while recording the new checkpoint's membership."""
    selection = json.loads(source.read_text())
    selection.setdefault('original_split_manifest_sha256', selection['split_manifest_sha256'])
    membership = {sid: name for name, ids in json.loads(split_manifest.read_text()).items() for sid in ids}
    for row in selection['samples']:
        row.setdefault('original_saved_line_splits', dict(row['saved_line_splits']))
        for key in ('selection_iou', 'selection_white_fraction', 'selection_foreground'):
            if key in row:
                row['original_' + key] = row.pop(key)
        row['saved_line_splits'] = {
            book: membership.get(f"Book{book}:" + row[book]['line_id'], 'unseen')
            for book in row['book_pair'].split('_')}
    selection.setdefault('original_policy', selection.get('policy', ''))
    selection['policy'] = (
        'Same fixed examples as the original review, selected using the previous checkpoint. '
        'They have not been selected or ranked using the current checkpoint. '
        'Current checkpoint split membership is recorded per sample, including training and validation lines. '
        'An individual line absent from the balanced split does not mean its book was absent from training. '
        'This mixed-split qualitative comparison is not a held-out performance estimate.')
    if selection.get('mode') == 'illustrative_examples':
        selection['scope'] = 'fixed_previous_checkpoint_examples'
        selection['ranking_method'] = 'fixed previous-checkpoint examples; no current-checkpoint ranking'
        selection['original_constraints'] = selection.pop('constraints', {})
    selection['split_manifest_sha256'] = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    # Eligibility counts describe the original selection, not this checkpoint.
    for group in selection['inventory'].values():
        for counts in group.get('types', {}).values():
            if isinstance(counts, dict) and 'eligible' in counts:
                counts['original_eligible'] = counts.pop('eligible')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    return selection


def load_settings(notebook):
    """Read the user's current matching settings without executing notebook setup."""
    cells = json.loads(notebook.read_text())['cells']
    source = next(''.join(c['source']) for c in cells if ''.join(c['source']).startswith('# EVALUATION-ONLY MATCHING SETTINGS.'))
    # Execute the settings cell and only its adjacent settings function; omit
    # the notebook's module reloads, session mutations and display actions.
    import ast
    namespace = {}
    exec(compile(ast.parse(source), str(notebook), 'exec'), namespace)
    function_source = next(''.join(c['source']) for c in cells if ''.join(c['source']).startswith('def current_match_settings('))
    function = ast.parse(function_source).body[0]
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), 'exec'), namespace)
    return namespace['current_match_settings']()


def load_evaluation_setup(notebook):
    """Read inference options without running imports, reloads or notebook UI."""
    import ast
    cells = json.loads(notebook.read_text())['cells']
    source = next(''.join(c['source']) for c in cells
                  if ''.join(c['source']).startswith('# MODEL/DATA SETUP.'))
    names = {'EVALUATION_CROP', 'DEVICE', 'REPRESENTATION', 'RANDOM_SEED'}
    return {target.id: ast.literal_eval(node.value)
            for node in ast.parse(source).body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name) and target.id in names}


def render_contact_sheets(dataset, output, selection, report):
    """Overview of selected pairs, with supplied/predicted foreground bands."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    details = {row['review_id']: row for row in report['samples']}
    for group in GROUPS:
        rows = [row for row in selection['samples'] if row['book_pair'] == group]
        if not rows:
            continue
        fig, axes = plt.subplots(len(rows), 2, figsize=(18, 2.1 * len(rows)), layout='constrained')
        fig.suptitle(f'{group.replace("_", "–")} | green band above = supplied mask; blue band below = prediction', fontsize=14)
        for row, axrow in zip(rows, axes):
            detail = details[row['review_id']]
            for index, (book, ax) in enumerate(zip(group.split('_'), axrow)):
                with Image.open(dataset / row[book]['image_path']) as image:
                    ax.imshow(image.convert('RGB'), aspect='auto')
                    width, height = image.size
                for y, intervals, color in [(-.15 * height, detail['supplied_intervals'][book], '#16a34a'),
                                             (height, detail['predicted_intervals'][book], '#2563eb')]:
                    for left, right in intervals:
                        ax.add_patch(Rectangle((left, y), right-left, .12*height, color=color))
                ax.set(xlim=(0, width), ylim=(1.15*height, -.18*height), xticks=[], yticks=[])
                metric = detail['metrics'][('a' if index == 0 else 'b') + '_iou']
                ax.set_title(f'{row["review_id"]} | Book {book} | IoU {metric:.3f}', fontsize=9)
        fig.savefig(output / f'contact_{group}.png', dpi=100)
        plt.close(fig)


def evaluate(args, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch
    from dataset import file_hash
    from train import load_checkpoint
    from evaluation_utils import (EvaluationSession, _side_key, compute_pair_metrics,
                                  plot_pair_alignment, load_inference_text_encoder)

    torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto'
                          else 'cpu' if args.device.startswith('cuda') and not torch.cuda.is_available()
                          else args.device)
    model, _, config, saved = load_checkpoint(args.checkpoint, device, load_text=False)
    book_counts = training_book_counts(saved['split_ids'])
    actual_membership = {sid: name for name, ids in saved['split_ids'].items() for sid in ids}
    for row in selection['samples']:
        for book, expected in row['saved_line_splits'].items():
            if actual_membership.get(f"Book{book}:" + row[book]['line_id'], 'unseen') != expected:
                raise ValueError('Checkpoint membership differs from selected split manifest')
    config = replace(config, crop=args.crop)
    model.eval()
    # This is an explicitly selected qualitative catalog, not a replacement for
    # EvaluationSession's saved-split catalog. It reuses the exact inference flow.
    session = object.__new__(EvaluationSession)
    session.model, session.config, session.saved, session.device = model, config, saved, device
    session.checkpoint = args.checkpoint.resolve()
    session.checkpoint_sha256 = file_hash(args.checkpoint)
    session.split = 'fixed_qualitative_mixed_splits' if args.selection else 'saved_test_and_external_books'
    session.settings = (json.loads(args.settings_report.read_text())['settings']
                        if args.settings_report else load_settings(args.settings_notebook))
    settings_path = (args.settings_report or args.settings_notebook).resolve()
    settings_source = dict(kind='report' if args.settings_report else 'notebook',
                           path=str(settings_path), sha256=file_hash(settings_path))
    session._text_encoder = load_inference_text_encoder(saved, config, device)
    session.feature_cache, session.metric_cache, session.line_aliases, session.allowed = {}, {}, {}, {}
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
        config=vars(config), settings=session.settings, selection_policy=selection['policy'],
        split=session.split, device=str(device), representation=args.representation,
        settings_source=settings_source, book_training_counts=book_counts, samples=[])
    showcase = selection.get('mode') == 'illustrative_examples'
    if showcase:
        report['selection_inventory'] = selection['inventory']
        report['ranking_method'] = selection.get('ranking_method', 'mean of the two source-mask IoUs, then weaker-side IoU')
        report['selection_scope'] = selection.get('scope', 'scored_subset')
        report['selection_constraints'] = selection.get('constraints', {})
    sample_description = (f"{len(selection['samples'])} selected examples: {selection['samples_per_pair']} per book pair.\n"
                          if showcase else '30 fixed samples: 5 consecutive and 5 non-consecutive per book pair.\n')
    cells = [dict(cell_type='markdown', metadata={}, source=[
        '# A–B, B–C and A–C qualitative review\n',
        sample_description,
        f'Checkpoint: `{args.checkpoint.resolve().relative_to(ROOT)}`.\n',
        'Saved training-line counts by book: '+', '.join(f'{b}={n:,}' for b,n in sorted(book_counts.items()))+'. '
        'Line membership below refers to the individual image, not whether its book was trained.\n',
        f'Matching parameters from `{settings_path.relative_to(ROOT)}`; '
        f'representation=`{args.representation}`, crop=`{args.crop}`, device=`{device}`.\n',
        selection['policy'] + '\n',
        ('Examples are selected for agreement with the supplied masks on both lines.\n' if showcase else
         'Consecutive means one connected supplied-mask interval per line; non-consecutive means at least two per line. '
         'Model paths may differ. Masks are used only for selection and scoring.\n'),
        'Saved outputs come from `scripts/eval/review_book_pairs.py`; no training or parameter search.\n'])]
    cards = []
    for row in selection['samples']:
        books = row['book_pair'].split('_')
        sides = [dict(image=str((args.dataset / row[b]['image_path']).resolve()),
                      text=None, raw_text=row[b].get('text', ''), bbox=None) for b in books]
        annotations = [dict(mask=str((args.dataset / row[b]['mask_path']).resolve())) for b in books]
        pair = dict(sample_id=row['book_pair'] + ':' + row['pair_id'], sides=sides, annotations=annotations,
                    target=1, label=row['path_type'], constructed_negative=False, group_ids=['', ''], anchor_id='',
                    annotation_provenance='supplied Drive pair masks')
        for side in sides:
            session.allowed[_side_key(side)] = dict(side=side)
        result = session.predict_pair(pair, args.representation)
        # Read masks only after prediction. Missing masks are a sync failure here.
        truth = []
        for index, annotation in enumerate(annotations):
            with Image.open(annotation['mask']) as image:
                mask = np.array(image.convert('L'))
            with Image.open(sides[index]['image']) as image:
                if mask.shape != (image.height, image.width):
                    raise ValueError('Mask/image size mismatch')
            if not set(np.unique(mask)).issubset({0, 255}):
                raise ValueError('Non-binary supplied mask')
            actual = np.flatnonzero(np.diff(np.r_[False, (mask > 0).any(axis=0), False])).reshape(-1, 2).tolist()
            if actual != merged_intervals(row[books[index]]['mask_regions_x']):
                raise ValueError('Manifest intervals disagree with supplied mask')
            truth.append(mask > 0)
        metrics, gt = compute_pair_metrics(result, config, preloaded_ground_truth=truth)
        metrics['ground_truth_provenance'] = pair['annotation_provenance']
        result.update(metrics=metrics, ground_truth=gt)
        split_labels = ' | '.join(membership_caption(book, row['saved_line_splits'][book], book_counts) for book in books)
        title = f"{row['review_id']} | physical books {books[0]}/{books[1]} (plot rows A/B)\n{split_labels}"
        figure = plot_pair_alignment(session, result, show_gt=True, show_window_indices=True)
        figure.suptitle(title)
        figure.savefig(args.output / (row['review_id'] + '.png'), dpi=90)
        plt.close(figure)
        candidates = result['match']['candidates']
        (args.output / (row['review_id'] + '_candidates.json')).write_text(json.dumps(candidates, indent=2, allow_nan=False))
        np.savez_compressed(args.output / (row['review_id'] + '_matrices.npz'),
                            cosine=result['cosine'], rewards=result['match']['rewards'])
        predicted_intervals = []
        for index, mask in enumerate(result['masks']):
            Image.fromarray(mask).save(args.output / (row['review_id'] + f'_pred_{books[index]}.png'))
            predicted_intervals.append(np.flatnonzero(np.diff(np.r_[False, (mask > 0).any(axis=0), False])).reshape(-1, 2).tolist())
        detail = dict(review_id=row['review_id'], book_pair=row['book_pair'], path_type=row['path_type'],
            pair_id=row['pair_id'], saved_line_splits=row['saved_line_splits'], metrics=metrics,
            supplied_intervals={b: row[b]['mask_regions_x'] for b in books},
            predicted_intervals=dict(zip(books, predicted_intervals)),
            accepted_regions=len(result['match']['regions']), annotation_origin=row.get('sample_origin'),
            image_sha256={b: file_hash(sides[i]['image']) for i, b in enumerate(books)},
            mask_sha256={b: file_hash(annotations[i]['mask']) for i, b in enumerate(books)})
        detail['predicted_white_fraction'] = {book: float((np.asarray(mask) > 0).mean())
                                               for book, mask in zip(books, result['masks'])}
        detail['foreground_quality'] = {}
        for prefix, book, mask, gt_mask in zip(('a', 'b'), books, result['masks'], truth):
            pred = np.asarray(mask) > 0
            tp = int(np.logical_and(pred, gt_mask).sum())
            p, t = int(pred.sum()), int(gt_mask.sum())
            quality = dict(white_f2=5*tp/(4*t+p) if 4*t+p else 0.,
                black_specificity=float(np.logical_and(~pred, ~gt_mask).sum()/np.count_nonzero(~gt_mask))
                    if (~gt_mask).any() else 1., supplied_white_fraction=float(gt_mask.mean()))
            if 'selection_foreground' in row:
                expected = row['selection_foreground'][prefix]
                assert abs(quality['white_f2'] - expected['f2']) < 1e-8
                assert abs(quality['black_specificity'] - expected['specificity']) < 1e-8
                assert abs(metrics[prefix+'_iou'] - row['selection_iou']['first' if prefix == 'a' else 'second']) < 1e-8
            detail['foreground_quality'][book] = quality
        report['samples'].append(detail)
        text = f"{title}\nIoU: {books[0]}={metrics['a_iou']:.3f}, {books[1]}={metrics['b_iou']:.3f}; accepted routes={detail['accepted_regions']}"
        if showcase:
            text += (f"\nMean mask IoU={(metrics['a_iou'] + metrics['b_iou']) / 2:.3f}"
                     f"\nWhite-mask precision: {books[0]}={metrics['a_precision']:.3f}, {books[1]}={metrics['b_precision']:.3f}"
                     f"; recall: {books[0]}={metrics['a_recall']:.3f}, {books[1]}={metrics['b_recall']:.3f}")
            text += '\nPredicted white coverage: ' + ', '.join(
                f"{book}={detail['predicted_white_fraction'][book]:.1%}" for book in books)
            text += '\nBlack-background specificity: ' + ', '.join(
                f"{book}={detail['foreground_quality'][book]['black_specificity']:.3f}" for book in books)
            text += '\nWhite F2 (recall priority): ' + ', '.join(
                f"{book}={detail['foreground_quality'][book]['white_f2']:.3f}" for book in books)
        print(text, flush=True)
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p><a href="{row["review_id"]}.png"><img loading="lazy" src="{row["review_id"]}.png"></a></article>')
        code = f"from IPython.display import Image, display\nprint({text!r})\ndisplay(Image(filename=str(REVIEW / {row['review_id'] + '.png'!r})))\n"
        import base64
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, source=code.splitlines(keepends=True), outputs=[
            dict(output_type='stream', name='stdout', text=[text + '\n']),
            dict(output_type='display_data', metadata={}, data={'image/png':base64.b64encode((args.output/(row['review_id']+'.png')).read_bytes()).decode(), 'text/plain':[title]})]))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n')
    report['summary'] = {group: {kind: dict(samples=len(rows),
        mean_iou=float(np.mean([(r['metrics']['a_iou'] + r['metrics']['b_iou']) / 2 for r in rows])),
        detected_pairs=sum(r['accepted_regions'] > 0 for r in rows))
        for kind in ('consecutive', 'non_consecutive')
        if (rows := [r for r in report['samples'] if r['book_pair'] == group and r['path_type'] == kind])}
        for group in GROUPS}
    if showcase:
        report['summary'] = {group: dict(samples=len(rows),
            mean_iou=float(np.mean([(r['metrics']['a_iou'] + r['metrics']['b_iou']) / 2 for r in rows])),
            detected_pairs=sum(r['accepted_regions'] > 0 for r in rows))
            for group in GROUPS if (rows := [r for r in report['samples'] if r['book_pair'] == group])}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n')
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>A/B/C qualitative review</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}article{margin:40px 0;border-top:1px solid #bbb}</style><h1>' + str(len(selection['samples'])) + '-sample A/B/C qualitative review</h1><p>' + html.escape(selection['policy']) + '</p><p>Qualitative review; automatic annotation candidates include unreviewed labels. No training or threshold tuning. Physical books are named above each figure; internal plot rows A/B refer to first/second input.</p>' + ''.join(cards))
    settings_text = 'Matching parameters from: ' + str(settings_path.relative_to(ROOT)) + '\n' + json.dumps(session.settings, indent=2) + '\n'
    setup = ("from pathlib import Path\nimport json\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'evaluation_utils.py').is_file())\nREVIEW = ROOT / "
             + repr(str(args.output.resolve().relative_to(ROOT))) + '\n'
             + "REPORT = json.loads((REVIEW / 'report.json').read_text())\n"
             + "print('Matching parameters from:', Path(REPORT['settings_source']['path']).relative_to(ROOT))\n"
             + "print(json.dumps(REPORT['settings'], indent=2))\n")
    cells.insert(1, dict(cell_type='code', metadata={}, execution_count=None, source=setup.splitlines(keepends=True),
                        outputs=[dict(output_type='stream', name='stdout', text=settings_text.splitlines(keepends=True))]))
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      qualitative_review=dict(checkpoint=str(session.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
                                              settings=session.settings, settings_source=settings_source,
                                              execution='imported real inference artifacts')))
    for index, cell in enumerate(cells):
        cell['id'] = f'book-review-{index:03d}'
    args.notebook.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    render_contact_sheets(args.dataset, args.output, selection, report)
    if showcase:
        add_white_mask_comparisons(args.dataset, args.output, selection, report, args.notebook)


def add_white_mask_comparisons(dataset, output, selection, report, notebook_path):
    """Show supplied/predicted white directly, preserving each source geometry."""
    import base64
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    notebook = json.loads(notebook_path.read_text())
    rows_by_id = {row['review_id']: row for row in selection['samples']}
    for group in GROUPS:
        rows = [row for row in report['samples'] if row['book_pair'] == group]
        if not rows:
            continue
        books = group.split('_')
        fig, axes = plt.subplots(len(rows), 2, figsize=(14, 2*len(rows)), squeeze=False)
        for rank, row in enumerate(rows):
            original = rows_by_id[row['review_id']]
            for col, (prefix, book) in enumerate(zip(('a', 'b'), books)):
                with Image.open(dataset / original[book]['mask_path']) as im:
                    truth = np.asarray(im.convert('L'))
                with Image.open(output / (row['review_id'] + f'_pred_{book}.png')) as im:
                    pred = np.asarray(im.convert('L'))
                assert pred.shape == truth.shape
                gap = max(2, truth.shape[0]//10)
                combined = np.concatenate((truth, np.full((gap, truth.shape[1]), 70, dtype='uint8'), pred))
                ax = axes[rank, col]
                ax.imshow(combined, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto')
                ax.set_yticks([truth.shape[0]/2, truth.shape[0]*1.5+gap], ['Supplied', 'Predicted'])
                ax.set_xticks([])
                m = row['metrics']
                ax.set_title(f"{rank+1}. Book {book} | white IoU {m[prefix+'_iou']:.3f} | white recall {m[prefix+'_recall']:.1%}", fontsize=10)
        fig.suptitle(group.replace('_', '–') + ' | White-mask recovery\nEach column uses the same source coordinates for supplied and predicted masks.', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94))
        filename = f'white_masks_{group}.png'
        fig.savefig(output / filename, dpi=110)
        plt.close(fig)
        # Keep one code cell per sample so both notebooks share the same figures.
        index = 2 + next(i for i, row in enumerate(report['samples']) if row['book_pair'] == group)
        cell = notebook['cells'][index]
        code = ''.join(cell['source'])
        display_line = f"display(Image(filename=str(REVIEW / {filename!r})))\n"
        if display_line not in code:
            start = code.index('display(Image(')
            cell['source'] = (code[:start]+display_line+code[start:]).splitlines(keepends=True)
            cell['outputs'].insert(1, dict(output_type='display_data', metadata={}, data={
                'image/png': base64.b64encode((output / filename).read_bytes()).decode(),
                'text/plain': [f'{group}: supplied and predicted white masks']}))
        else:
            cell['outputs'][1]['data']['image/png'] = base64.b64encode((output / filename).read_bytes()).decode()
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'DataSet/ArabicData')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'Results/Evaluation/book_pairs_qualitative')
    parser.add_argument('--notebook', type=Path, default=ROOT / 'notebooks/book_pairs_qualitative.ipynb')
    parser.add_argument('--settings-notebook', type=Path, default=ROOT / 'notebooks/model_evaluation.ipynb')
    parser.add_argument('--settings-report', type=Path, help='Reuse matching settings from a previous report')
    parser.add_argument('--selection', type=Path, help='Reuse fixed examples and refresh their checkpoint split labels')
    parser.add_argument('--device', help='Default: DEVICE in the settings notebook')
    parser.add_argument('--crop', choices=('content', 'none', 'auto', 'xml'), help='Default: EVALUATION_CROP in the settings notebook')
    parser.add_argument('--representation', choices=('local', 'context', 'fused'), help='Default: REPRESENTATION in the settings notebook')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, help='Default: RANDOM_SEED in the settings notebook')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    setup = load_evaluation_setup(args.settings_notebook)
    for option, name in (('device', 'DEVICE'), ('crop', 'EVALUATION_CROP'),
                         ('representation', 'REPRESENTATION'), ('seed', 'RANDOM_SEED')):
        if getattr(args, option) is None:
            setattr(args, option, setup[name])
    split_manifest = args.checkpoint.parent / 'split_manifest.json'
    selection = (reuse_selection(args.selection, split_manifest, args.output) if args.selection else
                 prepare(args.dataset, split_manifest, args.output, args.seed))
    print(json.dumps(selection['inventory'], indent=2), flush=True)
    if not args.prepare_only:
        evaluate(args, selection)


if __name__ == '__main__':
    main()

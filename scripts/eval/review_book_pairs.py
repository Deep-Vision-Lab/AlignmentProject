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
            # Test members for A/B, external Book C. No fallback or regenerated split.
            eligible = [row for row in pool if all(
                row['saved_line_splits'][book] == ('unseen' if book == 'C' else 'test') for book in books)]
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
        policy='A/B lines must belong to checkpoint saved test membership; C lines are external/unseen. '
               'Consecutive = one connected foreground x interval in each supplied mask; '
               'non-consecutive = at least two separated intervals in each. '
               'Selection is independent of model predictions. Distinct lines within each stratum.',
        split_manifest_sha256=hashlib.sha256(split_manifest.read_bytes()).hexdigest())
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


def render_contact_sheets(dataset, output, selection, report):
    """Overview of all 30 pairs, with supplied/predicted foreground bands."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    details = {row['review_id']: row for row in report['samples']}
    for group in GROUPS:
        rows = [row for row in selection['samples'] if row['book_pair'] == group]
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
    device = torch.device(args.device)
    model, _, config, saved = load_checkpoint(args.checkpoint, device, load_text=False)
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
    session.split = 'saved_test_AB_and_external_C'
    session.settings = load_settings(args.settings_notebook)
    session._text_encoder = load_inference_text_encoder(saved, config, device)
    session.feature_cache, session.metric_cache, session.line_aliases, session.allowed = {}, {}, {}, {}
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
        config=vars(config), settings=session.settings, selection_policy=selection['policy'],
        split=session.split, device=str(device), samples=[])
    cells = [dict(cell_type='markdown', metadata={}, source=[
        '# A–B, B–C and A–C qualitative review\n',
        '30 fixed samples: 5 consecutive and 5 non-consecutive per book pair.\n',
        'A/B sides use saved test line membership; Book C is external. '
        'This stratified review is not a population performance estimate.\n',
        'Consecutive means one connected supplied-mask interval per line; non-consecutive means at least two per line. '
        'Model paths may differ. Masks are used only for selection and scoring.\n',
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
        title = f"{row['review_id']} | physical books {books[0]}/{books[1]} (plot rows A/B)"
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
        report['samples'].append(detail)
        text = f"{title}\nIoU: {books[0]}={metrics['a_iou']:.3f}, {books[1]}={metrics['b_iou']:.3f}; accepted routes={detail['accepted_regions']}"
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
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n')
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>A/B/C qualitative review</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}article{margin:40px 0;border-top:1px solid #bbb}</style><h1>30-sample A/B/C qualitative review</h1><p>' + html.escape(selection['policy']) + '</p><p>Stratified qualitative review; automatic annotation candidates include unreviewed labels. No training or threshold tuning. Physical books are named above each figure; internal plot rows A/B refer to first/second input.</p>' + ''.join(cards))
    setup = "from pathlib import Path\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'evaluation_utils.py').is_file())\nREVIEW = ROOT / " + repr(str(args.output.resolve().relative_to(ROOT))) + '\n'
    cells.insert(1, dict(cell_type='code', metadata={}, execution_count=None, source=setup.splitlines(keepends=True), outputs=[]))
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      qualitative_review=dict(checkpoint_sha256=session.checkpoint_sha256, execution='imported real inference artifacts')))
    for index, cell in enumerate(cells):
        cell['id'] = f'book-review-{index:03d}'
    args.notebook.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    render_contact_sheets(args.dataset, args.output, selection, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'DataSet/ArabicData')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'Results/Evaluation/book_pairs_qualitative')
    parser.add_argument('--notebook', type=Path, default=ROOT / 'notebooks/book_pairs_qualitative.ipynb')
    parser.add_argument('--settings-notebook', type=Path, default=ROOT / 'notebooks/model_evaluation.ipynb')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--crop', default='content', choices=('content', 'none', 'auto', 'xml'))
    parser.add_argument('--representation', default='fused', choices=('local', 'context', 'fused'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    selection = prepare(args.dataset, args.checkpoint.parent / 'split_manifest.json', args.output, args.seed)
    print(json.dumps(selection['inventory'], indent=2), flush=True)
    if not args.prepare_only:
        evaluate(args, selection)


if __name__ == '__main__':
    main()

"""Select five illustrative examples per book pair with the frozen evaluator."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.review_book_pairs import evaluate, load_evaluation_setup, load_settings

GROUPS = ('A_B', 'A_C', 'B_C')


def choose_best(scored, count=5):
    """Rank by mean mask IoU, then the weaker side; deduplicate pair identities."""
    ranked = sorted(scored, key=lambda row: (
        -(row['a_iou'] + row['b_iou']) / 2, -min(row['a_iou'], row['b_iou']), row['row']['pair_id']))
    chosen, used = [], set()
    for item in ranked:
        row = item['row']
        identity = (row['book_pair'], row['pair_id'])
        if identity in used:
            continue
        chosen.append(item)
        used.add(identity)
        if len(chosen) == count:
            break
    return chosen


def predict_candidate(session, dataset, row, representation):
    from evaluation_utils import _side_key
    sides = [dict(image=str(dataset / row[b]['image_path']), text=None,
                  raw_text=row[b].get('text', ''), bbox=None) for b in row['book_pair'].split('_')]
    pair = dict(sample_id=row['book_pair'] + ':' + row['pair_id'], sides=sides, target=1,
                label='illustrative_candidate', constructed_negative=False, group_ids=['', ''], anchor_id='',
                annotation_provenance='supplied pair masks')
    for side in sides:
        session.allowed[_side_key(side)] = dict(side=side)
    return session.predict_pair(pair, representation)


def foreground_rank(item):
    """Recover white on both lines first; then prefer IoU and clean black."""
    return (-min(item['a_f2'], item['b_f2']),
            -(item['a_iou'] + item['b_iou']) / 2,
            -min(item['a_specificity'], item['b_specificity']), item['row']['pair_id'])


def choose_foreground(scored, session, dataset, representation, group, low=.10, high=.45,
                      min_precision=.8, min_specificity=.9, save_cache=lambda: None):
    """Search using rigorous F2 bounds, then verify actual unchanged masks.

    F2 weights foreground recall four times as much as precision. Both sides
    must meet precision/background constraints. A–B additionally requires the
    supplied and predicted regions to be small; an underprediction of a large
    supplied region cannot qualify as a successful small match.
    """
    import numpy as np
    from PIL import Image
    partial = group == 'A_B' and high < 1
    bounded = []
    for item in scored:
        row = item['row']
        for prefix, book in zip(('a', 'b'), group.split('_')):
            if prefix + '_gt_coverage' not in item:
                with Image.open(dataset / row[book]['mask_path']) as image:
                    gt = np.asarray(image.convert('L')) > 0
                xs = np.flatnonzero(gt.any(axis=0))
                item[prefix + '_gt_coverage'] = float(gt.mean())
                item[prefix + '_gt_span'] = float((xs[-1]-xs[0]+1)/gt.shape[1]) if len(xs) else 0.
        if partial and any(item[p + '_gt_span'] > high for p in ('a', 'b')):
            item['foreground_status'] = 'supplied_region_too_large'
            continue
        bounds = []
        for prefix in ('a', 'b'):
            q, t = item[prefix + '_iou'], item[prefix + '_gt_coverage']
            # IoU q, GT area t and predicted area p imply intersection=q*(t+p)/(1+q).
            # Bound p using precision, specificity, and the requested mask size.
            upper = min(1., t/q if q else 1.,
                        q*t/(min_precision*(1+q)-q) if min_precision*(1+q)>q else 1.,
                        q*t+(1+q)*(1-min_specificity)*(1-t), high if partial else 1.)
            if upper < max(q*t, low if partial else 0.) - 1e-12 or t == 0:
                bounds.append(0.)
            else:
                intersection = q*(t+upper)/(1+q)
                bounds.append(5*intersection/(4*t+upper))
        item['foreground_f2_upper_bound'] = min(bounds)
        bounded.append(item)
    chosen, measured = [], 0
    for item in sorted(bounded, key=lambda x: (-x['foreground_f2_upper_bound'], x['row']['pair_id'])):
        if len(chosen) == 5 and item['foreground_f2_upper_bound'] < min(chosen[-1]['a_f2'], chosen[-1]['b_f2']) - 1e-12:
            break
        if item['foreground_f2_upper_bound'] == 0:
            continue
        row = item['row']
        if 'a_f2' not in item:
            result = predict_candidate(session, dataset, row, representation)
            for prefix, book, prediction in zip(('a', 'b'), group.split('_'), result['masks']):
                with Image.open(dataset / row[book]['mask_path']) as image:
                    gt = np.asarray(image.convert('L')) > 0
                pred = np.asarray(prediction) > 0
                assert pred.shape == gt.shape
                union = np.logical_or(pred, gt).sum()
                tp = int(np.logical_and(pred, gt).sum())
                actual = float(tp / union) if union else 1.
                assert abs(actual - item[prefix + '_iou']) < 1e-8, 'Cached mask IoU changed'
                xs = np.flatnonzero(pred.any(axis=0))
                item[prefix + '_coverage'] = float(pred.mean())
                item[prefix + '_span'] = float((xs[-1]-xs[0]+1)/pred.shape[1]) if len(xs) else 0.
                item[prefix + '_precision'] = tp/int(pred.sum()) if pred.any() else 0.
                item[prefix + '_recall'] = tp/int(gt.sum()) if gt.any() else 0.
                item[prefix + '_f2'] = 5*tp/(4*int(gt.sum())+int(pred.sum())) if gt.any() or pred.any() else 0.
                item[prefix + '_specificity'] = float(np.logical_and(~pred, ~gt).sum()/np.count_nonzero(~gt)) if (~gt).any() else 1.
            measured += 1
            if measured % 10 == 0:
                save_cache()
                session.feature_cache.clear()
        eligible = all(item[p + '_precision'] >= min_precision and item[p + '_specificity'] >= min_specificity
                       and (not partial or low <= item[p + '_coverage'] <= high and item[p + '_span'] <= high)
                       for p in ('a', 'b'))
        item['foreground_status'] = 'eligible' if eligible else 'outside_measured_constraints'
        print(group, 'foreground check', row['pair_id'], 'IoU', round((item['a_iou']+item['b_iou'])/2, 3),
              'white recall', round(item['a_recall'], 3), round(item['b_recall'], 3), eligible, flush=True)
        if eligible:
            chosen.append(item)
            chosen = sorted(chosen, key=foreground_rank)[:5]
    save_cache()
    if len(chosen) != 5:
        raise ValueError(f'Only {len(chosen)} {group} pairs meet the foreground/background requirements')
    return chosen


def embed_examples(notebook_path, review_path, output):
    """Replace the old categorized section while retaining other notebook cells."""
    notebook = json.loads(notebook_path.read_text())
    review = json.loads(review_path.read_text())
    report = json.loads((output / 'report.json').read_text())
    complete_catalogs = report.get('selection_scope') == 'complete_catalogs'
    fixed_examples = report.get('selection_scope') == 'fixed_previous_checkpoint_examples'
    ab_constraint = report.get('selection_constraints', {}).get('A_B')
    source_cells = {}
    for row in report['samples']:
        matches = [cell for cell in review['cells'] if cell['cell_type']=='code'
                   and row['review_id']+'.png' in ''.join(cell['source'])]
        if len(matches)!=1:
            raise ValueError('Cannot identify the saved review figure: '+row['review_id'])
        source_cells[row['review_id']] = matches[0]

    def markdown(cid, source):
        return dict(cell_type='markdown', id=cid, metadata={}, source=source.splitlines(keepends=True))

    scope = ("Re-evaluated the same 15 examples with the current checkpoint. " if fixed_examples else
             "All 1,000 pairs in each catalog have foreground IoU scores. "
             if complete_catalogs else "Selection uses the scored candidate subset. ")
    scope += report['selection_policy'] + ' '
    if ab_constraint:
        scope += ("A short prediction over a large supplied white region is excluded from the A–B small-region examples. ")
    intro = ("<a id=\"book-pair-samples\"></a>\n## White-mask recovery: selected alignments\n\n"
             "**15 examples: five each for A–B, A–C, and B–C.**\n\n"
             + scope + "Selection ignores continuity, novelty, and transcript variety. "
             "Scores describe these deliberately selected examples and are not a population performance estimate. "
             "Checkpoint split membership is shown on each figure.\n\n"
             "White areas are the actual predicted foreground and the supplied ground-truth foreground. "
             "IoU measures their intersection divided by their union; precision shows how much predicted white is correct, and recall shows how much supplied white was recovered. "
             "Black specificity is the fraction of supplied black that remains black; it is reported separately and does not inflate foreground IoU. "
             "Each figure shows predicted and supplied white masks in the same source-image geometry.\n\n"
             "The frozen checkpoint, matcher parameters, mask generation, and metric formulas are used as configured. "
             "Run the setup cell below to redisplay individual examples; figures are already embedded.\n")
    setup = ("# BOOK-PAIR REVIEW — five selected examples per book pair.\n"
             "from pathlib import Path as BookPairReviewPath\n"
             "import json as book_pair_review_json\n"
             "from IPython.display import Image as BookPairReviewImage, display as display_book_pair_review\n"
             "BOOK_PAIR_REVIEW_ROOT = next(p for p in (BookPairReviewPath.cwd(), *BookPairReviewPath.cwd().parents) if (p / 'evaluation_utils.py').is_file())\n"
             f"BOOK_PAIR_REVIEW_DIR = BOOK_PAIR_REVIEW_ROOT / {str(output.relative_to(ROOT))!r}\n"
             "BOOK_PAIR_REVIEW_REPORT = book_pair_review_json.loads((BOOK_PAIR_REVIEW_DIR / 'report.json').read_text())\n"
             "BOOK_PAIR_REVIEW_SAMPLES = {row['review_id']: row for row in BOOK_PAIR_REVIEW_REPORT['samples']}\n"
             "print('Review checkpoint:', BOOK_PAIR_REVIEW_REPORT['checkpoint'])\n"
             "print('15 selected examples: five for each of A–B, A–C, and B–C.')\n")
    setup_text = f"Review checkpoint: {report['checkpoint']}\n15 selected examples: five for each of A–B, A–C, and B–C.\n"
    cells = [markdown('book-pair-review-intro', intro),
             dict(cell_type='code', id='book-pair-review-setup', metadata={}, execution_count=None,
                  source=setup.splitlines(keepends=True),
                  outputs=[dict(output_type='stream', name='stdout', text=setup_text.splitlines(keepends=True))])]
    for group in GROUPS:
        first_book, second_book = group.split('_')
        group_title = ('five fixed examples' if fixed_examples else
                       'five small-region white-mask matches' if group == 'A_B' and ab_constraint else
                       'five white-mask matches (larger supplied regions)')
        heading = (f"### {group.replace('_', '–')} — {group_title}\n\n"
                   f"Values are shown as **{first_book} / {second_book}**. Recall shows how much of the white mask was caught.\n\n"
                   "| Rank | White IoU | White recall | White precision | Black specificity |\n"
                   "| --- | ---: | ---: | ---: | ---: |\n")
        group_rows = [row for row in report['samples'] if row['book_pair'] == group]
        for rank, row in enumerate(group_rows, 1):
            m = row['metrics']
            q = row.get('foreground_quality', {})
            heading += (f"| {rank} | {m['a_iou']:.3f} / {m['b_iou']:.3f} | "
                f"{m['a_recall']:.1%} / {m['b_recall']:.1%} | {m['a_precision']:.1%} / {m['b_precision']:.1%} | "
                f"{q[first_book]['black_specificity']:.1%} / {q[second_book]['black_specificity']:.1%} |\n")
        if group != 'A_B':
            recalls = [row['metrics'][p+'_recall'] for row in group_rows for p in ('a', 'b')]
            heading += (f"\n**Recovery is limited:** these available results recover {min(recalls):.1%}–{max(recalls):.1%} "
                "of the supplied white foreground. Clean black background does not make these high-recall matches.\n")
        if group == 'A_B' and ab_constraint:
            heading += '\nPredicted white coverage (first / second line): ' + ', '.join(
                f"{row['predicted_white_fraction'][first_book]:.1%} / {row['predicted_white_fraction'][second_book]:.1%}"
                for row in group_rows) + '.\n'
        if complete_catalogs:
            ranking_path = Path('..') / output.relative_to(ROOT) / f'ranking_{group}.csv'
            heading += f"\n[Full catalog ranking]({ranking_path.as_posix()})\n"
        cells.append(markdown(f'book-pair-review-{group}-heading', heading))
        for row in report['samples']:
            if row['book_pair'] != group:
                continue
            cell = copy.deepcopy(source_cells[row['review_id']])
            cell['id'] = 'book-pair-review-' + row['review_id']
            code = ''.join(cell['source'])
            code = code.replace('from IPython.display import Image, display\n', '')
            code = code.replace('display(Image(filename=str(REVIEW / ',
                                'display_book_pair_review(BookPairReviewImage(filename=str(BOOK_PAIR_REVIEW_DIR / ')
            cell['source'] = code.splitlines(keepends=True)
            cells.append(cell)
    notebook['cells'] = [c for c in notebook['cells'] if not c.get('id', '').startswith('book-pair-review-')]
    index = next(i for i, c in enumerate(notebook['cells']) if c.get('id') == 'evaluation-006')
    notebook['cells'][index:index] = cells
    first = notebook['cells'][0]
    source = ''.join(first['source'])
    import re
    source = re.sub(r'\[View \d+ book-pair samples: A–B, A–C, B–C\]\(#book-pair-samples\)',
                    '[View 15 white-mask alignment examples](#book-pair-samples)', source)
    source = source.replace('[View 15 selected book-pair examples](#book-pair-samples)',
                            '[View 15 white-mask alignment examples](#book-pair-samples)')
    source = source.replace('[View 15 best mask-IoU alignments](#book-pair-samples)',
                            '[View 15 white-mask alignment examples](#book-pair-samples)')
    first['source'] = source.splitlines(keepends=True)
    notebook['metadata']['book_pair_review'] = dict(checkpoint=report['checkpoint'],
        checkpoint_sha256=report['checkpoint_sha256'], settings=report['settings'],
        settings_source=report['settings_source'], samples=15, samples_per_pair=5,
        selection=report['ranking_method'], selection_scope=report.get('selection_scope'),
        selection_constraints=report.get('selection_constraints', {}),
        source_notebook=str(review_path.relative_to(ROOT)))
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')


def main():
    import numpy as np
    from PIL import Image
    import torch
    from evaluation_utils import EvaluationSession, _side_key, compute_pair_metrics, load_inference_text_encoder
    from dataset import file_hash
    from train import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-limit', type=int, default=0, help='0 scores the complete catalog')
    parser.add_argument('--group', choices=GROUPS, help='Score one independent book-pair catalog')
    parser.add_argument('--scan-only', action='store_true', help='Save candidate scores without changing notebooks')
    parser.add_argument('--render-only', action='store_true', help='Require cached scores for every candidate')
    parser.add_argument('--ab-min-coverage', type=float, default=.10)
    parser.add_argument('--ab-max-coverage', type=float, default=.45)
    parser.add_argument('--min-white-precision', type=float, default=.8)
    parser.add_argument('--min-black-specificity', type=float, default=.9)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'Results/Evaluation/book_pairs_examples')
    args = parser.parse_args()
    if args.group and not args.scan_only:
        parser.error('--group requires --scan-only; render all three catalogs together')
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if not 0 <= args.ab_min_coverage < args.ab_max_coverage <= 1:
        parser.error('A–B coverage bounds must satisfy 0 <= min < max <= 1')
    if not 0 < args.min_white_precision <= 1 or not 0 < args.min_black_specificity <= 1:
        parser.error('Precision and specificity must be in (0, 1]')
    scan_lock = None
    if args.group:
        import fcntl
        args.output.mkdir(parents=True, exist_ok=True)
        scan_lock = (args.output / ('.scan_' + args.group + '.lock')).open('a')
        fcntl.flock(scan_lock.fileno(), fcntl.LOCK_EX)
    args.dataset = ROOT / 'DataSet/ArabicData'
    args.settings_notebook = ROOT / 'notebooks/model_evaluation.ipynb'
    args.notebook = ROOT / 'notebooks/book_pairs_qualitative.ipynb'
    args.settings_report = None
    args.threads = 4
    setup = load_evaluation_setup(args.settings_notebook)
    args.crop, args.device, args.representation = setup['EVALUATION_CROP'], setup['DEVICE'], setup['REPRESENTATION']
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, _, config, saved = load_checkpoint(args.checkpoint, device, load_text=False)
    config = replace(config, crop=args.crop)
    session = object.__new__(EvaluationSession)
    session.model, session.config, session.saved, session.device = model, config, saved, device
    session.settings = load_settings(args.settings_notebook)
    session._text_encoder = load_inference_text_encoder(saved, config, device)
    session.feature_cache, session.metric_cache, session.line_aliases, session.allowed = {}, {}, {}, {}
    session.split = 'illustrative_examples'
    membership = {sid: split for split, ids in saved['split_ids'].items() for sid in ids}
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = file_hash(args.checkpoint)
    legacy_fingerprint = hashlib.sha256(json.dumps(dict(settings=session.settings, checkpoint=str(args.checkpoint),
        crop=args.crop, representation=args.representation), sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(dict(settings=session.settings, checkpoint_sha256=checkpoint_sha256,
        crop=args.crop, representation=args.representation, device=str(device)), sort_keys=True).encode()).hexdigest()
    # Reuse earlier scores only after verifying their actual checkpoint and options.
    reference_path = args.output / 'report.json'
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    legacy_verified = (reference.get('checkpoint_sha256') == checkpoint_sha256
                       and reference.get('settings') == session.settings
                       and reference.get('device') == str(device)
                       and reference.get('representation') == args.representation
                       and reference.get('config', {}).get('crop') == args.crop)
    cache = {}
    for previous_path in [args.output / 'candidate_scores.json', *sorted(args.output.glob('candidate_scores_*.json'))]:
        if not previous_path.exists():
            continue
        previous = json.loads(previous_path.read_text())
        if previous.get('fingerprint') == fingerprint or (legacy_verified and previous.get('fingerprint') == legacy_fingerprint):
            for group, values in previous.get('groups', {}).items():
                merged = {item['row']['pair_id']: item for item in cache.get(group, [])}
                for item in values:
                    identity = item['row']['pair_id']
                    merged[identity] = {**item, **merged.get(identity, {})}
                cache[group] = list(merged.values())
    cache_path = args.output / ('candidate_scores_' + args.group + '.json' if args.group else 'candidate_scores.json')
    selected, inventory = [], {}
    for group in ((args.group,) if args.group else GROUPS):
        books = group.split('_')
        rows = [json.loads(line) for line in (args.dataset / 'alignedPairs' / group / 'pairs.jsonl').read_text().splitlines() if line.strip()]
        existing = {item['row']['pair_id']: item for item in cache.get(group, [])}
        ordered = sorted(rows, key=lambda row: row['pair_id'])
        # Cached scores affect execution order only, never which pairs qualify.
        known = [row for row in ordered if row['pair_id'] in existing]
        unseen = sorted([row for row in ordered if row['pair_id'] not in existing],
                        key=lambda row: hashlib.sha256(row['pair_id'].encode()).hexdigest())
        candidates = (known + unseen)[:args.candidate_limit or None]
        scored = []
        for index, row in enumerate(candidates, 1):
            row = dict(row, book_pair=group)
            if row['pair_id'] in existing:
                scored.append(existing[row['pair_id']])
            else:
                if args.render_only:
                    raise ValueError(f'Missing score for {group}/{row["pair_id"]}; run the scan first')
                result = predict_candidate(session, args.dataset, row, args.representation)
                truth = []
                for b in books:
                    with Image.open(args.dataset / row[b]['mask_path']) as image:
                        truth.append(np.asarray(image.convert('L')) > 0)
                metrics, _ = compute_pair_metrics(result, config, preloaded_ground_truth=truth)
                scored.append(dict(row=row, a_iou=metrics['a_iou'], b_iou=metrics['b_iou']))
            if index % 100 == 0 or index == len(candidates):
                cache[group] = scored
                stored = {args.group: scored} if args.group else cache
                temporary = cache_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(fingerprint=fingerprint, checkpoint_sha256=checkpoint_sha256,
                    settings=session.settings, groups=stored), ensure_ascii=False, indent=1))
                temporary.replace(cache_path)
                best = choose_best(scored)
                print(group, 'scored', index, 'best paired IoUs', [round((r['a_iou']+r['b_iou'])/2, 3) for r in best], flush=True)
                session.feature_cache.clear()
                if device.type == 'mps':
                    torch.mps.empty_cache()
        best = choose_best(scored)
        if not args.scan_only:
            cache[group] = scored
            def save_cache():
                temporary = cache_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(fingerprint=fingerprint, checkpoint_sha256=checkpoint_sha256,
                    settings=session.settings, groups=cache), ensure_ascii=False, indent=1))
                temporary.replace(cache_path)
            best = choose_foreground(scored, session, args.dataset, args.representation, group,
                args.ab_min_coverage, args.ab_max_coverage, args.min_white_precision,
                args.min_black_specificity, save_cache)
        if len(best) != 5:
            raise ValueError(f'Need five {group} pairs')
        for index, item in enumerate(best, 1):
            row = dict(item['row'], review_id=f'{group}_example_{index:02d}', path_type='example',
                saved_line_splits={b: membership.get(f"Book{b}:" + item['row'][b]['line_id'], 'unseen') for b in books},
                selection_iou=dict(first=item['a_iou'], second=item['b_iou']))
            if 'a_coverage' in item:
                row['selection_white_fraction'] = dict(first=item['a_coverage'], second=item['b_coverage'])
                row['selection_foreground'] = {p: {k: item[p+'_'+k] for k in
                    ('f2', 'recall', 'precision', 'specificity', 'gt_coverage', 'coverage', 'span')}
                    for p in ('a', 'b')}
            selected.append(row)
        inventory[group] = dict(available=len(rows), candidates_scored=len(scored), selected=5,
            supplied_small_pairs=sum(all(item.get(p+'_gt_span', 2) <= args.ab_max_coverage for p in ('a', 'b')) for item in scored))
        with (args.output / f'ranking_{group}.csv').open('w', newline='') as stream:
            fields = ('rank', 'pair_id', 'first_mask_iou', 'second_mask_iou', 'mean_mask_iou',
                      'selected_for_gallery', 'predicted_first_white_fraction', 'predicted_second_white_fraction',
                      'first_white_recall', 'second_white_recall', 'first_white_precision', 'second_white_precision',
                      'first_black_specificity', 'second_black_specificity', 'weaker_white_f2', 'foreground_status')
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            ranked = sorted(scored, key=lambda item: (-(item['a_iou'] + item['b_iou']) / 2,
                            -min(item['a_iou'], item['b_iou']), item['row']['pair_id']))
            for rank, item in enumerate(ranked, 1):
                writer.writerow(dict(rank=rank, pair_id=item['row']['pair_id'],
                    first_mask_iou=item['a_iou'], second_mask_iou=item['b_iou'],
                    mean_mask_iou=(item['a_iou'] + item['b_iou']) / 2,
                    selected_for_gallery=item['row']['pair_id'] in {x['row']['pair_id'] for x in best},
                    predicted_first_white_fraction=item.get('a_coverage'), predicted_second_white_fraction=item.get('b_coverage'),
                    first_white_recall=item.get('a_recall'), second_white_recall=item.get('b_recall'),
                    first_white_precision=item.get('a_precision'), second_white_precision=item.get('b_precision'),
                    first_black_specificity=item.get('a_specificity'), second_black_specificity=item.get('b_specificity'),
                    weaker_white_f2=min(item['a_f2'], item['b_f2']) if 'a_f2' in item else None,
                    foreground_status=item.get('foreground_status', 'bounded_out')))
    if args.scan_only:
        print('Candidate scan complete:', inventory, flush=True)
        return
    cache_path.write_text(json.dumps(dict(fingerprint=fingerprint, checkpoint_sha256=checkpoint_sha256,
        settings=session.settings, groups=cache), ensure_ascii=False, indent=1))
    complete_catalogs = all(item['available'] == item['candidates_scored'] for item in inventory.values())
    selection = dict(mode='illustrative_examples', samples_per_pair=5, samples=selected, inventory=inventory,
        scope='complete_catalogs' if complete_catalogs else 'scored_subset',
        ranking_method='weaker-side white F2, then mean foreground IoU, then weaker-side black specificity',
        policy='Five pairs per book pair with the highest mean of the two line mask IoUs; ties favor the stronger weaker-side IoU. '
               + ('The complete catalogs are scored with frozen parameters. ' if complete_catalogs else 'An explicitly limited subset is scored with frozen parameters. ')
               + 'No continuity, novelty, or transcript-variety filtering is applied. '
               'These are illustrative results, not a population performance estimate. Split labels refer to the current checkpoint.',
        split_manifest_sha256=hashlib.sha256((args.checkpoint.parent / 'split_manifest.json').read_bytes()).hexdigest())
    selection['constraints'] = dict(all_groups=dict(min_white_precision=args.min_white_precision,
        min_black_specificity=args.min_black_specificity))
    selection['policy'] = (f'Prioritize recovery of the supplied white mask on BOTH lines: rank by the weaker-side '
        f'white F2 (recall weighted four times as much as precision), then mean foreground IoU, then black specificity. '
        f'Each line must have white precision >= {args.min_white_precision:.0%} and black specificity >= {args.min_black_specificity:.0%}. ')
    if args.ab_max_coverage < 1:
        selection['constraints']['A_B'] = dict(min_white_fraction=args.ab_min_coverage,
            max_white_fraction=args.ab_max_coverage, max_span_fraction=args.ab_max_coverage,
            max_supplied_span_fraction=args.ab_max_coverage)
        selection['policy'] += (f'A–B: supplied white spans at most {args.ab_max_coverage:.0%} of each line; '
            f'predicted white covers {args.ab_min_coverage:.0%}–{args.ab_max_coverage:.0%} within a '
            f'{args.ab_max_coverage:.0%} span. A–C/B–C have '
            f'{inventory["A_C"]["supplied_small_pairs"]}/{inventory["B_C"]["supplied_small_pairs"]} pairs with supplied white confined to '
            f'{args.ab_max_coverage:.0%} of both lines; their examples have no size restriction. ')
    selection['policy'] += ('All catalog IoUs are scored; rigorous bounds prune foreground ranking candidates before exact verification. '
        'No continuity categories. These are selected examples, not population performance. Model, matcher and masks are unchanged.')
    args.selection = args.output / 'selection.json'
    args.selection.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    assert session.settings == load_settings(args.settings_notebook), 'Notebook parameters changed during selection'
    del session, model
    if device.type == 'mps':
        torch.mps.empty_cache()
    evaluate(args, selection)
    embed_examples(args.settings_notebook, args.notebook, args.output)
    print('Saved 15 selected examples in both evaluation notebooks.', flush=True)


if __name__ == '__main__':
    main()

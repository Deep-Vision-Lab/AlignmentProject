"""Select fragment acceptance on validation and score one final matcher on test.

The two user-inspected pairs and every comparison sharing either source image
are development data. Their masks are never inputs to candidate generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from dataset import file_hash
from evaluate import MATCH_VERSION, resolved_match_settings
from parameters import Config
from scripts.eval import mask_quantitative as q
from scripts.eval.review_reordered import prepare_cohorts, run_methods
from scripts.eval.tune_matching import write_json, write_csv

DEVELOPMENT_PAIRS = ('pair_000016__A_line_014__B_line_003',
                     'pair_000028__A_line_005__B_line_002')


def fragment_presets(baseline):
    """Include unequal-width words: two source windows may match three or more."""
    methods = []
    for support in (2, 3):
        for cosine in (.65, .70, .75, .80):
            for fraction in (.4, .6):
                for anchors in (7, 9):
                    settings = dict(baseline, allow_reordered_regions=True,
                        split_gap_candidates=True, require_corroborated_short_regions=True,
                        normal_min_distinct_a=9, normal_min_distinct_b=9,
                        normal_min_matched_pairs=12, normal_min_mean_cosine=.6,
                        min_path_density=.85, min_mutual_anchors_long=anchors,
                        allow_short_regions=True, short_min_distinct_a=support,
                        short_min_distinct_b=support, short_min_matched_pairs=4,
                        short_min_mean_cosine=cosine, short_min_median_cosine=cosine,
                        short_min_mutual_anchor_fraction=fraction,
                        min_mutual_anchor_fraction_short=fraction)
                    name = f'support{support}_cos{cosine}_fraction{fraction}_anchors{anchors}'
                    methods.append((name, resolved_match_settings(settings)))
    return methods


def exclude_reviewed(prepared, image_hashes):
    excluded, retained = [], []
    for item in prepared:
        if any(image['sha256'] in image_hashes for image in item['record']['images']):
            excluded.append(item['record']['sample_id'])
        else:
            retained.append(item)
    return retained, excluded


def select_final(results, max_fpr):
    eligible = [row for row in results if row['metrics']['negative_pair_fpr'] <= max_fpr]
    if not eligible:
        raise ValueError('No validation preset meets the requested false-positive budget')
    return max(eligible, key=lambda row: (row['metrics']['recall'],
        row['metrics']['positive_iou'], -row['metrics']['negative_pair_fpr'], row['name']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Results/Evaluation/final_fragments_20261109')
    parser.add_argument('--prepared-source', type=Path,
        default=ROOT/'Results/Evaluation/reordered_regions_20261109')
    parser.add_argument('--checkpoint', type=Path,
        default=ROOT/'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    parser.add_argument('--max-negative-pair-fpr', type=float, default=.10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if not 0 <= args.max_negative_pair_fpr <= 1:
        parser.error('FPR must be in [0, 1]')
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    # Reuse image features, never old metric values. Generation signatures also
    # verify checkpoint, matcher source, settings and exact feature bytes.
    for split in ('val', 'test'):
        source = args.prepared_source/(split+'_prepared.pt')
        destination = out/(split+'_prepared.pt')
        if source.is_file() and not destination.exists():
            destination.symlink_to(source.resolve())
    prepare_cohorts(out, args.checkpoint, args.device)
    catalog = ROOT/'DataSet/ArabicData/alignedPairs/A_B/pairs.jsonl'
    rows = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    reviewed = [next(row for row in rows if row['pair_id'] == name) for name in DEVELOPMENT_PAIRS]
    hashes = {file_hash(ROOT/'DataSet/ArabicData'/row[book]['image_path'])
              for row in reviewed for book in ('A', 'B')}
    validation = torch.load(out/'val_prepared.pt', map_location='cpu', weights_only=False)
    config = Config(**validation['provenance']['config'])
    val, val_excluded = exclude_reviewed(validation['prepared'], hashes)
    methods = fragment_presets(validation['provenance']['baseline_settings'])
    write_json(out/'selection_protocol.json', dict(
        objective='Maximum validation foreground recall, then foreground IoU',
        max_validation_negative_pair_fpr=args.max_negative_pair_fpr,
        presets=[dict(name=name, settings=settings) for name, settings in methods],
        development_pairs=list(DEVELOPMENT_PAIRS), excluded_validation_pairs=val_excluded,
        checkpoint_sha256=file_hash(args.checkpoint), matcher_version=MATCH_VERSION))
    for target, source in [('validation', 'validation_anchored'), ('test', 'test_current_anchored')]:
        previous = args.prepared_source/source/'candidate_cache'
        cache = out/target/'candidate_cache'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if previous.is_dir() and not cache.exists():
            cache.symlink_to(previous.resolve(), target_is_directory=True)
    torch.set_num_threads(1)
    trials = run_methods(args.checkpoint, config, 'val', val, methods, out/'validation', args.workers)
    selected = select_final(trials, args.max_negative_pair_fpr)
    frozen = dict(name=selected['name'], settings=selected['settings'],
        validation_metrics=selected['metrics'], checkpoint_sha256=file_hash(args.checkpoint),
        matcher_version=MATCH_VERSION, selection='Maximum validation foreground recall subject to presumed-negative pair FPR <= '+str(args.max_negative_pair_fpr),
        max_validation_negative_pair_fpr=args.max_negative_pair_fpr)
    write_json(out/'selected.json', frozen)
    print('FROZEN VALIDATION SELECTION', json.dumps(frozen), flush=True)
    test = torch.load(out/'test_prepared.pt', map_location='cpu', weights_only=False)
    cohort, excluded = exclude_reviewed(test['prepared'], hashes)
    write_json(out/'test_cohort.json', dict(records=[item['record'] for item in cohort],
        excluded_development_pairs=excluded, development_pairs=list(DEVELOPMENT_PAIRS),
        reason='User-inspected pair or shared source image', original_pairs=len(test['prepared']),
        evaluated_pairs=len(cohort)))
    final = run_methods(args.checkpoint, config, 'test', cohort,
        [('Final matcher', selected['settings'])], out/'test', args.workers)[0]
    final['metrics']['confidence_intervals'] = q.confidence_intervals(final['rows'], resamples=1000, seed=42)
    final['by_book_pair'] = {}
    for group in ('A_B', 'A_C', 'B_C'):
        ids = {item['record']['sample_id'] for item in cohort if item['record'].get('source_folder') == group}
        if ids:
            final['by_book_pair'][group] = q.aggregate([row for row in final['rows']
                if row['sample_id'] in ids or row['category']=='negative'])
    report = dict(checkpoint=str(args.checkpoint.relative_to(ROOT)), selected=frozen,
        final={key:value for key,value in final.items() if key!='rows'},
        development_pairs=list(DEVELOPMENT_PAIRS), excluded_test_pairs=excluded,
        validation_provenance=validation['provenance'], test_provenance=test['provenance'],
        source_sha256={name:file_hash(ROOT/name) for name in (
            'alignment_candidates.py', 'evaluate.py', 'evaluation_utils.py', 'scripts/eval/final_fragments.py')},
        interpretation='Saved-test evaluation after validation-only parameter selection. Both supplied cases and shared images are excluded. Test results were reported, never used to reselect settings. This cohort has appeared in earlier aggregate reports; it is not a new independent benchmark.')
    write_json(out/'final.json', report)
    write_csv(out/'summary.csv', [dict(method='Final matcher', **{key:value for key,value in final['metrics'].items() if not isinstance(value,dict)})])
    print('FINAL TEST METRICS', json.dumps(final['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()

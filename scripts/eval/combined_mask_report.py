"""Pool saved validation and test mask scores after validation-only selection.

This is a descriptive report. Optuna never sees test pairs, and the separate
test result remains the independent estimate of a selected configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from dataset import file_hash
from scripts.eval.mask_quantitative import aggregate, confidence_intervals, identity
from scripts.eval.tune_matching import write_csv, write_json


FLOAT_COLUMNS = ('positive_iou', 'positive_dice', 'precision', 'recall',
                 'negative_masked_area_fraction')
INT_COLUMNS = ('region_tp', 'region_fp', 'region_fn')
BOOL_COLUMNS = ('presumed_negative', 'negative_false_positive', 'candidate_limit_hit',
                'region_eligible')


def read_rows(path, split, expected, *, lines=False):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'No saved scores in {path}')
    for row in rows:
        row['split'] = split
        if lines:
            if row.get('side') not in ('A', 'B'):
                raise ValueError(f'Invalid line side in {path}')
            if row['sample_id'] not in expected or row['category'] != expected[row['sample_id']]:
                raise ValueError(f'Line IDs or labels differ from the {split} manifest: {path}')
            continue
        for key in FLOAT_COLUMNS:
            row[key] = float(row[key])
        for key in INT_COLUMNS:
            row[key] = int(row[key])
        for key in BOOL_COLUMNS:
            if row[key] not in ('True', 'False'):
                raise ValueError(f'Invalid {key} in {path}')
            row[key] = row[key] == 'True'
    if lines:
        counts = {sample: {r['side'] for r in rows if r['sample_id'] == sample}
                  for sample in {r['sample_id'] for r in rows}}
        if any(sides != {'A', 'B'} for sides in counts.values()):
            raise ValueError(f'Each pair needs both line scores in {path}')
        return rows
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError(f'Saved pair IDs do not match the {split} split manifest: {path}')
    if any(r['category'] != expected[r['sample_id']] for r in rows):
        raise ValueError(f'Saved labels do not match the {split} split manifest: {path}')
    return rows


def verified_selection(output):
    selected = json.loads((output/'selected.json').read_text())
    digest = selected.pop('selection_sha256')
    if identity(selected) != digest:
        raise ValueError('Selected validation configuration has an invalid checksum')
    selected['selection_sha256'] = digest
    marker = output/'test/selection_used.json'
    if not marker.is_file() or json.loads(marker.read_text()) != selected:
        raise ValueError('Run final test with the frozen validation selection first')
    if json.loads((output/'run_manifest.json').read_text()) != selected['run_manifest']:
        raise ValueError('Current run manifest differs from the frozen selection')
    return selected


def report(output, resamples=None, seed=None):
    output = Path(output).expanduser().resolve()
    selected = verified_selection(output)
    manifest = selected['run_manifest']
    resamples = int(resamples if resamples is not None else manifest['bootstrap_resamples'])
    seed = int(seed if seed is not None else manifest['seed'])
    if resamples < 1:
        raise ValueError('Bootstrap resamples must be positive')
    expected = {}
    for split in ('val', 'test'):
        contract = json.loads((output/f'{split}_split_manifest.json').read_text())
        expected[split] = {r['sample_id']: r['category'] for r in contract['records']}
        if len(expected[split]) != len(contract['records']):
            raise ValueError(f'Duplicate IDs in {split} split manifest')
        if split == 'val' and identity(contract) != manifest['split_manifest_sha256']:
            raise ValueError('Validation split manifest differs from the tuned population')
        image_hashes = {image['sha256'] for record in contract['records'] for image in record['images']}
        if split == 'test' and image_hashes & val_images:
            raise ValueError('Validation and test source images overlap')
        if split == 'val':
            val_images = image_hashes
    if set(expected['val']) & set(expected['test']):
        raise ValueError('Validation and test pair IDs overlap')
    paths = {
        'Current baseline': (output/'baseline', output/'test/method_0'),
        'Optuna selected': (output/'trials'/f"stage{selected['stage']}_{selected['trial']:05d}",
                            output/'test/method_1'),
    }
    methods = json.loads((output/'test/methods.json').read_text())
    if [m['method'] for m in methods[:2]] != list(paths):
        raise ValueError('Saved test methods differ from the expected baseline and selection')
    combined_pairs, combined_lines, summaries = [], [], []
    for method, (val_path, test_path) in paths.items():
        per_split = {}
        for split, directory in (('val', val_path), ('test', test_path)):
            pairs = read_rows(directory/'per_pair.csv', split, expected[split])
            lines = read_rows(directory/'per_line.csv', split, expected[split], lines=True)
            if {r['sample_id'] for r in lines} != set(expected[split]) or len(lines) != 2*len(pairs):
                raise ValueError(f'Line scores do not match pair scores: {directory}')
            per_split[split] = pairs
            combined_pairs.extend(dict(method=method, **r) for r in pairs)
            combined_lines.extend(dict(method=method, **r) for r in lines)
        if method == 'Optuna selected':
            observed = aggregate(per_split['val'])
            for key in ('positive_iou', 'negative_pair_fpr'):
                if not math.isclose(observed[key], selected['metrics'][key], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('Selected validation scores differ from the frozen trial')
        saved_test = methods[list(paths).index(method)].get('metrics')
        if saved_test:
            observed = aggregate(per_split['test'])
            for key in ('positive_iou', 'negative_pair_fpr'):
                if not math.isclose(observed[key], saved_test[key], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError('Test pair scores differ from saved test summary')
        pooled = per_split['val'] + per_split['test']
        metrics = aggregate(pooled)
        metrics['confidence_intervals'] = confidence_intervals(pooled, resamples, seed)
        metrics['presumed_negative_pairs'] = sum(r['presumed_negative'] for r in pooled)
        metrics['interpretation'] = ('Descriptive validation + test pool after validation-only selection; '
                                     'use the separate test result for an independent estimate. '
                                     'Folder-derived negative labels are presumed, not manually verified.'
                                     if metrics['presumed_negative_pairs'] else
                                     'Descriptive validation + test pool after validation-only selection; '
                                     'use the separate test result for an independent estimate.')
        summaries.append(dict(method=method,
                              validation=aggregate(per_split['val']),
                              test=aggregate(per_split['test']), combined=metrics))
    destination = output/'combined_validation_test'
    destination.mkdir(exist_ok=True)
    write_csv(destination/'per_pair.csv', combined_pairs)
    write_csv(destination/'per_line.csv', combined_lines)
    write_json(destination/'metrics.json', dict(
        selection_sha256=selected['selection_sha256'],
        checkpoint_sha256=manifest['checkpoint_sha256'],
        split_manifest_sha256={split:file_hash(output/f'{split}_split_manifest.json') for split in expected},
        seed=seed, bootstrap_resamples=resamples, methods=summaries))
    write_csv(destination/'summary.csv', [dict(method=s['method'], split=split,
        **{k:v for k,v in s[split].items() if not isinstance(v,dict)})
        for s in summaries for split in ('validation','test','combined')])
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='Existing completed tuner output with frozen final test')
    parser.add_argument('--bootstrap', type=int, help='Default: original run setting')
    parser.add_argument('--seed', type=int, help='Default: original run seed')
    args = parser.parse_args(argv)
    for result in report(args.output, args.bootstrap, args.seed):
        for split in ('validation','test','combined'):
            score = result[split]
            print(f"{result['method']} {split}: {score['positive_pairs']} positive, "
                  f"{score['negative_pairs']} negative, IoU={score['positive_iou']:.4f}, "
                  f"presumed-negative FPR={score['negative_pair_fpr']:.4f}", flush=True)


if __name__ == '__main__':
    main()

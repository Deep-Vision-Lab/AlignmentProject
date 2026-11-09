"""Publish only the final measured matcher and both supplied failure cases."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import gzip
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from evaluation_utils import compute_pair_metrics, plot_pair_alignment, revalidate_cached_candidates
from scripts.eval.tune_matching import write_json, write_csv


def case_artifacts(out, report):
    cases = []
    dataset = ROOT/'DataSet/ArabicData'
    session = None
    for name in report['development_pairs']:
        pool_path = out/(name+'_pool.pkl.gz')
        if pool_path.exists():
            with gzip.open(pool_path, 'rb') as stream:
                saved = pickle.load(stream)
            result = revalidate_cached_candidates(
                SimpleNamespace(config=saved['config'], split='development'),
                saved['prediction'], report['selected']['settings'])
            row, config = saved['row'], saved['config']
        else:
            from evaluation_utils import EvaluationSession
            from scripts.eval.select_book_pair_examples import predict_candidate
            if session is None:
                session = EvaluationSession(ROOT/report['checkpoint'], dataset,
                    device='cpu', pair_source='folders', negative_count=0, crop_override='content')
            session.settings = report['selected']['settings']
            row = next(json.loads(line) for line in (dataset/'alignedPairs/A_B/pairs.jsonl').read_text().splitlines()
                       if json.loads(line)['pair_id']==name)
            row['book_pair'] = 'A_B'
            result = predict_candidate(session, dataset, row, 'fused')
            config = session.config
            with gzip.open(pool_path, 'wb') as stream:
                pickle.dump(dict(prediction=result, row=row, config=config), stream)
        # Ground truth is opened only after prediction has been completed.
        truth = [np.asarray(Image.open(dataset/row[b]['mask_path']).convert('L'))>0 for b in 'AB']
        result['metrics'], result['ground_truth'] = compute_pair_metrics(result, config, preloaded_ground_truth=truth)
        coverage = []
        for side, book in enumerate('AB'):
            coverage.append([dict(interval=[left,right], foreground_recall=float(
                ((result['masks'][side]>0)&truth[side])[:,left:right].sum()/max(1,truth[side][:,left:right].sum())))
                for left,right in row[book]['mask_regions_x']])
        fig = plot_pair_alignment(SimpleNamespace(config=config, split='development'), result, show_gt=True)
        fig.suptitle('FINAL MATCHER | '+name+'\nUser-inspected development example; excluded from aggregate test metrics', fontsize=11)
        fig.savefig(out/(name+'.png'), dpi=90);plt.close(fig)
        case = dict(pair_id=name, metrics=result['metrics'], fragment_coverage=coverage,
            accepted_regions=[{key:c[key] for key in ('logical_ranges','support','matched_pairs',
                'mean_cosine','median_cosine','mutual_anchor_count','mutual_anchor_fraction')}
                for c in result['match']['regions']], image=name+'.png')
        with gzip.open(out/(name+'_final.pkl.gz'), 'wb') as stream:
            pickle.dump(result, stream)
        cases.append(case)
    return cases


def metric_table(metrics):
    return '\n'.join(['| Metric | Final value |','| --- | ---: |']+
        [f'| {label} | {metrics[key]:.2%} |' for label,key in (
            ('Foreground mask IoU','positive_iou'),('Dice','positive_dice'),
            ('Foreground precision','precision'),('Foreground recall','recall'),
            ('Positive-pair detection recall','positive_pair_detection_recall'),
            ('Presumed-negative pair false-positive rate','negative_pair_fpr'),
            ('Presumed-negative masked area','negative_masked_area_fraction'))])


def render(out):
    report = json.loads((out/'final.json').read_text())
    report['cases'] = case_artifacts(out, report)
    metrics = report['final']['metrics'];val = report['selected']['validation_metrics']
    budget = report['selected']['max_validation_negative_pair_fpr']
    status = ('exceeds' if metrics['negative_pair_fpr'] > budget else 'meets')
    versioned = ROOT/'docs/evaluation/final_fragments_20261109'
    versioned.mkdir(parents=True, exist_ok=True)
    write_json(versioned/'metrics.json', report)
    write_json(versioned/'settings.json', report['selected']['settings'])
    write_csv(versioned/'summary.csv', [dict(method='Final matcher', **{k:v for k,v in metrics.items() if not isinstance(v,dict)})])
    with (Path(report['final']['directory'])/'per_pair.csv').open() as stream:
        write_csv(versioned/'per_pair.csv', list(csv.DictReader(stream)))
    trials = json.loads((out/'validation/methods.json').read_text())
    write_csv(versioned/'validation_presets.csv', [dict(preset=t['name'], **{k:v for k,v in t['metrics'].items() if not isinstance(v,dict)}) for t in trials])
    lines = ['# Final optional-space alignment evaluation — 9 November 2026','',
        f"Checkpoint: `{report['checkpoint']}`. Matcher: `{report['selected']['matcher_version']}`. Representation: fused; crop: content.",'',
        f"Final results use **{metrics['positive_pairs']} saved-test positives + {metrics['negative_pairs']} seeded presumed-negative comparisons**. Both supplied pairs and comparisons sharing any of their source images are excluded. This is a fresh execution of the final settings on the saved-test cohort, which has appeared in earlier aggregate reports; it is not a new independent benchmark.",'',
        metric_table(metrics),'',
        f"**Test false-positive rate {status} the requested {budget:.0%} budget.** Validation qualification does not establish a reliable deployment ceiling.",'',
        f"False detections: **{metrics['negative_false_positive_pairs']}/{metrics['negative_pair_denominator']}**. Positive pairs with any detected region: **{metrics['positive_detected_pairs']}/{metrics['positive_pairs']}**. Foreground metrics are macro means over positive pairs (each pair averages its two line masks). Pair detection does not imply correct localization or correspondence.",'',
        f"Selection maximized validation foreground recall among 32 declared presets with presumed-negative pair FPR ≤ {report['selected']['max_validation_negative_pair_fpr']:.0%}. Validation: {val['positive_pairs']} positives, {val['negative_pairs']} negatives; recall {val['recall']:.2%}, FPR {val['negative_false_positive_pairs']}/{val['negative_pairs']} ({val['negative_pair_fpr']:.2%}). The selected settings were frozen before scoring test. The test FPR is measured independently and is not guaranteed to satisfy the validation budget.",'',
        '## What causes the missed passages?','',
        'Each passage must preserve its internal character order. Separate passages may change order, provided their window spans do not overlap on either line. Gap-free fragments are considered independently instead of being forced into one continuous path. Optional-space DTW follows the saved training alphabet and skip penalty; it cannot by itself remove image-to-image support and confidence rejections.','',
        '`pair_000016__A_line_014__B_line_003`: the short “الله” correspondence has a strong candidate with two distinct windows on A, three on B, four matched cells, mean cosine about 0.809 and three mutual anchors. A three-window minimum on each side rejects it even though the long “من المصلح ولو” passage is detected. Unequal window counts reflect writing scale and spacing; matched-cell count must not be mistaken for distinct support.','',
        '`pair_000028__A_line_005__B_line_002`: the supplied fragments “ابتغاء”, “الله”, and “من” occur in the same order. Its failure is not solely a reversed-order restriction. The full “ابتغاء” route is fragmented: one visually strong core supports only two A windows / three B windows and has only one mutual anchor (fraction 0.25). Some competing routes match the wrong word. “من” has a viable partial route near mean cosine 0.679. Relaxing confidence can recover foreground while also accepting wrong links. Masks measure location, not word identity; counting accepted regions cannot establish that every fragment was correctly paired.','',
        '## Final predictions for the two supplied cases','']
    for case in report['cases']:
        lines += [f"- `{case['pair_id']}`: {len(case['accepted_regions'])} accepted regions; mask IoU **{case['metrics']['pair_mean_iou']:.2%}**. Final figure is embedded in both notebooks."]
    r = metrics['region_f1']
    lines += ['', '| Book pair | Positive pairs | Mask IoU | Dice | Precision | Recall |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group, values in report['final']['by_book_pair'].items():
        lines += [f"| {group.replace('_','–')} | {values['positive_pairs']} | {values['positive_iou']:.2%} | {values['positive_dice']:.2%} | {values['precision']:.2%} | {values['recall']:.2%} |"]
    lines += ['', '## Additional metrics and limits','',
        f"Region interval F1: {r['f1']:.2%}; precision {r['precision']:.2%}; recall {r['recall']:.2%}. Counts over both lines of positive pairs: TP={r['tp']}, FP={r['fp']}, FN={r['fn']}. Candidate-cap hits: {metrics['candidate_limit_hits']}.",'',
        'Negatives are presumed unrelated comparisons, not manually verified negatives. Two train/validation duplicate images were quarantined before validation selection; the original saved split manifest was retained. Both positive and negative cohorts collapse to one connected dependency group, so valid grouped confidence intervals are unavailable. These are descriptive point estimates with limited independent evidence. Per-book-pair summaries use the same overall negative cohort.','',
        'The second supplied case remains a partial result. Recovering complete words without wrong links needs better candidate evidence and correspondence supervision, such as annotated fragment-to-fragment links and cross-book hard negatives, followed by validation on new pages. Merely increasing the allowed false-positive rate cannot guarantee those links.','',
        '[All metrics and provenance](evaluation/final_fragments_20261109/metrics.json) · [Final settings](evaluation/final_fragments_20261109/settings.json) · [Every scored pair](evaluation/final_fragments_20261109/per_pair.csv) · [Validation search](evaluation/final_fragments_20261109/validation_presets.csv)','',
        '## Reproduce','',
        'Run `python scripts/eval/final_fragments.py --device auto --workers 4`, then `python scripts/eval/render_final_fragments.py --update-notebooks`. Cached image features and complete image-only candidate pools are checked against the checkpoint and generation settings; metric values are recomputed. Reopen notebooks from disk to load edited sources. A kernel restart alone does not reload cells.']
    (ROOT/'docs/final_fragments_evaluation.md').write_text('\n'.join(lines)+'\n')
    return report


def update_settings(notebook, selected):
    aliases = dict(threshold='SIMILARITY_THRESHOLD', max_gap='MAX_INTERNAL_GAP', max_candidates='MAX_ALIGNMENT_CANDIDATES')
    assignments = {aliases.get(key,key.upper()):value for key,value in selected['settings'].items()}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if source.startswith('# EVALUATION-ONLY MATCHING SETTINGS.'):
            lines = source.splitlines(keepends=True)
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                    name=node.targets[0].id
                    if name in assignments:
                        suffix=lines[node.end_lineno-1].split('#',1)
                        comment='  # '+suffix[1].strip() if len(suffix)>1 else ''
                        lines[node.lineno-1:node.end_lineno]=[f'{name} = {assignments[name]!r}{comment}\n']
            source=''.join(lines)
            source=source.replace('# Validation IoU 70.73%, presumed-negative FPR 4/98. Held-out FPR is 8/100.',
                '# Final validation-selected preset: '+selected['name']+'.')
        source=source.replace('Active evaluator configuration (initial, uncalibrated):','Active evaluator configuration (final validation-selected settings):')
        cell['source']=source.splitlines(keepends=True)


def embed(out, report):
    metrics=report['final']['metrics'];val=report['selected']['validation_metrics']
    heading='\n\n'.join(['# Final results — optional-space checkpoint',
        '**Weights:** `real_abc_balanced_optional_space_20261108/checkpoint_best.pt`',
        f"**{metrics['positive_pairs']} saved-test positives + {metrics['negative_pairs']} presumed negatives.** Both supplied cases and shared source images are excluded. Figures below show only final predictions.",
        metric_table(metrics),
        f"**Test FPR {'exceeds' if metrics['negative_pair_fpr']>report['selected']['max_validation_negative_pair_fpr'] else 'meets'} the requested {report['selected']['max_validation_negative_pair_fpr']:.0%} budget.**",
        f"False detections: **{metrics['negative_false_positive_pairs']}/{metrics['negative_pairs']}**. Validation selection used your **10% false-positive budget** and maximized foreground recall; validation FPR was {val['negative_false_positive_pairs']}/{val['negative_pairs']}. Test scores did not select the settings.",
        'Separate passages can change order without reusing windows. Short passages are checked separately; spacing may make the same word cover different numbers of windows. Optional-space DTW follows the saved training settings.',
        '\n'.join(f"- `{case['pair_id']}`: **{len(case['accepted_regions'])} regions**, mask IoU **{case['metrics']['pair_mean_iou']:.2%}**." for case in report['cases']),
        '**The second case is still partially recovered and includes an ambiguous/wrong link. Region count is not proof of correct word correspondence.** The first case exposes a two-window support rejection; the report explains both cases.',
        '[Final report](../docs/final_fragments_evaluation.md) · [All metric values](../docs/evaluation/final_fragments_20261109/metrics.json) · [Per-pair scores](../docs/evaluation/final_fragments_20261109/per_pair.csv)',
        '**Reopen this notebook from disk, then restart the kernel to run the updated settings.** Earlier saved outputs are archived locally under `Results/Evaluation/final_fragments_20261109/notebooks_before/`.'])
    def figure(case):
        filename=case['image'];relative=str((out/filename).relative_to(ROOT))
        source=('from pathlib import Path\nfrom IPython.display import Image, display\n'
            "_final_root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'evaluation_utils.py').is_file())\n"
            f"_final_image = _final_root / {relative!r}\n"
            "if _final_image.is_file():\n    display(Image(filename=str(_final_image)))\n"
            "else:\n    print('Use the embedded final figure, or regenerate with scripts/eval/render_final_fragments.py.')\n")
        return dict(cell_type='code',id='final-fragments-'+case['pair_id'].split('__')[0],metadata={},execution_count=None,
            source=source.splitlines(keepends=True),outputs=[dict(output_type='display_data',metadata={},
            data={'image/png':base64.b64encode((out/filename).read_bytes()).decode(),'text/plain':[case['pair_id']]})])
    for name in ('model_evaluation','book_pairs_qualitative'):
        path=ROOT/'notebooks'/f'{name}.ipynb';notebook=json.loads(path.read_text())
        notebook['cells']=[c for c in notebook['cells'] if not c.get('id','').startswith(('reordered-update-','final-fragments-'))]
        update_settings(notebook,report['selected'])
        if name=='model_evaluation':
            for cell in notebook['cells']:
                if cell['cell_type']=='code' and not cell.get('id','').startswith('book-pair-review-'):
                    cell['outputs']=[];cell['execution_count']=None
        notebook['cells'].insert(0,dict(cell_type='markdown',id='final-fragments-intro',metadata={},source=heading.splitlines(keepends=True)))
        index=next(i for i,c in enumerate(notebook['cells']) if c['cell_type']=='code')+1
        notebook['cells'][index:index]=[figure(case) for case in report['cases']]
        notebook['metadata'].pop('reordered_evaluation',None)
        notebook['metadata']['final_evaluation']=dict(checkpoint_sha256=report['selected']['checkpoint_sha256'],
            preset=report['selected']['name'],report='docs/final_fragments_evaluation.md',metrics=metrics)
        path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
        print('Updated',path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'Results/Evaluation/final_fragments_20261109')
    parser.add_argument('--update-notebooks',action='store_true')
    args=parser.parse_args()
    report=render(args.output)
    if args.update_notebooks:embed(args.output,report)

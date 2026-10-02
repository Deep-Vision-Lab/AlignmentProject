"""Reproduce the latest problematic notebook pairs with frozen-checkpoint inference.

Run from repository root. No optimizer, training or threshold search. An optional
before-source snapshot can compare the prior evaluator on identical features.
"""
import argparse
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib.pyplot as plt
import numpy as np

from evaluate import local_repeat_regions,region_mask,MATCH_VERSION
from evaluation_utils import (EvaluationSession,plot_pair_alignment,evaluate_population,
                              predict_cached_pair,compute_pair_metrics,print_candidate_diagnostics)


def refresh_notebook(output,notebook_path=ROOT/'notebooks/model_evaluation.ipynb'):
    """Publish real review artifacts as clearly attributed saved notebook outputs.

    This does not claim Run All execution: execution counts stay None and the
    output provenance records the CPU inference report and fixed reviewed IDs.
    It removes stale outputs produced by the retired evaluator.
    """
    output=Path(output);report=json.loads((output/'report.json').read_text())
    notebook=json.loads(Path(notebook_path).read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            cell.update(outputs=[],execution_count=None)
            cell['metadata'].pop('historical_output',None)
    def stream(value):return dict(output_type='stream',name='stdout',text=value.splitlines(keepends=True))
    def saved_result(example):
        name=example['name'];metrics=example['metrics']
        candidates=json.loads((output/(name+'_candidates.json')).read_text())
        matrix=np.load(output/(name+'_matrices.npz'))['cosine']
        result=dict(pair=metrics,metrics=metrics,cosine=matrix,
            ground_truth=[True if (output/(name+f'_gt_{s}.png')).exists() else None for s in (0,1)],
            match=dict(regions=[c for c in candidates if c['accepted']],
                       rejected=[c for c in candidates if not c['accepted']]))
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print('Frozen-checkpoint CPU review:',example['sample_id'])
            print_candidate_diagnostics(result,5)
            print('Localization IoU A/B:',metrics['a_iou'],metrics['b_iou'])
            print('GT provenance:',metrics['ground_truth_provenance'])
        return [stream(buffer.getvalue()),dict(output_type='display_data',metadata={},
            data={'image/png':base64.b64encode((output/(name+'.png')).read_bytes()).decode(),
                  'text/plain':['Frozen-checkpoint image-only alignment prediction; GT masks hidden.']})]
    for slot,example in enumerate(report['examples'][:4],1):
        notebook['cells'][11+2*slot]['outputs']=saved_result(example)
    notebook['cells'][21]['outputs']=[stream('No unused unique positive sample remains; no duplicate will be displayed.\n')]
    notebook['cells'][9]['outputs']=[stream('Requested 5 unique positive samples, but only 4 eligible unique samples are available.\n')]
    notebook['cells'][33]['outputs']=[stream('Five distinct problematic manifest-negative examples reviewed.\n')]
    for example in report['examples'][4:]:notebook['cells'][33]['outputs']+=saved_result(example)
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ns={'session':types.SimpleNamespace()}
        exec(''.join(notebook['cells'][7]['source']),ns)
    assert ns['session'].settings==report['settings']
    notebook['cells'][7]['outputs']=[stream(buffer.getvalue())]
    note=('\n\nSaved review outputs below come from frozen-checkpoint CPU inference on the four distinct '
          'latest problematic positives and five distinct manifest negatives. They are imported from '
          '`Results/Evaluation/path_quality_review_v3/report.json`; this is not a full Run All execution. '
          'Execution counts are cleared. Rerunning cells draws from the live deduplicated pool. '
          'All saved plots use image-only predictions and hide GT masks.\n')
    source=''.join(notebook['cells'][0]['source'])
    source=source.split('\n\nSaved review outputs below come from')[0]+note
    notebook['cells'][0]['source']=source.splitlines(keepends=True)
    notebook['metadata']['path_quality_review']=dict(report=str(output/'report.json'),
        checkpoint_sha256=report['checkpoint_sha256'],device='cpu',
        sample_ids=[e['sample_id'] for e in report['examples']],execution='imported real review artifacts')
    Path(notebook_path).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='Weights/real_sum_nogate_hybridneg/checkpoint_best.pt')
    parser.add_argument('--dataset',default='DataSet/ArabicDataset')
    parser.add_argument('--output',default='Results/Evaluation/path_quality_review_v3')
    parser.add_argument('--before-source')
    parser.add_argument('--refresh-notebook',action='store_true')
    args=parser.parse_args()
    print('Loading frozen checkpoint and saved validation membership...',flush=True)
    session=EvaluationSession(args.checkpoint,args.dataset,split='val',device='cpu',construct_negatives=False)
    notebook=json.loads((ROOT/'notebooks/model_evaluation.ipynb').read_text())
    ns={'session':session}
    cell=next(c for c in notebook['cells'] if ''.join(c['source']).startswith('# EVALUATION-ONLY MATCHING'))
    exec(''.join(cell['source']),ns)
    before=None
    preserved_before=Path(args.output)/'before_evaluate_v2.py'
    before_source=args.before_source or (preserved_before if preserved_before.is_file() else None)
    if before_source:
        before=types.ModuleType('before_evaluation')
        exec(Path(before_source).read_text(),before.__dict__)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    requested=[('positive_1','pair_000028:1580'),('positive_2','pair_000028:3638'),
        ('positive_3','pair_000028:455'),('positive_4','pair_000028:1989'),
        ('unaligned_1','pair_000028:981'),('unaligned_2','pair_000028:1887'),
        ('negative_shared_word','pair_000028:1487'),('unaligned_4','pair_000028:2496'),
        ('unaligned_5','pair_000028:2437')]
    report=dict(checkpoint=str(session.checkpoint),checkpoint_sha256=session.checkpoint_sha256,
                implementation_version=MATCH_VERSION,split='val',device='cpu',settings=session.settings,examples=[])
    for name,sample_id in requested:
        pair=next((p for p in session.pairs if p['sample_id']==sample_id),None)
        if pair is None:
            report['examples'].append(dict(name=name,sample_id=sample_id,status='unavailable'));continue
        result=session.evaluate_pair(pair)
        c=result['cosine'];lines=result['lines']
        cached=predict_cached_pair(session,pair,lines,settings=session.settings,cosine=c)
        assert cached['match']['candidates']==result['match']['candidates']
        assert all(np.array_equal(a,b) for a,b in zip(cached['masks'],result['masks']))
        prior=None
        if before:
            prior=before.local_repeat_regions(c,*(r['physical'] for r in lines),threshold=.5,
                min_windows=5,min_distinct_windows_a=5,min_distinct_windows_b=5,min_matched_pairs=5,
                use_strong_start_anchor=True,start_top_k=3)
        summary_keys=('candidate_id','candidate_type','support','matched_pairs','score','score_per_match',
            'mean_cosine','path_density','mutual_anchor_count','mutual_anchor_fraction',
            'mean_bidirectional_margin','median_bidirectional_margin','ambiguous',
            'letter_evidence_mean','letter_evidence_positive_fraction','reason','logical_ranges')
        row=dict(name=name,sample_id=sample_id,shape=list(c.shape),metrics=result['metrics'],
            before=[dict(score=r['score'],support=r['support'],logical_ranges=r['logical_ranges']) for r in prior['regions']] if prior else None,
            after=[{k:r[k] for k in summary_keys} for r in result['match']['regions']],
            candidate_count=len(result['match']['candidates']),candidate_limit_reached=result['match']['candidate_limit_reached'])
        if prior:
            prior_result=dict(result,match=prior,
                masks=[region_mask(prior['regions'],side,lines[side]['geometry'],
                                   session.config.window_width,session.config.window_stride) for side in (0,1)])
            row['before_metrics'],_=compute_pair_metrics(prior_result,session.config,
                preloaded_ground_truth=result['ground_truth'])
        # Match old diagnostic candidates to the new pool by their pair identity.
        if prior:
            important=sorted(prior['rejected'],key=lambda r:r.get('score',0),reverse=True)[:5]
            row['previously_rejected']=[dict(old_support=r.get('support'),old_score=r.get('score'),old_reason=r['reason'],
                old_pairs=r.get('pairs'),current=[{k:new[k] for k in summary_keys} for new in result['match']['candidates']
                    if r.get('pairs') and set(r['pairs'])<=set(new['pairs'])]) for r in important]
        # An ablation is evidence about the anchor filter, not a tuned deployment:
        # keep all scores/thresholds fixed, disable just the path-anchor check.
        ablation=predict_cached_pair(session,pair,lines,settings=dict(session.settings,use_path_anchor_check=False),cosine=c)
        row['path_anchor_check_disabled_ablation']=[{k:r[k] for k in summary_keys} for r in ablation['match']['regions']]
        report['examples'].append(row)
        prefix=output/name
        np.savez(output/(name+'_matrices.npz'),cosine=c,rewards=result['match']['rewards'],
            letter_evidence=result['match'].get('letter_evidence'),physical_a=lines[0]['physical'],physical_b=lines[1]['physical'])
        (output/(name+'_candidates.json')).write_text(json.dumps(result['match']['candidates'],indent=2,allow_nan=False))
        fig=plot_pair_alignment(session,result,show_gt=False);fig.savefig(output/(name+'.png'),dpi=80);plt.close(fig)
        from PIL import Image
        for side in (0,1):Image.fromarray(result['masks'][side]).save(output/(name+f'_mask_{side}.png'))
        # GT overlay is evaluation-only and separate from the default prediction figure.
        for side,gt in enumerate(result['ground_truth']):
            if gt is not None:Image.fromarray(np.asarray(gt,np.uint8)*255).save(output/(name+f'_gt_{side}.png'))
        print(json.dumps(dict(name=name,category=result['metrics']['evaluation_category'],
            before=row['before'],after=row['after'],limited=row['candidate_limit_reached'])),flush=True)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    chosen={sample_id for _,sample_id in requested}
    session.pairs=[p for p in session.pairs if p['sample_id'] in chosen]
    from evaluation_utils import aggregate_summary
    population=evaluate_population(session)
    report['population_summary']=aggregate_summary(population)
    report['population_rows']=population['rows']
    by_id={r['sample_id']:r for r in population['rows']}
    for example in report['examples']:
        if example.get('status')!='unavailable':
            assert by_id[example['sample_id']]==example['metrics'], 'Population/session mismatch'
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved',output/'report.json',flush=True)
    if args.refresh_notebook:refresh_notebook(output)


if __name__=='__main__':main()

"""Render the measured reordered-passage report and genuine before/after examples."""
from pathlib import Path
import sys,json,csv,gzip,pickle,base64
from collections import Counter
from inspect import cleandoc
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from evaluation_utils import plot_pair_alignment
from evaluate import local_repeat_regions
from scripts.eval.tune_matching import write_json,write_csv


def ensure_example_artifacts(output, report):
    """Recreate the development example from images and frozen settings if needed."""
    before=output/'user_example_before.pkl.gz';after=output/'user_example_selected.pkl.gz'
    if before.exists() and after.exists():
        with gzip.open(after,'rb') as stream:cached=pickle.load(stream)
        if cached['settings']==report['selected']['settings']:
            return
    import torch
    from evaluation_utils import EvaluationSession,compute_pair_metrics
    from scripts.eval.select_book_pair_examples import predict_candidate
    from PIL import Image
    from train import resolve_device
    provenance=torch.load(output/'val_prepared.pt',map_location='cpu',weights_only=False)['provenance']
    checkpoint=Path(provenance['checkpoint'])
    if not checkpoint.exists():checkpoint=ROOT/'Weights'/checkpoint.parent.name/checkpoint.name
    dataset=ROOT/'DataSet/ArabicData'
    name=report.get('development_pair','pair_000028__A_line_005__B_line_002')
    row=next(json.loads(line) for line in (dataset/'alignedPairs/A_B/pairs.jsonl').read_text().splitlines()
             if json.loads(line)['pair_id']==name)
    row['book_pair']='A_B'
    session=EvaluationSession(checkpoint,dataset,device=str(resolve_device('auto')),
        pair_source='folders',negative_count=0,crop_override='content')
    truth=[np.asarray(Image.open(dataset/row[book]['mask_path']).convert('L'))>0 for book in ('A','B')]
    for settings,path,is_before in [(provenance['baseline_settings'],before,True),
                                   (report['selected']['settings'],after,False)]:
        session.settings=settings
        result=predict_candidate(session,dataset,row,'fused')
        result['metrics'],result['ground_truth']=compute_pair_metrics(result,session.config,preloaded_ground_truth=truth)
        payload=dict(result=result,row=row,settings=settings,config=session.config) if is_before else result
        with gzip.open(path,'wb') as stream:pickle.dump(payload,stream)


def ensure_real_reordered_artifacts(output, report):
    path=output/'real_reordered_example.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('settings')==report['selected']['settings']:
            return saved
    from PIL import Image
    from evaluation_utils import EvaluationSession,revalidate_cached_candidates,compute_pair_metrics
    from scripts.eval.select_book_pair_examples import predict_candidate
    from train import resolve_device
    dataset=ROOT/'DataSet/ArabicData';name='pair_000001__A_line_011__B_line_005'
    row=next(json.loads(line) for line in (dataset/'alignedPairs/A_B/pairs.jsonl').read_text().splitlines()
             if json.loads(line)['pair_id']==name)
    row['book_pair']='A_B'
    checkpoint=ROOT/'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt'
    session=EvaluationSession(checkpoint,dataset,device=str(resolve_device('auto')),
        pair_source='folders',negative_count=0,crop_override='content')
    session.settings=report['selected']['settings']
    after=predict_candidate(session,dataset,row,'fused')
    before=revalidate_cached_candidates(session,after,dict(session.settings,allow_reordered_regions=False))
    truth=[np.asarray(Image.open(dataset/row[book]['mask_path']).convert('L'))>0 for book in ('A','B')]
    for label,result in [('before',before),('after',after)]:
        result['metrics'],result['ground_truth']=compute_pair_metrics(result,session.config,preloaded_ground_truth=truth)
        fig=plot_pair_alignment(session,result,show_gt=True)
        fig.suptitle(f'REAL REORDERED PASSAGES | {name} | {label}\nTraining-pair illustration; parameters frozen; not a held-out score',fontsize=11)
        fig.savefig(output/f'real_reordered_{label}.png',dpi=90);plt.close(fig)
    example=dict(pair_id=name,book_pair='A_B',annotated_word_order=dict(A=['الذين','قولا','غير'],B=['قولا','غير','الذين']),
        before_regions=len(before['match']['regions']),after_regions=len(after['match']['regions']),
        before_iou=before['metrics']['pair_mean_iou'],after_iou=after['metrics']['pair_mean_iou'],
        settings=session.settings,saved_line_splits={book:next((split for split,ids in session.evaluation_split_ids.items()
            if 'Book'+book+':'+row[book]['line_id'] in ids),'unseen') for book in ('A','B')},
        interpretation='Fixed training-pair illustration identified after parameters were frozen; not an extra held-out estimate')
    write_json(path,example)
    return example


def render(output):
    root=ROOT;out=Path(output)
    report=json.loads((out/'comparison.json').read_text());selected=report['selected'];methods=report['test_methods']
    ensure_example_artifacts(out,report)
    report['real_reordered_example']=ensure_real_reordered_artifacts(out,report)
    versioned=root/'docs/evaluation/reordered_regions_20261109';versioned.mkdir(parents=True,exist_ok=True)
    exposure=Counter(('positive' if r['category']=='positive' else 'negative',r['includes_reference_training_line']) for r in report['reference_membership'])
    report['reference_exposure_counts']={f'{category}_{"includes_training" if trained else "no_recorded_training"}':value for (category,trained),value in exposure.items()}
    report['limitations']=[
     'Negative comparisons are presumed unrelated, not manually verified; 100 were sampled with seed 42.',
     'The user-supplied development example and comparisons sharing its source lines are excluded from headline test metrics.',
     'The reference and current checkpoints have zero common saved-test positive pairs; old-weight fixed-cohort scores include reference-training exposure.',
     'Foreground masks measure localization, not word-to-word correspondence correctness. Reversed-order correctness is also covered by a separate synthetic regression.',
     'Validation selection met a 5% presumed-negative FPR limit, but this did not transfer: final test FPR is 8%.',
     'The provided three-word example remains only partially recovered; no claim is made that all missed matches are solved.'
    ]
    write_json(versioned/'metrics.json',report)
    summary=[];pairs=[]
    for method in methods:
     m=method['metrics'];summary.append(dict(method=method['name'],**{k:v for k,v in m.items() if not isinstance(v,dict)}))
     with (Path(method['directory'])/'per_pair.csv').open() as stream:
      pairs.extend(dict(method=method['name'],**row) for row in csv.DictReader(stream))
    write_csv(versioned/'summary.csv',summary);write_csv(versioned/'per_pair.csv',pairs)
    trials=json.loads((out/'validation_anchored/methods.json').read_text())
    write_csv(versioned/'validation_presets.csv',[dict(preset=r['name'],**{k:v for k,v in r['metrics'].items() if not isinstance(v,dict)},settings=json.dumps(r['settings'],sort_keys=True)) for r in trials])
    history=[]
    for folder in ('validation_initial_tradeoffs','validation','validation_joint','validation_anchored'):
        path=out/folder/'methods.json'
        if path.exists():
            for row in json.loads(path.read_text()):
                history.append(dict(round=folder,preset=row['name'],metrics=row['metrics'],settings=row['settings']))
    write_json(versioned/'validation_search_history.json',history)
    # Genuine before/after figures on the supplied example, with supplied GT visible.
    x=pickle.load(gzip.open(out/'user_example_before.pkl.gz','rb'))
    after=pickle.load(gzip.open(out/'user_example_selected.pkl.gz','rb'))
    s=SimpleNamespace(config=x['config'],split='test')
    for label,result in [('before',x['result']),('selected',after)]:
     fig=plot_pair_alignment(s,result,show_gt=True)
     fig.suptitle(f'USER-SUPPLIED EXAMPLE | {label} | {x["row"]["pair_id"]}\nGround truth shown below predictions; partial recovery remains',fontsize=11)
     fig.savefig(out/f'user_example_{label}.png',dpi=90);plt.close(fig)
    # An explicit synthetic regression, not a model-score example.
    cos=np.full((18,18),-.4)
    for i in range(5):cos[i,i+12]=cos[i+12,i]=.95
    common=dict(threshold=.5,score_mode='raw',normal_min_distinct_a=3,normal_min_distinct_b=3,
     normal_min_matched_pairs=3,allow_short_regions=False,use_path_anchor_check=False,
     enable_path_extension=False,enable_region_merging=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,allow in zip(axes,[False,True]):
     match=local_repeat_regions(cos,np.arange(18),np.arange(18),allow_reordered_regions=allow,**common)
     ax.imshow(cos,cmap='coolwarm',vmin=-1,vmax=1)
     for region,color in zip(match['regions'],['gold','lime']):
      a,b=zip(*region['pairs']);ax.plot(b,a,'o-',color=color,ms=4,lw=2)
     ax.set(title=f'{"New disjoint passages" if allow else "Previous global order"}: {len(match["regions"])} accepted',
      xlabel='Line 2: passage B then A',ylabel='Line 1: passage A then B')
    fig.suptitle('Synthetic reversed-order regression — not measured model performance')
    fig.savefig(out/'reversed_order_regression.png',dpi=120);plt.close(fig)
    labels=['Previous weights*','Current weights','Reordering only','Validated fragments']
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].bar(labels,[100*r['metrics']['positive_iou'] for r in methods],color=['gray','steelblue','teal','darkorange'])
    axes[0].set(ylabel='Mean foreground IoU (%)',ylim=(0,100),title='Same 32 positive pairs')
    axes[1].bar(labels,[r['metrics']['negative_false_positive_pairs'] for r in methods],color=['gray','steelblue','teal','darkorange'])
    axes[1].set(ylabel='Pairs with any predicted match',title='False detections on 100 presumed negatives')
    for ax in axes:ax.tick_params(axis='x',rotation=15)
    fig.suptitle('* Previous-weight reference includes training exposure; comparison is descriptive')
    fig.savefig(out/'quantitative_comparison.png',dpi=120);plt.close(fig)
    # Standalone, versioned text report.
    lines=['# Optional-space weights and independent passage evaluation — 9 November 2026','',
     'Both notebooks use `real_abc_balanced_optional_space_20261108/checkpoint_best.pt`. The matcher now permits disjoint passages in different orders, proposes gap-free passage candidates, checks long-region visual confidence, and can require multiple short passages to corroborate a line match. Individual paths remain monotonic and window spans cannot be reused.','',
     '## Held-out localization benchmark','',
     '32 saved-test positive pairs and 100 seeded presumed-negative comparisons. The user-provided test example was excluded before these headline scores; saved split IDs were never moved. Two duplicated Book C validation images were quarantined from tuning (147 positives, 98 negatives remain).','',
     '| Method | IoU | Dice | Precision | Recall | Positive pairs detected | Presumed-negative false detections |',
     '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in methods:
     m=r['metrics'];lines.append(f"| {r['name']} | {m['positive_iou']:.2%} | {m['positive_dice']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['positive_detected_pairs']}/{m['positive_pairs']} | {m['negative_false_positive_pairs']}/{m['negative_pairs']} ({m['negative_pair_fpr']:.1%}) |")
    lines+=['','Higher recall costs more false detections: the validation-selected matcher improves held-out IoU by 11.60 percentage points and recall by 15.61 points, while presumed-negative FPR rises from 3% to 8%. No test parameters were changed after inspecting those scores. Reordering alone leaves these mask scores unchanged; the structural fix is verified separately by reversed-passage regression tests.','',
     f"The reference checkpoint has recorded training exposure in {sum(r['includes_reference_training_line'] for r in report['reference_membership'])}/132 evaluated pairs. It and the current checkpoint share zero saved-test positives. These same-cohort old/new weight scores are descriptive, not an independent causal claim about training improvements.",'',
     '## More metrics','', '| Method | Region F1 (IoU ≥ 0.5) | Negative masked area | Candidate-limit hits |','| --- | ---: | ---: | ---: |']
    for r in methods:
     m=r['metrics'];lines.append(f"| {r['name']} | {m['region_f1']['f1']:.2%} | {m['negative_masked_area_fraction']:.2%} | {m['candidate_limit_hits']}/132 |")
    lines+=['','[All metrics, confidence intervals, validation settings and split/exposure details](evaluation/reordered_regions_20261109/metrics.json). [Summary CSV](evaluation/reordered_regions_20261109/summary.csv), [per-pair scores](evaluation/reordered_regions_20261109/per_pair.csv), [validation presets](evaluation/reordered_regions_20261109/validation_presets.csv).','',
     '## Supplied example','',
     '`alignedPairs:pair_000028__A_line_005__B_line_002` contains three supplied matching fragments, “ابتغاء”, “الله”, and “من”, in the same order. Original rejection reasons: 556 insufficient-support candidates, 170 below path density, 2 insufficient anchors. The prior notebook required 9 windows per side, 12 matched cells, 95% density, and disabled short regions.','',
     f"The selected matcher returns {len(after['match']['regions'])} regions instead of zero, but mean mask IoU is only {after['metrics']['pair_mean_iou']:.2%}. This remains a weak, partially recovered case; the smaller fragment and exact correspondence need further improvement. Lower short-region thresholds produce more regions but also more false positives on validation, so they were not silently adopted.",'',
     '## Reproduction and notebook refresh','',
     'The quantitative section is embedded near the top of both notebooks, including real before/after figures and an explicitly labeled synthetic reversed-order example. Reload/reopen the notebook from disk before restarting the kernel; restarting a kernel alone does not reload edited cell source. Previously rendered outputs from the original matcher are archived locally under `Results/Evaluation/reordered_regions_20261109/notebooks_before/`.','',
     'Run `python scripts/eval/review_reordered.py --output Results/Evaluation/reordered_regions_20261109 --device auto --workers 4`. Prepared feature/candidate caches retain the frozen baseline. A fresh output directory rebuilds features while restoring the pre-change matcher from the versioned metric report when available. The `--development-pair` images are excluded from headline test scoring. Then run `python scripts/eval/render_reordered_report.py --output Results/Evaluation/reordered_regions_20261109 --update-notebooks` to embed the measured report and before/after figures. Full inference artifacts stay under the output directory.','',
     'Mask localization metrics do not verify the identity of every word-to-word correspondence. Confidence intervals account for available shared line/page dependencies; 100 presumed negatives do not establish a universal false-positive rate.']
    real=report['real_reordered_example']
    lines+=['','## Real reordered-passage illustration','',
        f"`{real['pair_id']}` has reordered annotated fragments: A contains “الذين … قولا غير”, while B contains “قولا غير … الذين”. Enforcing global order accepts {real['before_regions']} region; allowing disjoint reordered passages accepts {real['after_regions']}. Mask IoU changes from {real['before_iou']:.2%} to {real['after_iou']:.2%} with all other settings fixed.",'',
        'Both source lines are saved training members. This is an illustration of matcher behavior identified after parameter selection, not additional held-out model performance. The separate supplied three-word example above remains only partially recovered.']
    (root/'docs/reordered_regions_evaluation.md').write_text('\n'.join(lines)+'\n')
    print('Saved versioned metrics and report',versioned)


def embed_updates(output):
    root=ROOT;out=Path(output)
    s=json.loads((out/'comparison.json').read_text());methods=s['test_methods'];selected=s['selected'];rows=[methods[0],methods[1],methods[-1]]
    heading='''# Updated evaluation — 9 November 2026

    **Checkpoint:** `real_abc_balanced_optional_space_20261108/checkpoint_best.pt`<br>
    **Matcher:** independent, disjoint passages; gap fragments; visual-confidence floor; corroborated short regions.

    **Reload/reopen this notebook from disk first, then restart the kernel. A kernel restart alone does not reload edited cells.**

    The previous settings rejected your separated-word example: 556 candidates failed support, 170 failed density, and 2 failed anchors. Reversed-order passages also faced an explicit global ordering restriction. Each passage now keeps its internal order, while separate passages may swap order without reusing their window spans.

    ## Quantitative results on the same fixed cohort

    32 saved-test positives + 100 seeded presumed-negative comparisons. The inspected example was excluded from headline test scores. Two duplicated validation images were quarantined; settings were selected using 147 validation positives and 98 presumed negatives.

    | Method | Mask IoU | Dice | Precision | Recall | False detections |
    | --- | ---: | ---: | ---: | ---: | ---: |
    '''
    labels=['Previous weights, old settings*','Current weights, old settings','Current weights, validation-selected matcher']
    for label,r in zip(labels,rows):
     m=r['metrics'];heading+=f"| {label} | {m['positive_iou']:.2%} | {m['positive_dice']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['negative_false_positive_pairs']}/100 |\n"
    heading+='''
    **Trade-off:** IoU improves from **51.0% to 62.6%** and recall from **55.2% to 70.8%**, while presumed-negative false detections rise from **3/100 to 8/100**. The validation FPR was 4/98; the 5% validation limit did not transfer to test.

    *The previous-checkpoint reference includes recorded training exposure in 60/132 pairs. The checkpoints share no saved-test positive pairs, so this is a descriptive weight comparison.*

    [Full report and all metrics](../docs/reordered_regions_evaluation.md) · [Metric JSON](../docs/evaluation/reordered_regions_20261109/metrics.json) · [Per-pair scores](../docs/evaluation/reordered_regions_20261109/per_pair.csv)

    ## Your supplied example remains a partial result

    `alignedPairs:pair_000028__A_line_005__B_line_002` has three supplied common fragments: “ابتغاء”, “الله”, and “من”, in the same order. The selected matcher returns **two regions instead of zero**, but mean mask IoU is only **21.9%**. The third fragment and exact correspondence are not fully recovered. The before/after figures show supplied ground truth beside the predictions; they are not claims of complete recovery.

    The synthetic reversed-passage figure separately demonstrates the structural order fix. It is an algorithm regression, **not measured model performance**.
    '''
    def markdown(cid,text):return dict(cell_type='markdown',id=cid,metadata={},source=cleandoc(text).splitlines(keepends=True))
    def figure(cid,title,filename):
     code=f'''# {title}
    from pathlib import Path
    from IPython.display import Image, display
    _review_root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'evaluation_utils.py').is_file())
    _review_figure = _review_root / 'Results/Evaluation/reordered_regions_20261109/{filename}'
    if _review_figure.is_file():
        display(Image(filename=str(_review_figure)))
    else:
        print('The saved notebook contains this figure. To regenerate local artifacts, run scripts/eval/review_reordered.py followed by scripts/eval/render_reordered_report.py --update-notebooks.')
    '''
     return dict(cell_type='code',id=cid,metadata={},execution_count=None,source=cleandoc(code).splitlines(keepends=True),outputs=[dict(output_type='display_data',metadata={},data={'image/png':base64.b64encode((out/filename).read_bytes()).decode(),'text/plain':[title]})])
    figures=[figure('reordered-update-metrics','Measured mask localization and false detections','quantitative_comparison.png'),
     figure('reordered-update-before','Supplied example: previous settings, no accepted region','User_example_before.png'.lower()),
     figure('reordered-update-after','Supplied example: selected settings, partial recovery','user_example_selected.png'),
     figure('reordered-update-reversed','Synthetic reversed-passage regression','reversed_order_regression.png')]
    real_path=out/'real_reordered_example.json'
    if real_path.exists():
        real=json.loads(real_path.read_text())
        heading+=f"\n\n## Real reordered passages: separate training-pair illustration\n\n`{real['pair_id']}` changes from {real['before_regions']} accepted region to {real['after_regions']}, with mask IoU {real['before_iou']:.1%} → {real['after_iou']:.1%}. The annotated shared fragments occur in different orders. **Both source lines are training members**; this illustrates the structural fix and is not an extra held-out score. All matching settings were already frozen.\n"
        figures.extend([figure('reordered-update-real-before','Real reordered passages before the order fix (training-pair illustration)','real_reordered_before.png'),
                        figure('reordered-update-real-after','Real reordered passages after the order fix (training-pair illustration)','real_reordered_after.png')])
    for name in ('model_evaluation','book_pairs_qualitative'):
     p=root/'notebooks'/f'{name}.ipynb';n=json.loads(p.read_text())
     n['cells']=[c for c in n['cells'] if not c.get('id','').startswith('reordered-update-')]
     n['cells'].insert(0,markdown('reordered-update-intro',heading))
     # Keep the main setup as the first code cell for independent setup/test use.
     index=next(i for i,c in enumerate(n['cells']) if c['cell_type']=='code')+1
     n['cells'][index:index]=figures
     if name=='model_evaluation':
      for c in n['cells']:
       if c['cell_type']=='code' and not c.get('id','').startswith(('book-pair-review-','reordered-update-')):
        c['outputs']=[];c['execution_count']=None
     n['metadata']['reordered_evaluation']=dict(matcher_version=s['selected']['matcher_version'],checkpoint_sha256=selected['checkpoint_sha256'],selected_preset=selected['name'],report='docs/reordered_regions_evaluation.md',test_positive_pairs=32,test_presumed_negative_pairs=100)
     p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
     print('Updated',p)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'Results/Evaluation/reordered_regions_20261109')
    parser.add_argument('--update-notebooks',action='store_true',help='Embed the report and saved figures near the top of both notebooks')
    args=parser.parse_args()
    render(args.output)
    if args.update_notebooks:embed_updates(args.output)

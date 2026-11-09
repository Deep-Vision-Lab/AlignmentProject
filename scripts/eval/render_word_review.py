"""Embed the measured image-only word review, including explicit book exposure."""
from pathlib import Path
import argparse
import ast
import base64
import csv
import json
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from dataset import file_hash
from parameters import Config
from evaluation_utils import predict_cached_pair,compute_pair_metrics,plot_pair_alignment
from scripts.eval.review_reordered import cached_session
from scripts.eval.render_final_fragments import update_settings
from scripts.eval.tune_matching import write_json,write_csv

WORD_NAMES='word_vocabulary_path word_max_nll word_min_posterior word_min_posterior_short word_min_cosine word_boundary_weight word_lattice_temperature word_keep_long_regions word_visual_weight word_min_distinct_types word_min_total_characters'


def configure_main(selected,out):
    path=ROOT/'notebooks/model_evaluation.ipynb'
    backup=out/'notebooks_before';backup.mkdir(parents=True,exist_ok=True)
    for name in ('model_evaluation','book_pairs_qualitative'):
        source=ROOT/'notebooks'/f'{name}.ipynb'
        if not (backup/source.name).exists():shutil.copy2(source,backup/source.name)
    notebook=json.loads(path.read_text())
    for cell in notebook['cells']:
        source=''.join(cell['source'])
        if source.startswith('# MODEL/DATA SETUP.'):
            lines=source.splitlines(keepends=True)
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EVALUATION_CROP' for t in node.targets):
                    lines[node.lineno-1]='EVALUATION_CROP = "none"  # Preserve the full-line geometry used in training.\n'
            source=''.join(lines)
            source=source.replace('"train", "evaluate", "alignment_overlays"','"train", "word_alignment", "evaluate", "alignment_overlays"')
        if source.startswith('# EVALUATION-ONLY MATCHING SETTINGS.'):
            if 'WORD_VOCABULARY_PATH =' not in source:
                source+='\n# ---- Image-only whole-word verification --------------------------------\n'
                source+=''.join(f'{name.upper()} = {selected["settings"][name]!r}\n' for name in WORD_NAMES.split())
            source='\n'.join('# Final validation-selected preset: '+selected['name']+'.'
                if line.startswith('# Final validation-selected preset:') else line for line in source.split('\n'))
        if source.startswith('def current_match_settings('):
            if 'word_vocabulary_path' not in source:
                source=source.replace('    adaptive =',f'    names += {WORD_NAMES!r}.split()\n    adaptive =',1)
                source=source.replace('_groups = (','_groups = (\n    ("Whole-word verification", '+repr(WORD_NAMES)+'),',1)
            source=source.replace('"dtw", "alignment_candidates",','"dtw", "word_alignment", "alignment_candidates",')
        if cell.get('id')=='margin-crop-intro':
            source=('### Preprocessing follows the trained image geometry\n\n'
                '`EVALUATION_CROP = "none"` preserves the full exported line before resizing, as in training. '
                'Cropping margins changed the scale of C writing and reduced several cross-book alignments. '
                '`content` remains available as an explicitly reported preprocessing experiment.\n')
        cell['source']=source.splitlines(keepends=True)
    update_settings(notebook,selected)
    for cell in notebook['cells']:
        if cell['cell_type']=='code' and not cell.get('id','').startswith('book-pair-review-'):
            cell['outputs']=[];cell['execution_count']=None
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')


def examples(out,report):
    prepared=torch.load(out/'full_line/test_prepared.pt',map_location='cpu',weights_only=False)
    if prepared['provenance']['checkpoint_sha256']!=report['selected']['checkpoint_sha256']:
        raise ValueError('Example features belong to different weights')
    config=Config(**prepared['provenance']['config'])
    session=cached_session(ROOT/report['checkpoint'],config,'development')
    session.checkpoint_sha256=report['selected']['checkpoint_sha256']
    results=[]
    for name in report['development_pairs']:
        item=next(item for item in prepared['prepared'] if item['record']['sample_id'].split(':')[-1]==name)
        for side,record in zip(item['inference']['sides'],item['record']['images']):
            if file_hash(side['image'])!=record['sha256']:raise ValueError('Example image changed after feature preparation')
        pair=dict(item['inference'],label='user_supplied_development',target=1,constructed_negative=False,
            annotation_provenance='Supplied masks, used only after prediction')
        prediction=predict_cached_pair(session,pair,item['lines'],settings=report['selected']['settings'],cosine=item['cosine'])
        prediction['metrics'],prediction['ground_truth']=compute_pair_metrics(prediction,config,preloaded_ground_truth=item['truth'])
        fig=plot_pair_alignment(session,prediction,show_gt=True)
        fig.suptitle('FINAL IMAGE-ONLY WORD REVIEW | '+name+'\nDevelopment example; excluded from aggregate test metrics',fontsize=11)
        filename=name+'_final.png';fig.savefig(out/filename,dpi=90);plt.close(fig)
        results.append(dict(pair_id=name,image=filename,metrics=prediction['metrics'],
            regions=[{key:c[key] for key in ('candidate_type','logical_ranges','support','mean_cosine','word',
                'word_nll','word_posteriors','word_pair_posterior','source_word_ranges') if key in c}
                for c in prediction['match']['regions']]))
    return results


def table(report):
    lines=['| Book pair | Positive pairs | Mask IoU | Precision | Recall |','| --- | ---: | ---: | ---: | ---: |']
    for group,m in report['final']['by_book_pair'].items():
        lines.append(f"| {group.replace('_','–')} | {m['positive_pairs']} | {m['positive_iou']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} |")
    return '\n'.join(lines)


def render(out):
    report=json.loads((out/'word_final.json').read_text());report['cases']=examples(out,report)
    saved=torch.load(ROOT/report['checkpoint'],map_location='cpu',weights_only=False)
    report['training_audit'].update(negative_loss_type=saved['config']['negative_loss_type'],
        hybrid_weights={key:saved['config'][key] for key in ('strong_negative_weight','wrong_letter_weight','order_negative_weight')},
        recorded_hybrid_losses={key:saved['metrics']['train'][key] for key in ('strong_neg_loss','local_wrong_letter_loss','order_loss')})
    versioned=ROOT/'docs/evaluation/word_alignment_20261109';versioned.mkdir(parents=True,exist_ok=True)
    write_json(versioned/'metrics.json',report);write_json(versioned/'settings.json',report['selected']['settings'])
    write_csv(versioned/'summary.csv',[dict(method='Image-only word verification',**{
        key:value for key,value in report['final']['metrics'].items() if not isinstance(value,dict)})])
    with (Path(report['final']['directory'])/'per_pair.csv').open() as stream:
        write_csv(versioned/'per_pair.csv',list(csv.DictReader(stream)))
    trials=json.loads((out/'word_validation/methods.json').read_text())
    write_csv(versioned/'validation.csv',[dict(preset=t['name'],book_f05=t['book_f05'],
        development_regression_passed=t['development_regression_passed'],
        reordered_development_regression_passed=t['reordered_development_regression_passed'],
        **{k:v for k,v in t['metrics'].items() if not isinstance(v,dict)}) for t in trials])
    m=report['final']['metrics'];budget=report['selected']['max_validation_negative_pair_fpr']
    status='meets' if m['negative_pair_fpr']<=budget else 'exceeds'
    lines=['# Image-only word verification and A/B/C evaluation — 9 November 2026','',
        f"Weights: `{report['checkpoint']}`; full-line preprocessing (`crop=none`); decoder `word_dtw`.",'',
        f"All three books are recorded in training: "+', '.join(f'{b}={n:,} lines' for b,n in report['book_training_counts'].items())+'. '
        'A figure’s line split describes that individual image. A line outside the balanced saved subset does not imply that its book was absent from training. The refreshed figures show book exposure and line membership separately.','',
        'The decoder preserves strong long visual passages, including words absent from its fixed vocabulary. Short regions require a whole-word hypothesis on each image with matching normalized identity. A full-line lattice makes longer words compete with their internal letter fragments. Every proposed word also undergoes a full-span visual DTW check; masks are clipped to its inferred edges. Without a strong long passage, multiple distinct words and sufficient combined support must corroborate the line match; an isolated common word does not suffice. Separate passages may appear in different orders without reusing either span.','',
        'Inference reads image pixels, frozen model features and a fixed vocabulary built from saved training IDs only. It does not read the evaluated pair’s transcriptions, masks or annotated boxes. The vocabulary is an additional domain prior, not a change to the supplied neural weights. It is built from current local training-split files; their historical equivalence to the original HPC text snapshot is unverified.','',
        '## Final saved-test point estimates','',table(report),'',
        f"Overall: IoU **{m['positive_iou']:.2%}**, Dice **{m['positive_dice']:.2%}**, precision **{m['precision']:.2%}**, recall **{m['recall']:.2%}**. "
        f"Detected positive pairs: {m['positive_detected_pairs']}/{m['positive_pairs']}. Presumed-negative false detections: **{m['negative_false_positive_pairs']}/{m['negative_pairs']} ({m['negative_pair_fpr']:.2%})**. "
        f"Test FPR **{status}** the requested {budget:.0%} budget. Negative masked area: {m['negative_masked_area_fraction']:.2%}.",'',
        f"Validation selection gives equal weight to A–B, A–C and B–C and maximizes mean foreground F0.5, which weights precision more than recall, under the {budget:.0%} presumed-negative FPR budget. "
        'Both supplied development cases are regression requirements: the three-word case must recover its three word hypotheses without extra regions; the reordered case must retain its long passage and short word without an extra region. Settings were frozen before the final test run. Both supplied examples and all test comparisons sharing their source images are excluded. This saved-test cohort has been examined in previous aggregate reports and is not a new independent benchmark.','',
        '## Supplied cases','']
    for case in report['cases']:
        words=[r['word'] for r in case['regions'] if r['candidate_type']=='word']
        passages=sum(r['candidate_type']=='visual_passage' for r in case['regions'])
        lines.append(f"- `{case['pair_id']}`: IoU **{case['metrics']['pair_mean_iou']:.2%}**; normalized word hypotheses {', '.join('`'+w+'`' for w in words) or 'none'}; {passages} retained long visual passage(s).")
    lines+=['','Canonical word identities merge common letter-form variants and remove standalone hamza; they are hypotheses, not verified OCR transcriptions. In particular, `ابتغا` is the normalized template used when checking “ابتغاء”. Repeated occurrences of the same word remain ambiguous and localization may be partial.','',
        '## Training and spacing findings','',
        'The checkpoint uses optional spaces and was trained on A, B and C. Optional-space image-to-text DTW can consume a space across zero or several windows; it does not, by itself, certify whole-word image-to-image matches. '
        'The checkpoint uses active hybrid typed negatives: strong-negative weight 0.5, wrong-letter weight 0.5 and order-negative weight 0.25. Their recorded losses are nonzero. `negative_dtw_weight=0` applies to the separate legacy term and does not disable those hybrid branches. These losses train image-to-transcript discrimination; they do not provide direct supervision for every complete-word image-pair boundary.','',
        'Current local C text files are concatenated. The user confirmed that the images are the same on Drive and the HPC; that does not establish the literal whitespace in the original training text files. Original training-text formatting is left unverified. The current word prior therefore receives no spaced-word tokens from local C files. This limits vocabulary coverage; it does not mean C was absent from neural training.','',
        '## Limits and reproducibility','',
        'The vocabulary can omit rare words and can contain imperfect tokens inherited from source transcripts. Long visual passages preserve some unknown-word coverage. Word boundaries and word identities remain predictions. Supplied foreground masks score localization, not correctness of every word correspondence. The region interval F1 in the metric JSON is not word-recognition F1.','',
        'Negatives remain presumed unrelated comparisons. Shared short words or incomplete source transcriptions can complicate their interpretation. Available line/page dependencies collapse the cohorts to one connected group, so grouped confidence intervals are unavailable. A–C and B–C test counts are particularly small.','',
        '[All metric values and provenance](evaluation/word_alignment_20261109/metrics.json) · [Final settings](evaluation/word_alignment_20261109/settings.json) · [Every scored pair](evaluation/word_alignment_20261109/per_pair.csv) · [Validation presets](evaluation/word_alignment_20261109/validation.csv)','',
        'Run `python scripts/eval/review_words.py`, then `python scripts/eval/render_word_review.py --update-notebooks`. To refresh the fixed gallery, run `scripts/eval/review_book_pairs.py` with `--settings-report Results/Evaluation/word_alignment_20261109/word_selected.json --crop none` and the saved fixed selection. Reopen the notebooks from disk to load edited cell sources.']
    (ROOT/'docs/word_alignment_evaluation.md').write_text('\n'.join(lines)+'\n')
    return report


def embed(out,report):
    m=report['final']['metrics'];budget=report['selected']['max_validation_negative_pair_fpr']
    heading='\n\n'.join(['# Final image-only word verification — A, B and C',
        '**Weights:** `real_abc_balanced_optional_space_20261108`. **All three books were trained:** A=1,107, B=1,107, C=1,107 lines. Figure captions distinguish each line’s membership from its book’s training exposure.',
        'Strong long visual passages are retained. Short masks require matching whole-word hypotheses, full-line context and a full-span visual check. Predictions use images and a fixed training-only vocabulary; supplied line transcripts are not inference inputs.',
        table(report),
        f"Overall IoU **{m['positive_iou']:.2%}**, Dice **{m['positive_dice']:.2%}**, precision **{m['precision']:.2%}**, recall **{m['recall']:.2%}**. False detections: **{m['negative_false_positive_pairs']}/{m['negative_pairs']}**. Test FPR {'meets' if m['negative_pair_fpr']<=budget else 'exceeds'} the requested {budget:.0%} budget.",
        f"{m['positive_pairs']} saved-test positives + {m['negative_pairs']} presumed negatives; supplied cases and shared source images excluded. Validation selection weights the three book pairs equally and prioritizes foreground precision. Counts for C are small.",
        '\n'.join(f"- `{c['pair_id']}`: **{c['metrics']['pair_mean_iou']:.2%} IoU**." for c in report['cases']),
        '**Masks and word identities are predictions. Unknown vocabulary and repeated words remain limitations.** The local C text files are concatenated; original HPC training-text whitespace is unverified. This does not mean C was untrained.',
        '[Full current report](../docs/word_alignment_evaluation.md) · [All metrics](../docs/evaluation/word_alignment_20261109/metrics.json) · [Per-pair scores](../docs/evaluation/word_alignment_20261109/per_pair.csv)',
        '**Reopen the notebook from disk, then restart its kernel to load the edited configuration.**'])
    for name in ('model_evaluation','book_pairs_qualitative'):
        path=ROOT/'notebooks'/f'{name}.ipynb';n=json.loads(path.read_text())
        n['cells']=[c for c in n['cells'] if not c.get('id','').startswith(('final-fragments-','reordered-update-','word-review-'))]
        n['cells'].insert(0,dict(cell_type='markdown',id='word-review-intro',metadata={},source=heading.splitlines(keepends=True)))
        figures=[]
        for case in report['cases']:
            relative=str((out/case['image']).relative_to(ROOT));words=[r.get('word') for r in case['regions'] if r.get('word')]
            code=('from pathlib import Path\nfrom IPython.display import Image, display\n'
                "_word_root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'word_alignment.py').is_file())\n"
                f"_word_figure = _word_root / {relative!r}\n"
                f"print('Normalized word hypotheses:', {words!r})\n"
                "if _word_figure.is_file():\n    display(Image(filename=str(_word_figure)))\n"
                "else:\n    print('Use the embedded final figure or regenerate with scripts/eval/render_word_review.py.')\n")
            figures.append(dict(cell_type='code',id='word-review-'+case['pair_id'].split('__')[0],metadata={},execution_count=None,
                source=code.splitlines(keepends=True),outputs=[dict(output_type='display_data',metadata={},data={
                'image/png':base64.b64encode((out/case['image']).read_bytes()).decode(),'text/plain':[case['pair_id']]})]))
        index=next(i for i,c in enumerate(n['cells']) if c['cell_type']=='code')+1
        n['cells'][index:index]=figures
        n['metadata'].pop('final_evaluation',None);n['metadata'].pop('reordered_evaluation',None)
        n['metadata']['word_evaluation']=dict(report='docs/word_alignment_evaluation.md',selected=report['selected'],metrics=m)
        path.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
        print('Updated',path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'Results/Evaluation/word_alignment_20261109')
    parser.add_argument('--update-notebooks',action='store_true')
    args=parser.parse_args();report=render(args.output)
    if args.update_notebooks:
        configure_main(report['selected'],args.output)
        embed(args.output,report)

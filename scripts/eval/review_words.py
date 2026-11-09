"""Validate image-only whole-word masks separately for A–B, A–C and B–C."""
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
import csv
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataset import file_hash
from evaluate import MATCH_VERSION,resolved_match_settings
from parameters import Config
from scripts.eval import mask_quantitative as q
from scripts.eval.build_word_vocabulary import build
from scripts.eval.final_fragments import DEVELOPMENT_PAIRS,exclude_reviewed
from scripts.eval.review_reordered import run_methods,cached_session
from evaluation_utils import predict_cached_pair,revalidate_cached_candidates
from scripts.eval.tune_matching import write_json,write_csv
from scripts.eval.review_book_pairs import training_book_counts


def presets(baseline,vocabulary):
    methods=[]
    for nll in (1.,1.4,1.8):
        for posterior in (.03,.06,.075,.10):
            for short in (.15,.25):
                for cosine in (.4,.5):
                    settings=resolved_match_settings(dict(baseline,alignment_mode='word_dtw',
                        similarity_mode='letter_evidence',candidate_similarity_mode='letter_evidence',
                        word_vocabulary_path=str(vocabulary.relative_to(ROOT)),
                        word_max_nll=nll,word_min_posterior=posterior,word_min_posterior_short=short,
                        word_min_cosine=cosine,word_boundary_weight=2.,word_lattice_temperature=1.,
                        enable_path_extension=False,enable_region_merging=False,allow_reordered_regions=True))
                    methods.append((f'word_nll{nll}_posterior{posterior}_short{short}_cos{cosine}',settings))
    return methods


def by_book(rows,prepared):
    result={}
    for group in ('A_B','A_C','B_C'):
        ids={r['record']['sample_id'] for r in prepared if r['record'].get('source_folder')==group}
        if ids:result[group]=q.aggregate([r for r in rows if r['sample_id'] in ids or r['category']=='negative'])
    return result


def book_objective(groups):
    values=[]
    for metrics in groups.values():
        p,r=metrics['precision'],metrics['recall']
        values.append(1.25*p*r/(.25*p+r) if p+r else 0.)
    return float(np.mean(values)) if values else 0.


def reuse_validation(output,checkpoint,prepared,methods,config,workers):
    """Reuse scored validation rows only when weights, matcher and cohort agree."""
    previous=json.loads((output/'word_final.json').read_text())
    if previous['selected']['checkpoint_sha256']!=file_hash(checkpoint):
        raise ValueError('Stored validation belongs to different weights')
    for name in ('word_alignment.py','evaluate.py','evaluation_utils.py'):
        if previous['source_sha256'][name]!=file_hash(ROOT/name):
            raise ValueError('Matcher changed; recompute validation')
    saved=json.loads((output/'word_validation/methods.json').read_text())
    lookup={r['name']:r for r in saved}
    for name,settings in methods:
        if name in lookup and lookup[name]['settings']!=settings:
            raise ValueError('Validation settings changed; recompute validation')
    expected={r['record']['sample_id'] for r in prepared}
    for result in saved:
        with (Path(result['directory'])/'per_pair.csv').open() as stream:rows=list(csv.DictReader(stream))
        if {r['sample_id'] for r in rows}!=expected:raise ValueError('Validation cohort changed')
        for row in rows:
            for key in ('positive_iou','positive_dice','precision','recall','negative_masked_area_fraction'):row[key]=float(row[key])
            for key in ('negative_false_positive','candidate_limit_hit','region_eligible','presumed_negative'):row[key]=row[key]=='True'
            for key in ('region_tp','region_fp','region_fn'):row[key]=int(row[key])
        for key,value in q.aggregate(rows).items():
            if value!=result['metrics'][key]:raise ValueError('Stored validation aggregate does not match its per-pair rows')
        result['rows']=rows
    missing=[(name,settings) for name,settings in methods if name not in lookup]
    if missing:
        destination=output/'word_validation_refinement';destination.mkdir(exist_ok=True)
        cache=destination/'candidate_cache'
        if not cache.exists():cache.symlink_to((output/'word_validation/candidate_cache').resolve(),target_is_directory=True)
        fresh=run_methods(checkpoint,config,'val',prepared,missing,destination,workers)
        lookup.update({r['name']:r for r in fresh})
    return [lookup[name] for name,_ in methods]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'Results/Evaluation/word_alignment_20261109')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    p.add_argument('--vocabulary',type=Path,default=ROOT/'docs/evaluation/word_alignment_20261109/vocabulary.json')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-validation-fpr',type=float,default=.10)
    p.add_argument('--reuse-validation',action='store_true',help='Verify and reuse existing validation predictions; recheck both development cases')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    vocabulary=build(a.checkpoint,ROOT/'DataSet/ArabicData',a.vocabulary)
    prepared=out/'full_line'
    from scripts.eval.tune_masks import parser
    arguments=parser().parse_args(['prepare','--checkpoint',str(a.checkpoint),'--crop','none',
        '--device','mps' if torch.backends.mps.is_available() else 'cpu','--threads','4',
        '--negative-count','100','--pairs-per-split','all','--class-balance','natural',
        '--quarantine-leaked-evaluation-images','--output',str(prepared)])
    for split in ('val','test'):
        path=prepared/(split+'_prepared.pt')
        if not path.exists():
            _,items,provenance=q.build_cache(arguments,split)
            torch.save(dict(prepared=items,provenance=provenance),path)
    validation=torch.load(prepared/'val_prepared.pt',map_location='cpu',weights_only=False)
    config=Config(**validation['provenance']['config'])
    if config.crop!='none' or validation['provenance']['checkpoint_sha256']!=file_hash(a.checkpoint):
        raise ValueError('Prepared features have another checkpoint or crop')
    catalog=ROOT/'DataSet/ArabicData/alignedPairs/A_B/pairs.jsonl'
    rows=[json.loads(line) for line in catalog.read_text().splitlines()]
    reviewed=[r for r in rows if r['pair_id'] in DEVELOPMENT_PAIRS]
    hashes={file_hash(ROOT/'DataSet/ArabicData'/r[b]['image_path']) for r in reviewed for b in ('A','B')}
    val,val_excluded=exclude_reviewed(validation['prepared'],hashes)
    baseline=validation['provenance']['baseline_settings']
    methods=presets(baseline,a.vocabulary)
    # This explicitly identified development case is excluded, with all shared
    # source images, from the final test cohort. Its intended word identities
    # are a regression requirement, never matcher inputs.
    development_name='pair_000028__A_line_005__B_line_002'
    development_words={'ابتغا','الله','من'}
    development_data=torch.load(prepared/'test_prepared.pt',map_location='cpu',weights_only=False)
    development_item=next(item for item in development_data['prepared']
        if item['record']['sample_id'].split(':')[-1]==development_name)
    development_session=cached_session(a.checkpoint,config,'development')
    development_session.checkpoint_sha256=file_hash(a.checkpoint)
    development_pool=predict_cached_pair(development_session,development_item['inference'],
        development_item['lines'],settings=methods[0][1],cosine=development_item['cosine'])
    first_item=next(item for item in development_data['prepared']
        if item['record']['sample_id'].split(':')[-1]==DEVELOPMENT_PAIRS[0])
    first_pool=predict_cached_pair(development_session,first_item['inference'],first_item['lines'],
        settings=methods[0][1],cosine=first_item['cosine'])
    write_json(out/'word_selection_protocol.json',dict(objective='Equal-book-pair mean foreground F0.5 (precision weighted), then mean IoU',
        max_validation_fpr=a.max_validation_fpr,presets=[dict(name=n,settings=s) for n,s in methods],
        validation_exclusions=val_excluded,development_pairs=list(DEVELOPMENT_PAIRS),
        development_regression=dict(pair_id=development_name,expected_normalized_words=sorted(development_words),
            requirement='Recover these three word hypotheses with no additional regions; excluded from headline test'),
        reordered_development_regression=dict(pair_id=DEVELOPMENT_PAIRS[0],
            requirement='Retain the long visual phrase and one الله word correspondence, without extra regions'),
        interpretation='Image-only inference with a fixed training-only word prior. Supplied pair transcriptions and boxes are not prediction inputs.'))
    torch.set_num_threads(1)
    trials=(reuse_validation(out,a.checkpoint,val,methods,config,a.workers) if a.reuse_validation else
            run_methods(a.checkpoint,config,'val',val,methods,out/'word_validation',a.workers))
    for trial in trials:
        trial['by_book_pair']=by_book(trial['rows'],val)
        trial['book_f05']=book_objective(trial['by_book_pair'])
        development=revalidate_cached_candidates(development_session,development_pool,trial['settings'])
        regions=development['match']['regions']
        trial['development_regression_passed']=(len(regions)==3 and
            {r.get('word') for r in regions}==development_words)
        first=revalidate_cached_candidates(development_session,first_pool,trial['settings'])['match']['regions']
        trial['reordered_development_regression_passed']=(len(first)==2 and
            sum(r['candidate_type']=='visual_passage' for r in first)==1 and
            [r.get('word') for r in first if r['candidate_type']=='word']==['الله'])
    eligible=[t for t in trials if t['metrics']['negative_pair_fpr']<=a.max_validation_fpr
              and t['development_regression_passed'] and t['reordered_development_regression_passed']]
    if not eligible:raise ValueError('No word preset meets the validation false-positive budget')
    selected=max(eligible,key=lambda t:(t['book_f05'],np.mean([m['positive_iou'] for m in t['by_book_pair'].values()]),-t['metrics']['negative_pair_fpr']))
    frozen=dict(name=selected['name'],settings=selected['settings'],validation_metrics=selected['metrics'],
        validation_by_book_pair=selected['by_book_pair'],validation_book_f05=selected['book_f05'],
        checkpoint_sha256=file_hash(a.checkpoint),matcher_version=MATCH_VERSION,crop='none',
        max_validation_negative_pair_fpr=a.max_validation_fpr,
        development_regression_passed=selected['development_regression_passed'],
        reordered_development_regression_passed=selected['reordered_development_regression_passed'],
        selection='Pass both excluded user-supplied development regressions, then maximum equal-book-pair validation F0.5 under FPR budget; test scores not used to select settings')
    write_json(out/'word_selected.json',frozen);print('FROZEN',json.dumps(frozen),flush=True)
    write_json(out/'word_validation/methods.json',[{k:v for k,v in t.items() if k!='rows'} for t in trials])
    test=torch.load(prepared/'test_prepared.pt',map_location='cpu',weights_only=False)
    cohort,excluded=exclude_reviewed(test['prepared'],hashes)
    final=run_methods(a.checkpoint,config,'test',cohort,[('Image-only word alignment',frozen['settings'])],out/'word_test',a.workers)[0]
    final['by_book_pair']=by_book(final['rows'],cohort)
    final['metrics']['confidence_intervals']=q.confidence_intervals(final['rows'],resamples=1000,seed=42)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    report=dict(checkpoint=str(a.checkpoint.relative_to(ROOT)),selected=frozen,
        final={k:v for k,v in final.items() if k!='rows'},development_pairs=list(DEVELOPMENT_PAIRS),
        excluded_test_pairs=excluded,validation_provenance=validation['provenance'],test_provenance=test['provenance'],
        book_training_counts=training_book_counts(saved['split_ids']),training_vocabulary=vocabulary,
        training_audit=dict(space_mode=saved['config']['space_mode'],negative_dtw_weight=saved['config']['negative_dtw_weight'],
            recorded_negative_objective=saved['metrics']['train']['negative_objective'],
            negative_loss_type=saved['config']['negative_loss_type'],
            hybrid_weights={key:saved['config'][key] for key in ('strong_negative_weight','wrong_letter_weight','order_negative_weight')},
            recorded_hybrid_losses={key:saved['metrics']['train'][key] for key in ('strong_neg_loss','local_wrong_letter_loss','order_loss')},
            current_training_lines_with_spaces=vocabulary['training_lines_with_spaces'],
            historical_transcript_hashes_available=False),
        source_sha256={name:file_hash(ROOT/name) for name in ('word_alignment.py','evaluate.py','evaluation_utils.py','scripts/eval/review_words.py')},
        interpretation='Descriptive saved-test results after validation-only selection. This cohort was previously inspected in aggregate. Vocabulary is learned only from current local saved-training transcripts; words absent from it may be missed.')
    write_json(out/'word_final.json',report)
    print('FINAL',json.dumps(final['metrics'],indent=2),flush=True)


if __name__=='__main__':main()

"""Validation-only refinement: retain two independently strong whole words.

The additional regression image is a fixed TRAINING example, never a test pair.
No score from the interrupted 600-pair evaluation is used in this refinement.
"""
from pathlib import Path
import gzip
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from dataset import file_hash, prepare_image
from evaluation_utils import EvaluationSession, load_folder_masks
from parameters import Config
from scripts.eval.calibrate_ab_confidence import CHECKPOINT, PRIOR, replay, scores, development, development_passed, choose
from scripts.eval.tune_matching import write_json, write_csv
from evaluate import resolved_match_settings

OUT = ROOT / 'Results/Evaluation/ab_low_fpr_20261109'


def choose_refinement(rows, options):
    eligible=[r for r in rows if r['negative_fpr']<=.1 and r['development_passed']]
    if not eligible:raise ValueError('No strong-pair refinement satisfies validation/development constraints')
    return max(eligible,key=lambda r:(round(r['iou'],12),round(r['recall'],12),round(r['precision'],12),
        -r['negative_fpr'],-options[r['name']]['word_strong_pair_max_nll'],
        options[r['name']]['word_strong_pair_min_cosine']))


def training_guard(methods, config):
    row = next(r for r in json.loads((PRIOR/'fixed_examples/selection.json').read_text())['samples']
               if r['review_id']=='A_B_example_03')
    if any(split != 'train' for split in row['saved_line_splits'].values()):
        raise ValueError('Additional guard must contain training images only')
    candidates = json.loads((OUT/'fixed_examples/A_B_example_03_candidates.json').read_text())
    lines, truth = [], []
    for book in 'AB':
        image = ROOT/'DataSet/ArabicData'/row[book]['image_path']
        geometry = prepare_image(image,(config.image_height,config.image_width),config.grayscale,'none')[2]
        lines.append(dict(geometry=geometry))
        with Image.open(ROOT/'DataSet/ArabicData'/row[book]['mask_path']) as source: truth.append(np.asarray(source)>0)
    results = {}
    for name, options in methods:
        regions, masks = replay(dict(match=dict(candidates=candidates),lines=lines),options,config)
        results[name] = dict(iou=float(scores(masks,truth)[0]),
            words=sorted({c['word'] for c in regions if c['candidate_type']=='word'}))
    return row, results


def main():
    torch.set_num_threads(1)
    coarse = json.loads((OUT/'coarse_selected.json').read_text())
    base = resolved_match_settings(coarse['settings'])
    methods = [('coarse',base)] + [(f'strong_pair_nll{nll}_cos{cos}',dict(base,
        word_strong_pair_max_nll=nll,word_strong_pair_min_cosine=cos))
        for nll in (.55,.6,.7,.8,1.) for cos in (.7,.75,.8)]
    write_json(OUT/'refinement_protocol.json',dict(split='val',
        reason='A fixed training example exposed rejection of two correct words after removal of a weak extra fragment',
        additional_development='A_B_example_03; both images in saved TRAIN split',
        exception='Exactly two distinct words may corroborate a match if every selected word meets stronger lexical and whole-word visual confidence. Existing total-character and physical-support requirements still apply.',
        max_negative_fpr=.1, objective='Maximum validation positive IoU; recall then precision, lower FPR, stricter strong-pair thresholds break ties',
        test_use='Earlier computation stopped before aggregate test scores were read; no test score used to revise thresholds',
        methods=[dict(name=name,settings=s) for name,s in methods]))
    session=EvaluationSession(CHECKPOINT,ROOT/'DataSet/ArabicData',split='val',device='cpu',
        construct_negatives=False,pair_source='folders',crop_override='none')
    session.model.eval().requires_grad_(False);session.text_encoder.eval().requires_grad_(False)
    row,train=training_guard(methods,session.config)
    dev=development(session,methods,OUT)
    # The reviewed examples are compared with the original baseline, not a
    # stricter intermediate point. Read their established reference scores.
    reference=[dict(sample_id='alignedPairs:'+name,iou=iou,words=words)
        for name,iou,words in [('pair_000016__A_line_014__B_line_003',.8949980881707333,['الله']),
                              ('pair_000028__A_line_005__B_line_002',.7601307804227682,['ابتغا','الله','من'])]]
    methods=[(name,s) for name,s in methods if train[name]['iou']>=.797
             and {'الذين','كفروا'}<=set(train[name]['words']) and development_passed(dev[name],reference)]
    print('Strong-pair refinements passing development guards:',len(methods),flush=True)
    cohort=json.loads((OUT/'validation_cohort.json').read_text())['records']
    by_id={r['sample_id']:r for r in cohort}; values={name:[] for name,_ in methods}; identities=[]
    # Candidate generation is unchanged, so the complete, verified lexical
    # candidate pools from the first validation pass remain exact.
    for number,path in enumerate(sorted((OUT/'candidate_cache').glob('*.pkl.gz')),1):
        with gzip.open(path,'rb') as stream: prediction=pickle.load(stream)
        sample=prediction['pair']['sample_id']; record=by_id[sample]
        pair=dict(sides=[dict(image=im['image']) for im in record['images']])
        truth,_=load_folder_masks(pair)
        if not all(np.all(m==m[:1]) for m in truth):raise ValueError('Non-column GT')
        truth=[m[:1] for m in truth];identities.append(record)
        for name,settings in methods:
            _,masks=replay(prediction,settings,session.config,columns=True)
            values[name].append([*scores(masks,truth),float(any(m.any() for m in masks))])
        if number%50==0:print('Refinement validation pairs:',number,'/',len(cohort),flush=True)
    if len(identities)!=len(cohort):raise ValueError('Incomplete validation candidate cache')
    positive=np.array([r['target']==1 for r in identities]);rows=[]
    for name,settings in methods:
        a=np.array(values[name]);mean=a[positive,:4].mean(0)
        rows.append(dict(name=name,iou=float(mean[0]),dice=float(mean[1]),precision=float(mean[2]),recall=float(mean[3]),
            positive_pairs=int(positive.sum()),negative_pairs=int((~positive).sum()),
            negative_false_positives=int(a[~positive,4].sum()),negative_fpr=float(a[~positive,4].mean()),
            positive_detection_rate=float(a[positive,4].mean()),development_passed=True))
    write_csv(OUT/'refinement_validation.csv',rows)
    best=choose_refinement(rows,dict(methods));settings=dict(methods)[best['name']]
    frozen=dict(coarse,name=best['name'],settings=settings,validation_metrics=best,development=dev[best['name']],
        training_regression=dict(review_id=row['review_id'],**train[best['name']]),
        selection='Saved-validation maximum positive IoU under 10% FPR; two reviewed cases plus one fixed training example are declared development guards')
    write_json(OUT/'training_regression.json',dict(row=row,results=train))
    write_json(OUT/'selected.json',frozen)
    print('FINAL FROZEN SELECTION:',best,flush=True)
    print('PARAMETERS:',{k:settings[k] for k in ('word_strong_pair_max_nll','word_strong_pair_min_cosine')},flush=True)


if __name__=='__main__':main()

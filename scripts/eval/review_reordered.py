"""Validate disjoint reordered passages and report fixed-cohort checkpoint metrics.

Confidence presets are selected on saved validation data only. Previously
inspected test images are excluded from headline test metrics. Reference weights
use the same remaining images, with their original membership reported explicitly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import replace
import gzip
import inspect
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataset import file_hash,prepare_image
from evaluate import MATCH_VERSION,resolved_match_settings
from evaluation_utils import (load_inference_text_encoder,predict_cached_pair,
    revalidate_cached_candidates,evaluation_split_ids,CANDIDATE_REVALIDATION_KEYS)
from parameters import Config
from scripts.eval import mask_quantitative as q
from scripts.eval.tune_matching import write_json,write_csv


def presets(baseline):
    methods = [('Current settings',baseline),
        ('Reordering only',dict(baseline,allow_reordered_regions=True)),
        ('Gap fragments only',dict(baseline,split_gap_candidates=True)),
        ('Visual confidence floor',dict(baseline,allow_reordered_regions=True,normal_min_mean_cosine=.6)),
        ('Gap fragments with visual floor',dict(baseline,allow_reordered_regions=True,
            split_gap_candidates=True,normal_min_mean_cosine=.6)),
        ('Conservative validation preset',dict(baseline,allow_reordered_regions=True,split_gap_candidates=True,
            normal_min_distinct_a=7,normal_min_distinct_b=7,normal_min_matched_pairs=7,
            normal_min_mean_cosine=.6,min_mutual_anchors_long=3))]
    for density in (.85,.95):
        for short_cosine in (.65,.75):
            for anchors in (5,7,9):
                name=f'joint_normal9_density{density}_short{short_cosine}_anchors{anchors}'
                settings=dict(baseline,allow_reordered_regions=True,split_gap_candidates=True,
                    normal_min_distinct_a=9,normal_min_distinct_b=9,normal_min_matched_pairs=12,
                    normal_min_mean_cosine=.6,min_path_density=density,
                    min_mutual_anchors_long=anchors,allow_short_regions=True,
                    require_corroborated_short_regions=True,
                    short_min_mean_cosine=short_cosine,short_min_median_cosine=short_cosine,
                    short_min_mutual_anchor_fraction=.6,min_mutual_anchor_fraction_short=.6)
                methods.append((name,settings))
    return [(name,resolved_match_settings(settings)) for name,settings in methods]


def pool_signature(checkpoint,settings):
    # Confidence/selection changes do not alter this GT-free pool. Fingerprint
    # the actual path/evidence generators separately from final validators.
    import alignment_candidates as candidates
    import evaluate as matching
    import dtw
    functions=[matching._local_repeat_path,matching.match_features,matching.checkpoint_letter_evidence,
        candidates.candidate_segments,candidates.segment_path,candidates.candidate_statistics,
        candidates.uniqueness_matrices,candidates.physical_gaps,candidates.valid_window_gaps,
        candidates._stat,candidates._number,matching.mutual_top_k_anchors,
        dtw.cosine_similarity_matrix,dtw.alphabet_log_probabilities,dtw.letter_evidence_matrix,
        dtw.bidirectional_window_scores]
    source={function.__name__:hashlib.sha256(inspect.getsource(function).encode()).hexdigest() for function in functions}
    source['candidate_construction']=hashlib.sha256(inspect.getsource(candidates.decode_candidates).split('    regions=select_candidates')[0].encode()).hexdigest()
    if settings['decoder']=='word_dtw':
        import word_alignment as words
        from word_alignment import load_vocabulary
        for function in (words._word_spots,words.word_lattice,words.image_word_boundaries,
                         words.full_span_path,words.long_visual_passages):
            source['word_'+function.__name__]=hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        source['word_construction']=hashlib.sha256(inspect.getsource(words.word_regions).split('    regions=select_word_candidates')[0].encode()).hexdigest()
        source['word_vocabulary']=q.identity(load_vocabulary(settings['word_vocabulary_path']))
    generating={key:value for key,value in settings.items() if key not in CANDIDATE_REVALIDATION_KEYS or key=='split_gap_candidates'}
    return q.identity(dict(checkpoint=file_hash(checkpoint),settings=generating,source=source,matcher=MATCH_VERSION))


def candidate_pool_key(generation,item):
    digest=hashlib.sha256()
    for line in item['lines']:
        digest.update(line['features']['fused'].contiguous().numpy().tobytes())
        digest.update(np.asarray(line['physical'],dtype=np.int64).tobytes())
    return q.identity(dict(generation=generation,sample=item['record']['sample_id'],
                           images=item['record']['images'],feature_sha256=digest.hexdigest()))


def cached_session(checkpoint,config,split):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    text=load_inference_text_encoder(saved,config,'cpu')
    return SimpleNamespace(config=config,text_encoder=text,split=split,settings={})


def score_partition(task):
    checkpoint,config,split,items,methods,destination,partition=task
    torch.set_num_threads(1)
    session=cached_session(checkpoint,config,split)
    pool_settings=dict(methods[0][1],split_gap_candidates=True,allow_reordered_regions=True)
    cache={}
    pool_dir=Path(destination)/'candidate_cache';pool_dir.mkdir(parents=True,exist_ok=True)
    generation=pool_signature(checkpoint,pool_settings)
    for number,item in enumerate(items,1):
        sample=item['record']['sample_id']
        path=pool_dir/(candidate_pool_key(generation,item)+'.pkl.gz')
        if path.exists():
            with gzip.open(path,'rb') as stream:match=pickle.load(stream)
            prediction=dict(pair=item['inference'],representation='fused',lines=item['lines'],
                cosine=item['cosine'],match=match,settings=match['settings'],split=split)
        else:
            prediction=predict_cached_pair(session,item['inference'],item['lines'],settings=pool_settings,cosine=item['cosine'])
            # Per-cell prose is unnecessary for validation or source masks. Keep
            # all route steps, supports and scalar evidence used by production.
            for candidate in prediction['match']['candidates']:
                candidate.pop('pair_confidence',None)
            with gzip.open(path,'wb') as stream:pickle.dump(prediction['match'],stream)
        cache[sample]=prediction
        if number%20==0:print(split,'partition',partition,'candidates',number,'/',len(items),flush=True)
    reports={}
    production=q.predict_cached_pair
    try:
        for index,(name,settings) in enumerate(methods):
            def replay(_session,pair,lines,representation,**kwargs):
                return revalidate_cached_candidates(_session,cache[pair['sample_id']],kwargs['settings'])
            q.predict_cached_pair=replay
            directory=Path(destination)/f'method_{index:02d}'/f'partition_{partition}'
            summary,rows=q.evaluate(session,items,'fused',settings,output=directory)
            reports[name]=rows
    finally:q.predict_cached_pair=production
    return reports


def run_methods(checkpoint,config,split,prepared,methods,destination,workers):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    workers=min(workers,sum(item['record']['category']=='positive' for item in prepared),
                sum(item['record']['category']=='negative' for item in prepared))
    if workers<1:raise ValueError('Evaluation requires positive and negative pairs and at least one worker')
    tasks=[(str(checkpoint),config,split,prepared[index::workers],methods,str(destination),index)
           for index in range(workers)]
    combined={name:[] for name,_ in methods}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for report in executor.map(score_partition,tasks):
            for name,rows in report.items():combined[name].extend(rows)
    results=[]
    for index,(name,settings) in enumerate(methods):
        rows=sorted(combined[name],key=lambda row:row['sample_id'])
        metrics=q.aggregate(rows)
        metrics['positive_detected_pairs']=sum(row['negative_false_positive'] for row in rows if row['category']=='positive')
        metrics['positive_pair_detection_recall']=metrics['positive_detected_pairs']/metrics['positive_pairs']
        directory=destination/f'method_{index:02d}'
        write_json(directory/'aggregate.json',metrics);write_json(directory/'settings.json',settings)
        write_csv(directory/'per_pair.csv',rows)
        results.append(dict(name=name,settings=settings,metrics=metrics,rows=rows,directory=str(directory)))
    write_json(destination/'methods.json',[{k:v for k,v in row.items() if k!='rows'} for row in results])
    return results


def reference_features(checkpoint,prepared,output,device='auto'):
    from train import load_checkpoint,build_loaders,resolve_device
    from dtw import cosine_similarity_matrix
    device=resolve_device(device)
    model,text,config,saved=load_checkpoint(checkpoint,device)
    config=replace(config,crop='content');model.eval()
    ids,_=evaluation_split_ids(checkpoint,saved)
    membership={}
    for split,loader in zip(('train','val','test'),build_loaders('DataSet/ArabicData',config,ids)):
        for record in loader.dataset.records:
            for side in record['sides']:membership[file_hash(side['image'])]=split
    features={};result=[];exposure=[]
    for number,item in enumerate(prepared,1):
        lines=[];owners=[]
        for side,image_record in zip(item['inference']['sides'],item['record']['images']):
            owners.append(membership.get(image_record['sha256'],'unseen'))
            key=side['image']
            if key not in features:
                _,tensor,geometry=prepare_image(key,(config.image_height,config.image_width),
                    config.grayscale,config.crop,side.get('bbox'),augment=False,binarize=config.binarize)
                with torch.no_grad():value=model(tensor[None].to(device))
                valid=value['token_valid'][0]
                features[key]=dict(features={rep:value[rep][0][valid].float().cpu() for rep in ('local','context','fused')},
                    physical=value['physical_window_indices'][0][valid].cpu().numpy(),geometry=geometry)
            lines.append(features[key])
        result.append(dict(item,lines=lines,cosine=cosine_similarity_matrix(*(line['features']['fused'] for line in lines)).numpy()))
        exposure.append(dict(sample_id=item['record']['sample_id'],category=item['record']['category'],
            reference_line_splits=owners,includes_reference_training_line='train' in owners))
        if number%25==0:print('Reference features',number,'/',len(prepared),flush=True)
    torch.save(dict(prepared=result,config=config,exposure=exposure,checkpoint_sha256=file_hash(checkpoint)),output)
    return result,config,exposure


def prepare_cohorts(output,checkpoint,device):
    from scripts.eval.tune_masks import parser as tuning_parser
    arguments=tuning_parser().parse_args(['prepare','--checkpoint',str(checkpoint),
        '--output',str(output/'current'),'--device',device,'--threads','4',
        '--negative-count','100','--pairs-per-split','all','--class-balance','natural',
        '--bootstrap','500','--quarantine-leaked-evaluation-images'])
    output.mkdir(parents=True,exist_ok=True)
    for split in ('val','test'):
        path=output/(split+'_prepared.pt')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['provenance']['checkpoint_sha256']!=file_hash(checkpoint):
                raise ValueError('Prepared cohort belongs to another checkpoint; use a new output directory')
            continue
        _,prepared,provenance=q.build_cache(arguments,split)
        baseline_report=ROOT/'docs/evaluation/reordered_regions_20261109/metrics.json'
        if baseline_report.is_file():
            historical=json.loads(baseline_report.read_text())
            reference=next(method for method in historical['test_methods'] if method['name']=='Current settings')
            provenance['baseline_settings']=resolved_match_settings(reference['settings'])
            provenance['baseline_source']=dict(path=str(baseline_report),sha256=file_hash(baseline_report),
                interpretation='Frozen pre-change matcher, rather than the now-updated notebook defaults')
        torch.save(dict(prepared=prepared,provenance=provenance),path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'Results/Evaluation/reordered_regions_20261109')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--development-pair',default='pair_000028__A_line_005__B_line_002',
        help='Previously inspected pair; its source images are excluded from headline test metrics')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    parser.add_argument('--reference',type=Path,default=ROOT/'Weights/book_c_phase4_sum_nogate_ft_20261108/checkpoint_best.pt')
    args=parser.parse_args();out=args.output
    prepare_cohorts(out,args.checkpoint,args.device)
    torch.set_num_threads(1)
    validation=torch.load(out/'val_prepared.pt',map_location='cpu',weights_only=False)
    baseline=validation['provenance']['baseline_settings']
    methods=presets(baseline);config=Config(**validation['provenance']['config'])
    validation_results=run_methods(args.checkpoint,config,'val',validation['prepared'],methods,out/'validation_anchored',args.workers)
    eligible=[row for row in validation_results if row['metrics']['negative_pair_fpr']<=.05]
    if not eligible:raise ValueError('No validation preset meets the predeclared 5% presumed-negative FPR limit')
    selected=max(eligible,key=lambda row:(row['metrics']['positive_iou'],-row['metrics']['negative_pair_fpr'],-row['metrics']['negative_masked_area_fraction']))
    frozen=dict(name=selected['name'],settings=selected['settings'],validation_metrics=selected['metrics'],
        selection='Highest validation foreground IoU with presumed-negative pair FPR <= 5%; test metrics were not used',
        checkpoint_sha256=file_hash(args.checkpoint),matcher_version=MATCH_VERSION)
    write_json(out/'selected.json',frozen);print('SELECTED',frozen,flush=True)
    test=torch.load(out/'test_prepared.pt',map_location='cpu',weights_only=False)
    row=json.loads(next(line for line in (ROOT/'DataSet/ArabicData/alignedPairs/A_B/pairs.jsonl').read_text().splitlines()
        if json.loads(line)['pair_id']==args.development_pair))
    reviewed_hashes={file_hash(ROOT/'DataSet/ArabicData'/row[book]['image_path']) for book in ('A','B')}
    excluded=[item['record']['sample_id'] for item in test['prepared'] if any(image['sha256'] in reviewed_hashes for image in item['record']['images'])]
    cohort=[item for item in test['prepared'] if item['record']['sample_id'] not in excluded]
    write_json(out/'test_cohort.json',dict(excluded_development_pairs=excluded,reason='Previously inspected user example or shared source line',
        records=[item['record'] for item in cohort],original_pairs=len(test['prepared']),evaluated_pairs=len(cohort)))
    test_methods=[('Current settings',baseline),('Reordering only',dict(baseline,allow_reordered_regions=True)),('Validation selected',selected['settings'])]
    current=run_methods(args.checkpoint,config,'test',cohort,test_methods,out/'test_current_anchored',args.workers)
    reference_path=out/'reference_prepared.pt'
    if reference_path.exists():
        saved=torch.load(reference_path,map_location='cpu',weights_only=False)
        if saved.get('checkpoint_sha256')!=file_hash(args.reference):
            raise ValueError('Reference cache belongs to different or unverified weights; use a new output directory')
        reference,reference_config,exposure=saved['prepared'],saved['config'],saved['exposure']
        if {item['record']['sample_id'] for item in reference}!={item['record']['sample_id'] for item in cohort}:
            raise ValueError('Reference cache belongs to a different fixed cohort')
    else:reference,reference_config,exposure=reference_features(args.reference,cohort,reference_path,args.device)
    old=run_methods(args.reference,reference_config,'current_test_reference_membership_reported',reference,
        [('Previous checkpoint, same cohort/settings',baseline)],out/'test_reference_anchored',args.workers)
    reports=old+current
    for report in reports:
        report['metrics']['confidence_intervals']=q.confidence_intervals(report['rows'],resamples=1000,seed=42)
        report['by_book_pair']={group:q.aggregate([row for row in report['rows'] if row['category']=='negative' or
            any(item['record']['sample_id']==row['sample_id'] and item['record'].get('source_folder')==group for item in cohort)])
            for group in ('A_B','A_C','B_C') if any(item['record'].get('source_folder')==group and item['record']['category']=='positive' for item in cohort)}
    summary=dict(selected=frozen,development_pair=args.development_pair,
        source_sha256={name:file_hash(ROOT/name) for name in ('alignment_candidates.py','evaluate.py','evaluation_utils.py','scripts/eval/review_reordered.py')},
        test_methods=[{k:v for k,v in report.items() if k!='rows'} for report in reports],
        reference_membership=exposure,development_pairs_excluded=excluded,
        comparison_limitation='Previous and current checkpoints have zero common saved-test positive pairs. The same-cohort reference is descriptive; reference training membership is reported per pair.',
        validation_quarantine=validation['provenance'].get('evaluation_quarantine'))
    write_json(out/'comparison.json',summary)
    write_csv(out/'summary.csv',[dict(method=report['name'],**{k:v for k,v in report['metrics'].items() if not isinstance(v,dict)}) for report in reports])
    print(json.dumps([{k:v for k,v in row.items() if k not in ('rows','settings','by_book_pair')} for row in reports],indent=2),flush=True)


if __name__=='__main__':main()

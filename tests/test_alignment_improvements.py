"""Sequence-negative, padding, evidence and decoder contracts (CPU only)."""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest
import torch

from dataset import generate_negative_transcripts
from dtw import cosine_similarity_matrix, letter_cost_matrix, letter_evidence_matrix
from evaluate import decode_stretched_regions, stretch_candidates, match_features, region_mask
from losses import compute_loss, negative_dtw_margin_loss, positive_dtw_loss
from parameters import Config, validate_objective
from text_embedding import OrthogonalCharEmbedding, clean_letters
from train import build_model, build_loaders, train_one_epoch, validate_one_epoch, save_checkpoint, load_checkpoint
from test_dataset import make_synthetic
from test_evaluation_notebook import evaluation_fixture
from evaluation_utils import EvaluationSession, evaluate_population


@pytest.mark.parametrize('operation',['substitute','adjacent','blocks','words','shift','shuffle'])
def test_corruptions(operation):
    text='السَّلام عليكم ورحمة الله'
    kw=dict(count=4,operations=operation,seed=91,sample_id='stable',epoch=3)
    values,stats=generate_negative_transcripts(text,**kw)
    assert (values,stats)==generate_negative_transcripts(text,**kw)
    assert values and len(values)==len(set(values))
    positive=''.join(clean_letters(text))
    assert all(v!=positive and len(v)==len(positive) and ''.join(clean_letters(v))==v for v in values)
    assert stats['attempts']<=96
    forbidden=generate_negative_transcripts(text,**kw,equivalents=values)[0]
    assert not set(forbidden)&set(values)


def test_repeated_short_and_empty():
    for text in ('','َ','ا','اااا'):
        values,info=generate_negative_transcripts(text,operations='adjacent,blocks,shift,shuffle',count=5)
        assert values==[] and info['attempts']<=120
    a=generate_negative_transcripts('السلامعليكم',sample_id='a',epoch=1)[0]
    b=generate_negative_transcripts('السلامعليكم',sample_id='a',epoch=2)[0]
    assert a!=b


def test_ranking_sign_both_cost_gradients(monkeypatch):
    import losses
    raw=torch.tensor([[[2.,1.]]],requires_grad=True)
    def cost(v,letters,*args): return v[0,0] if ''.join(letters)=='اب' else v[0,1]
    monkeypatch.setattr(losses,'_line_loss',cost)
    loss=negative_dtw_margin_loss(raw,['اب'],[['با']],None,Config(negative_margin=.2))
    assert loss.item()==pytest.approx(1.2)
    loss.backward()
    torch.testing.assert_close(raw.grad,torch.tensor([[[1.,-1.]]]))


def test_ddp_ranking_uses_its_own_valid_line_denominator(monkeypatch):
    import losses
    monkeypatch.setattr(losses,'_line_loss',lambda v,t,*args: v[0,0] if ''.join(t)=='اب' else v[0,1])
    monkeypatch.setattr(losses.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(losses.dist,'get_world_size',lambda:2)
    counts=iter([4.,1.])  # positive global lines vs ranked global lines
    monkeypatch.setattr(losses.dist,'all_reduce',lambda value:value.fill_(next(counts)))
    raw=torch.tensor([[[2.,1.]]],requires_grad=True)
    cfg=Config(negative_dtw_weight=.1)
    loss,stats=compute_loss(dict(fused=raw,fused_pre_l2=raw,token_valid=torch.ones(1,1,dtype=torch.bool)),
                           ['اب'],None,cfg,negative_texts=[['با']],distributed_statistics=True)
    assert loss.item()==pytest.approx(.5*2+.1*2*1.2)
    loss.backward()
    torch.testing.assert_close(raw.grad,torch.tensor([[[.7,-.2]]]))


def test_disabled_negative_and_rectangular(monkeypatch):
    import losses
    def forbidden(*args,**kwargs): raise AssertionError('negative path called')
    monkeypatch.setattr(losses,'negative_dtw_margin_loss',forbidden)
    raw=torch.randn(2,3,8,requires_grad=True)
    text=OrthogonalCharEmbedding(8,4096)
    config=Config(embedding_dim=8)
    output=dict(fused=raw,fused_pre_l2=raw,token_valid=torch.ones(2,3,dtype=torch.bool))
    value,stats=compute_loss(output,['سلامعليكم','باب'],text,config)
    positive,_=positive_dtw_loss(raw,['سلامعليكم','باب'],text,config)
    torch.testing.assert_close(value,positive)
    value.backward()
    assert torch.isfinite(raw.grad).all() and stats['ranking_candidates']==0
    assert cosine_similarity_matrix(raw[0],text.encode('سلام')).shape==(3,4)
    with pytest.raises(ValueError,match='dimensions'):
        letter_cost_matrix(raw[0],torch.randn(5,7),alphabet=torch.randn(6,8),letter_ids=list(range(5)))
    with pytest.raises(ValueError,match='dimensions'):
        cosine_similarity_matrix(raw[0],torch.randn(9,7))


def test_padding_before_cnn_rtl_boundary():
    cfg=Config(cnn_type='simple',cnn_pretrained=False,embedding_dim=16,local_dropout=0.,
               transformer_layers=1,transformer_heads=1,image_height=32,image_width=96)
    model=build_model(cfg).train()
    twin=copy.deepcopy(model)
    x=torch.randn(2,1,32,96)
    changed=x.clone(); changed[:,:,:,64:]=float('nan')
    # Width 70: boundary window [48,80) is only partially real, so excluded.
    a=model(x,valid_widths=[70,70]); b=twin(changed,valid_widths=[70,70])
    valid=a['token_valid']
    assert valid[0].tolist()==[False,False,True,True,True]
    assert a['physical_window_indices'][0].tolist()==[-1,-1,2,1,0]
    torch.testing.assert_close(a['fused'][valid],b['fused'][valid],rtol=0,atol=0)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,twin.state_dict()[key],rtol=0,atol=0)


def test_evidence_uniform_prior_blank_uncertainty():
    a=torch.full((3,7),-np.log(7)); b=torch.full((9,7),-np.log(7))
    result=letter_evidence_matrix(a,b,acceptance_offset=.1)
    assert result.shape==(3,9)
    torch.testing.assert_close(result,torch.full((3,9),-.1),atol=1e-6,rtol=0)
    assert torch.isfinite(letter_evidence_matrix(a,b,prior=[1e-50]*6+[1.])).all()
    assert (letter_evidence_matrix(a-10,b-10)<0).all()  # blank mass retained


@pytest.mark.parametrize('short,long',[(3,5),(5,9)])
def test_unequal_local_matching_and_trace_objective(short,long):
    r=np.full((short+5,long+8),-1.)
    expected=[]
    for j in range(long):
        i=round(j*(short-1)/(long-1))
        r[i+1,j+4]=.8;expected.append((i+1,j+4))
    candidates,_=stretch_candidates(r)
    assert candidates and candidates[0]['score']==pytest.approx(sum(s['delta'] for s in candidates[0]['path']))
    result=decode_stretched_regions(r,np.arange(len(r)),np.arange(r.shape[1]))
    region=result['regions'][0]
    assert region['pairs']==expected and region['support']==[short,long]
    assert region['score']==pytest.approx(region['matching_evidence']-region['repeat_penalties']-region['gap_penalties'])
    assert region['score']==pytest.approx(sum(s['delta'] for s in region['path']))


def test_weak_large_gaps_crossing_multiple_and_no_match():
    r=np.full((20,25),-2.)
    for i in range(5): r[i+1,i+4]=.9
    r[3,6]=-.1
    for i in range(4): r[i+12,i+17]=.8
    r[2,21]=2.  # unsupported distracting peak
    result=decode_stretched_regions(r,np.arange(19,-1,-1),np.arange(24,-1,-1))
    assert len(result['regions'])==2
    assert result['regions'][0]['weak_spans']
    assert len(result['regions'][0]['filled_physical'][0])==1
    geo=dict(source_size=[400,50],crop=[20,5,380,45],scale_x=320/360,model_size=[320,32])
    mask=region_mask(result['regions'],0,geo,16,16)
    assert mask.shape==(50,400) and np.array_equal(mask[0],mask[-1]) and not mask[:,:20].any()
    assert not decode_stretched_regions(np.full((4,9),-.1),np.arange(4),np.arange(9))['regions']
    crossing=np.full((12,12),-1.)
    for i in range(3): crossing[i,i+7]=1.; crossing[i+7,i]=.9
    regions=decode_stretched_regions(crossing,np.arange(12),np.arange(12))['regions']
    assert len(regions)==1
    features=torch.ones(8,16)
    assert not match_features(features,features,np.arange(8),np.arange(8))['regions']


def test_ctc_complete_transcript_and_feasibility():
    config=Config(alignment_objective='ctc',position_prior=0.,embedding_dim=8)
    validate_objective(config)
    text=OrthogonalCharEmbedding(8,4096)
    raw=torch.randn(2,4,8,requires_grad=True)
    loss,stats=positive_dtw_loss(raw,['باب','ااا'],text,config)
    assert stats['evaluated']==1 and stats['skip_reasons']=={'ctc_repeated_label_feasibility':1}
    loss.backward(); assert torch.isfinite(raw.grad).all()
    with pytest.raises(ValueError,match='position-prior'):
        validate_objective(replace(config,position_prior=.15))


def test_generation_epoch_fixed_validation_and_training(tmp_path,monkeypatch):
    root=make_synthetic(tmp_path,count=10)
    cfg=Config(cnn_type='simple',cnn_pretrained=False,image_height=32,image_width=64,
        embedding_dim=16,transformer_layers=1,transformer_heads=1,num_workers=0,batch_size=2,
        negative_dtw_weight=.1,negative_count=2,augmentation=False)
    train,val,test=build_loaders(root,cfg)
    train.dataset.set_epoch(1); first=train.dataset[0]
    assert first['negative_texts'][0]
    val.dataset.set_epoch(1); before=val.dataset[0]['negative_texts']
    val.dataset.set_epoch(9); assert val.dataset[0]['negative_texts']==before
    assert 'negative_texts' not in test.dataset[0]
    model=build_model(cfg); text=OrthogonalCharEmbedding(16,4096)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    stats=train_one_epoch(model,text,train,optimizer,cfg,'cpu',max_batches=1,epoch=1)
    assert stats['ranking_candidates']>0 and stats['corruption']['generated']>0
    assert stats['total']==pytest.approx(stats['positive_dtw']+.1*stats['negative_dtw'])
    stats=validate_one_epoch(model,text,val,cfg,'cpu',max_batches=1)
    assert stats['ranking_candidates']>0
    def forbidden(*args,**kwargs): raise AssertionError('disabled generation called')
    monkeypatch.setattr('dataset.generate_negative_transcripts',forbidden)
    off=build_loaders(root,replace(cfg,negative_dtw_weight=0.))[0]
    assert 'negative_texts' not in off.dataset[0]


def test_dispatch_cache_gt_independence_and_old_checkpoint(evaluation_fixture):
    root,cp=evaluation_fixture
    session=EvaluationSession(cp,root,device='cpu')
    pair=session.pairs[0]
    a=session.predict_pair(pair)
    key=session.matching_cache_key(pair)
    session.settings['similarity_mode']='letter_evidence'
    assert key!=session.matching_cache_key(pair)
    b=session.predict_pair(pair)
    assert len(session.feature_cache)==2 and b['match']['prior_metadata']['source']=='explicit uniform fallback'
    assert a['cosine'].shape==b['cosine'].shape
    hostile=dict(pair,annotations=[{'mask':'not-a-file'}]*2,label='no_shared_content',target=0)
    np.testing.assert_array_equal(b['match']['rewards'],session.predict_pair(hostile)['match']['rewards'])
    session.pairs=[pair,pair]
    assert evaluate_population(session)['evaluated_pairs']==1
    model,text,cfg,saved=load_checkpoint(cp)
    old=torch.load(cp)
    for name in ('alignment_objective','negative_count','alphabet_inventory'):
        old['config'].pop(name)
    old.pop('letter_evidence_prior',None)
    torch.save(old,cp)
    model,text,cfg,saved=load_checkpoint(cp)
    assert cfg.alignment_objective=='dtw' and text.letter_evidence_prior is None


def test_masked_loss_padding_and_warmup(tmp_path,monkeypatch):
    raw=torch.randn(1,6,8,requires_grad=True)
    valid=torch.tensor([[True,True,True,False,False,False]])
    padded=raw.detach().clone();padded[:,3:]=float('nan');padded.requires_grad_()
    text=OrthogonalCharEmbedding(8,4096);cfg=Config(embedding_dim=8)
    a,_=compute_loss(dict(fused=raw,fused_pre_l2=raw,token_valid=valid),['سلام'],text,cfg)
    b,_=compute_loss(dict(fused=padded,fused_pre_l2=padded,token_valid=valid),['سلام'],text,cfg)
    torch.testing.assert_close(a,b)
    b.backward(); assert torch.isfinite(padded.grad).all() and not padded.grad[:,3:].any()
    root=make_synthetic(tmp_path,count=10)
    cfg=replace(cfg,negative_dtw_weight=.1,negative_warmup_epochs=2,num_workers=0)
    view=build_loaders(root,cfg)[0].dataset
    def forbidden(*args,**kwargs): raise AssertionError('warmup generated negatives')
    monkeypatch.setattr('dataset.generate_negative_transcripts',forbidden)
    view.set_epoch(2)
    assert not view[0]['negative_texts'][0]


def test_workers_do_not_change_negatives(tmp_path):
    root=make_synthetic(tmp_path,count=10)
    cfg=Config(negative_dtw_weight=.1,batch_size=3,num_workers=0,augmentation=False)
    result=[]
    for workers in (0,2):
        from torch.utils.data import DataLoader
        from dataloader import collate_samples
        view=build_loaders(root,cfg)[0].dataset
        # Spawn avoids forking an already initialized OpenMP pool in this CPU test.
        loader=DataLoader(view,batch_size=3,num_workers=workers,collate_fn=collate_samples,
                          **dict(multiprocessing_context='spawn',timeout=30) if workers else {})
        loader.dataset.set_epoch(3)
        result.append({sid:n for batch in loader for sid,n in zip(batch['sample_id'],batch['negative_texts'])})
    assert result[0]==result[1]


def test_exact_resume_and_explicit_finetune(tmp_path):
    from train import main
    root=make_synthetic(tmp_path/'data',count=10)
    base=['--dataset',str(root),'--output-root',str(tmp_path),'--cnn-type','simple','--no-cnn-pretrained',
          '--num-workers','0','--image-height','32','--image-width','64','--embedding-dim','16',
          '--transformer-layers','1','--transformer-heads','1','--batch-size','2','--max-batches','1',
          '--negative-dtw-weight','.1']
    continuous=main(base+['--run-name','continuous','--epochs','2'])
    interrupted=main(base+['--run-name','interrupted','--epochs','1'])
    checkpoint=interrupted/'checkpoint_latest.pt'
    main(base+['--run-name','interrupted','--epochs','2','--resume',str(checkpoint)])
    a=load_checkpoint(continuous/'checkpoint_latest.pt')[0].state_dict()
    b=load_checkpoint(checkpoint)[0].state_dict()
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='configuration differs'):
        main(base+['--run-name','interrupted','--epochs','3','--resume',str(checkpoint),'--negative-dtw-weight','.2'])
    fine=main(base+['--run-name','finetune','--epochs','1','--finetune',str(checkpoint),'--negative-dtw-weight','.2'])
    saved=load_checkpoint(fine/'checkpoint_latest.pt')[3]
    assert saved['run_mode']=='finetune' and saved['source_config']['negative_dtw_weight']==.1
    assert saved['config']['negative_dtw_weight']==.2 and saved['letter_evidence_prior']['source'].startswith('training-only')


def _distributed_ranking_worker(rank, rendezvous, output):
    from datetime import timedelta
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(19)
        raw=torch.randn(1,5,8,requires_grad=True)
        texts=['باب'] if rank==0 else ['سلام','قال']
        negatives=[['ابب']] if rank==0 else [[],[]]
        values=raw.expand(len(texts),-1,-1)
        cfg=Config(embedding_dim=8,negative_dtw_weight=.3,negative_margin=10.)
        loss,_=compute_loss(dict(fused=values,fused_pre_l2=values,
                                token_valid=torch.ones(len(texts),5,dtype=torch.bool)),
                            texts,OrthogonalCharEmbedding(8,4096),cfg,
                            negative_texts=negatives,distributed_statistics=True)
        loss.backward()
        dist.all_reduce(raw.grad);raw.grad.div_(2)  # DDP's gradient averaging
        if rank==0:torch.save(raw.grad,output)
    finally:dist.destroy_process_group()


def test_two_rank_counts_match_single_population_gradient(tmp_path):
    torch.multiprocessing.spawn(_distributed_ranking_worker,
        args=(str(tmp_path/'rendezvous'),str(tmp_path/'gradient.pt')),nprocs=2,join=True)
    torch.manual_seed(19)
    raw=torch.randn(1,5,8,requires_grad=True)
    values=raw.expand(3,-1,-1)
    cfg=Config(embedding_dim=8,negative_dtw_weight=.3,negative_margin=10.)
    loss,_=compute_loss(dict(fused=values,fused_pre_l2=values,token_valid=torch.ones(3,5,dtype=torch.bool)),
                        ['باب','سلام','قال'],OrthogonalCharEmbedding(8,4096),cfg,
                        negative_texts=[['ابب'],[],[]])
    loss.backward()
    torch.testing.assert_close(torch.load(tmp_path/'gradient.pt'),raw.grad,atol=1e-6,rtol=1e-5)

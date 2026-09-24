"""Standalone training: data -> model -> objective -> train/validate -> checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import create_dataloaders, collate_samples
from losses import compute_loss
from model import AlignmentModel
from parameters import Config
from text_embedding import OrthogonalCharEmbedding


def build_model(config, initialize=True):
    return AlignmentModel(cnn_type=config.cnn_type, transformer_type=config.transformer_type,
                          embedding_dim=config.embedding_dim, window_width=config.window_width,
                          stride=config.window_stride, pretrained_cnn=initialize and config.cnn_pretrained,
                          use_positional_encoding=config.use_positional_encoding,
                          input_channels=1 if config.grayscale else 3, local_dropout=config.local_dropout,
                          transformer_dropout=config.transformer_dropout, rtl=config.rtl)


def build_loaders(dataset, config, split_ids=None):
    return create_dataloaders(dataset, config.dataset_type, config.train_ratio, config.val_ratio,
                              config.test_ratio, config.split_mode, config.split_seed,
                              config.batch_size, config.num_workers, config.augmentation,
                              split_ids=split_ids, paired=config.paired,
                              image_size=(config.image_height,config.image_width),
                              grayscale=config.grayscale, crop=config.crop, binarize=config.binarize)


def _batch_lines(batch, device):
    images, texts = batch['image'], list(batch['text'])
    if torch.is_tensor(batch['image2']):
        images = torch.cat((images,batch['image2']))
        texts.extend(batch['text2'])
    return images.to(device), texts


def _run_epoch(model, text_encoder, loader, config, device, optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    text_encoder.eval()
    distributed = training and dist.is_available() and dist.is_initialized()
    # DTW is aggregated per LINE, including both sides, excluding empty text.
    # SIGReg is a token-weighted batch-population statistic, not a per-line loss.
    totals = torch.zeros(8,dtype=torch.float64,device=device)
    started = time.monotonic()
    for index,batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        images,texts = _batch_lines(batch,device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.use_amp and device.type=='cuda'):
                output = model(images)
            loss,stats = compute_loss(output,texts,text_encoder,config,
                                     distributed_statistics=distributed,
                                     sketch_seed=None if training else config.seed)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective; no optimizer step taken')
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradients; no optimizer step taken')
                optimizer.step()
        tokens = stats['valid_tokens']
        totals += totals.new_tensor([stats['dtw_sum'],stats['evaluated'],stats['skipped'],
                                      (stats['sigreg'] or 0.)*tokens,tokens,
                                      sum(t for t,l in stats['lengths']),
                                      sum(l for t,l in stats['lengths']),1])
    if distributed:
        dist.all_reduce(totals)
    dtw_sum,n,skipped,sig_sum,tokens,windows,letters,batches = totals.tolist()
    positive = dtw_sum/n if n else None
    sigreg = sig_sum/tokens if tokens and config.sigreg_weight else None
    total = None if positive is None else config.positive_dtw_weight*positive + config.sigreg_weight*(sigreg or 0.)
    return dict(total=total,positive_dtw=positive,negative_dtw=None,sigreg=sigreg,
                weighted_sigreg=config.sigreg_weight*(sigreg or 0.),evaluated=int(n),skipped=int(skipped),
                valid_tokens=int(tokens),batches=int(batches),mean_windows=windows/n if n else None,
                mean_letters=letters/n if n else None,gamma=config.dtw_gamma,
                population='explicit-batch-subset' if max_batches else 'full-split',
                seconds=time.monotonic()-started)


def train_one_epoch(model,text_encoder,loader,optimizer,config,device,max_batches=0):
    return _run_epoch(model,text_encoder,loader,config,torch.device(device),optimizer,max_batches)


def validate_one_epoch(model,text_encoder,loader,config,device,max_batches=0):
    """Local unwrapped evaluation: no DDP collectives, updates or RNG side effects."""
    if isinstance(model,DistributedDataParallel):
        raise ValueError('Coordinated rank-zero validation requires model.module')
    if loader.dataset.augment:
        raise ValueError('Validation must use an augmentation-free dataset view')
    device = torch.device(device)
    modes = [(module,module.training) for module in list(model.modules())+list(text_encoder.modules())]
    python_rng,numpy_rng = random.getstate(),np.random.get_state()
    loader_rng = loader.generator.get_state() if loader.generator is not None else None
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []), torch.no_grad():
            return _run_epoch(model,text_encoder,loader,config,device,max_batches=max_batches)
    finally:
        for module,mode in modes:
            module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if loader_rng is not None:
            loader.generator.set_state(loader_rng)


def save_checkpoint(path,model,optimizer,epoch,best_val,config,text_encoder,split_ids,**metadata):
    raw = model.module if isinstance(model,DistributedDataParallel) else model
    payload = dict(format_version=1,architecture_family='simple-alignment-core',
                   model=raw.state_dict(),text_embedding=text_encoder.state_dict(),
                   optimizer=optimizer.state_dict() if optimizer else None,epoch=epoch,best_val=best_val,
                   config=asdict(config),split_ids=split_ids,
                   split_manifest_sha256=hashlib.sha256(json.dumps(split_ids,sort_keys=True).encode()).hexdigest(),
                   parameter_count=sum(p.numel() for p in raw.parameters()),
                   initialization=raw.cnn.initialization,**metadata)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def load_checkpoint(path,device='cpu'):
    """Strict new-format reconstruction. Legacy checkpoints are not migrated."""
    saved = torch.load(path,map_location='cpu')
    if saved.get('format_version') != 1 or saved.get('architecture_family') != 'simple-alignment-core':
        raise ValueError('Incompatible legacy checkpoint: expected standalone simple-alignment-core format 1')
    config = Config(**saved['config'])
    model = build_model(config,initialize=False).to(device)
    model.load_state_dict(saved['model'],strict=True)
    model.cnn.initialization = saved['initialization']
    text = OrthogonalCharEmbedding(config.embedding_dim,config.text_vocab_size,config.text_embedding_seed).to(device)
    text.load_state_dict(saved['text_embedding'],strict=True)
    if hashlib.sha256(json.dumps(saved['split_ids'],sort_keys=True).encode()).hexdigest() != saved['split_manifest_sha256']:
        raise ValueError('Checkpoint split identity is inconsistent')
    model.eval()
    text.eval()
    return model,text,config,saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--output-root',default='Weights')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--max-batches',type=int,default=0,help='Explicit smoke cap on BOTH passes; 0=full splits')
    parser.add_argument('--resume',help='New-format checkpoint; config must match (epochs may increase)')
    for field in fields(Config):
        kwargs = dict(default=field.default)
        if isinstance(field.default,bool): kwargs['action']=argparse.BooleanOptionalAction
        else: kwargs['type']=type(field.default)
        parser.add_argument('--'+field.name.replace('_','-'),**kwargs)
    args = parser.parse_args(argv)
    config = Config(**{f.name:getattr(args,f.name) for f in fields(Config)})
    if config.negative_dtw_weight:
        raise ValueError('This CLI has no negative-transcript source; leave negative_dtw_weight=0')
    if args.max_batches < 0 or Path(args.run_name).name != args.run_name:
        raise ValueError('Use a simple run name and nonnegative smoke cap')
    rank,world,local_rank = [int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')]]
    device = torch.device(('cuda:'+str(local_rank) if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    random.seed(config.seed+rank)
    np.random.seed(config.seed+rank)
    torch.manual_seed(config.seed+rank)
    output_dir = Path(args.output_root)/args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite run {output_dir}; use a new run name')
    if world>1: dist.barrier()
    if rank==0: output_dir.mkdir(parents=True,exist_ok=bool(args.resume))
    if world>1: dist.barrier()
    start,best,saved = 0,float('inf'),None
    if args.resume:
        model,text,saved_config,saved = load_checkpoint(args.resume,device)
        expected,actual = asdict(saved_config),asdict(config)
        expected.pop('epochs')
        actual.pop('epochs')
        if expected != actual: raise ValueError('Resume configuration differs from checkpoint')
        start,best = saved['epoch'],saved['best_val']
    else:
        model = build_model(config).to(device)
        text = OrthogonalCharEmbedding(config.embedding_dim,config.text_vocab_size,config.text_embedding_seed).to(device)
    loaders = build_loaders(args.dataset,config,saved['split_ids'] if saved else None)
    train_loader,val_loader,_test_loader = loaders  # Never iterate test during training.
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError('Training and validation need nonempty group splits')
    split_ids = {name:[r['sample_id'] for r in loader.dataset.records]
                 for name,loader in zip(('train','val','test'),loaders)}
    sampler = None
    if world>1:
        sampler = DistributedSampler(train_loader.dataset,seed=config.seed,shuffle=True)
        train_loader = DataLoader(train_loader.dataset,batch_size=config.batch_size,sampler=sampler,
                                  num_workers=config.num_workers,collate_fn=collate_samples)
        model = DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    if saved: optimizer.load_state_dict(saved['optimizer'])
    history_path = output_dir/'history.json'
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []
    if rank==0:
        print('CONFIG',json.dumps(asdict(config),sort_keys=True),flush=True)
        print('SPLITS',{name:len(ids) for name,ids in split_ids.items()},flush=True)
        print('SIGReg: token-weighted batch statistic; validation seeded sketches, local collectives disabled.',flush=True)
        (output_dir/'split_manifest.json').write_text(json.dumps(split_ids,indent=2))
    for epoch in range(start+1,config.epochs+1):
        if sampler: sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model,text,train_loader,optimizer,config,device,args.max_batches)
        if world>1: dist.barrier()
        result = [None]
        if rank==0:
            try:
                raw = model.module if world>1 else model
                val_stats = validate_one_epoch(raw,text,val_loader,config,device,args.max_batches)
                if val_stats['total'] is None: raise ValueError('Validation has no nonempty cleaned transcripts')
                result[0] = dict(stats=val_stats)
            except Exception as exc:
                result[0] = dict(error=f'{type(exc).__name__}: {exc}')
        if world>1: dist.broadcast_object_list(result,src=0)
        if 'error' in result[0]: raise RuntimeError(result[0]['error'])
        val_stats = result[0]['stats']
        improved = val_stats['total'] < best
        best = min(best,val_stats['total'])
        if rank==0:
            print(f'Epoch {epoch}/{config.epochs}\nTrain loss: {train_stats["total"]}\nValidation loss: {val_stats["total"]}',flush=True)
            entry = dict(epoch=epoch,train=train_stats,validation=val_stats)
            history.append(entry)
            history_path.write_text(json.dumps(history,indent=2))
            kwargs = dict(selection_metric='validation.total',selection_population=val_stats['population'],
                          dataset=str(Path(args.dataset).resolve()),max_batches=args.max_batches,metrics=entry)
            save_checkpoint(output_dir/'checkpoint_latest.pt',model,optimizer,epoch,best,config,text,split_ids,**kwargs)
            if improved: save_checkpoint(output_dir/'checkpoint_best.pt',model,optimizer,epoch,best,config,text,split_ids,**kwargs)
        if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()
    return output_dir


if __name__=='__main__':
    main()

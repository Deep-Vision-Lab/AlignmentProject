"""Standalone training: data -> model -> objective -> train/validate -> checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from dataloader import create_dataloaders, collate_samples
from cnn_encoder import simple_cnn_channels
from losses import compute_loss
from model import AlignmentModel, validate_fusion_config
from parameters import Config, validate_objective
from text_embedding import OrthogonalCharEmbedding
from text_embedding import ARABIC_LETTERS, clean_letters, fit_letter_prior


def build_model(config, initialize=True):
    return AlignmentModel(cnn_type=config.cnn_type, transformer_type=config.transformer_type,
                          embedding_dim=config.embedding_dim, window_width=config.window_width,
                          stride=config.window_stride, pretrained_cnn=initialize and config.cnn_pretrained,
                          use_positional_encoding=config.use_positional_encoding,
                          transformer_layers=config.transformer_layers,
                          transformer_heads=config.transformer_heads,
                          input_channels=1 if config.grayscale else 3, local_dropout=config.local_dropout,
                          transformer_dropout=config.transformer_dropout, rtl=config.rtl,
                          fusion_mode=config.fusion_mode,
                          use_gated_fusion=config.use_gated_fusion,
                          cnn_layers=config.cnn_layers)


def resolve_device(device='auto', local_rank=0):
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device(f'cuda:{local_rank}')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device)


def gradient_stats(model):
    """Return compact gradient diagnostics and fail on the first bad gradient."""
    model = model.module if isinstance(model, DistributedDataParallel) else model
    total_sq = 0.0
    maximum = 0.0
    absolute_sum = 0.0
    value_count = 0
    grad_params = grad_none = zero_grad = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            grad_none += 1
            continue
        grad = parameter.grad.detach().float()
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f'Non-finite gradient detected: {name}')
        grad_params += 1
        norm = grad.norm(2).item()
        total_sq += norm * norm
        maximum = max(maximum, grad.abs().max().item())
        absolute_sum += grad.abs().sum().item()
        value_count += grad.numel()
        if norm == 0.:
            zero_grad += 1

    def block_norm(prefixes):
        total = 0.
        found = False
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
                value = parameter.grad.detach().float().norm(2).item()
                total += value * value
                found = True
        return total ** .5 if found else None

    return dict(global_norm=total_sq ** .5, grad_max=maximum,
                grad_mean=absolute_sum / value_count if value_count else 0.,
                grad_params=grad_params, grad_none=grad_none, grad_zero=zero_grad,
                grad_nonfinite=0, cnn=block_norm(('cnn',)),
                transformer=block_norm(('transformer',)), local_projection=block_norm(('cnn.projection',)),
                fusion=block_norm(('fusion', 'fusion_norm')))


def _format_value(value):
    return 'n/a' if value is None else f'{value:.4f}'


def _batch_postfix(stats, optimizer=None, gradients=None, device=None):
    values = dict(loss=_format_value(stats['total']), pos=_format_value(stats['positive_cost']),
                  sigreg=_format_value(stats['sigreg']),
                  evaluated=stats['evaluated'], skipped=stats['skipped'],
                  windows=_format_value(stats.get('mean_windows')), letters=_format_value(stats.get('mean_letters')))
    if stats.get('negative_training_enabled', stats.get('ranking_candidates', 0) > 0):
        count = stats['ranking_candidates']
        values.update(negCost=_format_value(stats['negative_cost_sum'] / count if count else None),
                      hardNeg=_format_value(stats['hard_cost_sum'] / stats['hard_count'] if stats['hard_count'] else None),
                      absNeg=_format_value(stats['absolute_negative_loss']),
                      rank=_format_value(stats['ranking_loss']),
                      targetOK=f'{100 * stats["target_success"] / count:.0f}%' if count else 'n/a')
    elif stats.get('negative_warmup'):
        values['negatives'] = 'warmup'
    if optimizer is not None:
        values['lr'] = f'{optimizer.param_groups[0]["lr"]:.2e}'
    if gradients:
        values.update(g=f'{gradients["global_norm"]:.3f}', cnn=_format_value(gradients['cnn']),
                      tr=_format_value(gradients['transformer']), fus=_format_value(gradients['fusion']))
    if device is not None and device.type == 'cuda':
        values['mem'] = f'{torch.cuda.memory_allocated(device) / 1024 ** 3:.1f}G'
    return values


def build_loaders(dataset, config, split_ids=None):
    loaders = create_dataloaders(dataset, config.dataset_type, config.train_ratio, config.val_ratio,
                              config.test_ratio, config.split_mode, config.split_seed,
                              config.batch_size, config.num_workers, config.augmentation,
                              split_ids=split_ids, paired=config.paired,
                              image_size=(config.image_height,config.image_width),
                              grayscale=config.grayscale, crop=config.crop, binarize=config.binarize)
    for loader in loaders[:2]:
        loader.dataset.negative_config = config if config.negative_dtw_weight else None
    return loaders


def _batch_lines(batch, device):
    images, texts = batch['image'], list(batch['text'])
    if torch.is_tensor(batch['image2']):
        images = torch.cat((images,batch['image2']))
        texts.extend(batch['text2'])
    return images.to(device), texts


def _embedding_diagnostics(output, texts, text_encoder):
    """Detached cosine and representation diagnostics for a batch."""
    values = {key: [] for key in ('positive_similarity', 'negative_similarity',
              'similarity_min', 'similarity_max', 'similarity_mean', 'similarity_std',
              'local_vector_norm', 'contextual_vector_norm', 'fused_vector_norm')}
    with torch.no_grad():
        for index, raw_text in enumerate(texts):
            letters = clean_letters(raw_text)
            valid = output['token_valid'][index]
            if not letters or not valid.any():
                continue
            visual = output['fused'][index][valid].detach().float()
            own = text_encoder.encode(''.join(letters)).detach().float()
            matrix = visual @ own.T
            values['positive_similarity'].append(float(matrix.max(dim=0).values.mean()))
            values['similarity_min'].append(float(matrix.min()))
            values['similarity_max'].append(float(matrix.max()))
            values['similarity_mean'].append(float(matrix.mean()))
            values['similarity_std'].append(float(matrix.std(unbiased=False)))
            absent = ''.join(letter for letter in ARABIC_LETTERS if letter not in set(letters))
            if absent:
                other = text_encoder.encode(absent).detach().float()
                values['negative_similarity'].append(float((visual @ other.T).max(dim=1).values.mean()))
            for key, tensor_key in (('local_vector_norm', 'local'),
                                    ('contextual_vector_norm', 'context'),
                                    ('fused_vector_norm', 'fused')):
                values[key].append(float(output[tensor_key][index][valid].detach().float().norm(dim=-1).mean()))
        gate = output.get('fusion_gate')
        gate_stats = None
        if gate is not None:
            gate = gate[output['token_valid']].detach().float()
            gate_stats = dict(mean=float(gate.mean()), std=float(gate.std(unbiased=False)),
                              min=float(gate.min()), max=float(gate.max()))
    return values, gate_stats


def _run_epoch(model, text_encoder, loader, config, device, optimizer=None, max_batches=0,
               epoch=None, epochs=None, rank=0):
    training = optimizer is not None
    validate_objective(config)
    if hasattr(loader.dataset, 'set_epoch'):
        loader.dataset.set_epoch((epoch or 1) if training else 0)
    active_config = (replace(config, negative_dtw_weight=0.) if training and
                     (epoch or 1) <= config.negative_warmup_epochs else config)
    model.train(training)
    text_encoder.eval()
    distributed = training and dist.is_available() and dist.is_initialized()
    # DTW is aggregated per LINE, including both sides, excluding empty text.
    # SIGReg is a token-weighted batch-population statistic, not a per-line loss.
    totals = torch.zeros(23,dtype=torch.float32 if device.type == 'mps' else torch.float64,
                         device=device)
    corruption, skip_reasons = Counter(), Counter()
    negative_cost_min, negative_cost_max = float('inf'), float('-inf')
    positive_cost_min, positive_cost_max = float('inf'), float('-inf')
    hard_cost_min = float('inf')
    started = time.monotonic()
    gradient_batches = []
    diagnostics = []
    gates = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    batch_total = min(len(loader), max_batches) if max_batches else len(loader)
    phase = 'TRAIN' if training else 'VAL'
    progress = tqdm(loader, desc=f'Epoch {epoch}/{epochs} {phase}' if epoch else phase,
                    total=batch_total, dynamic_ncols=True, leave=True, disable=rank != 0)
    for index,batch in enumerate(progress):
        if max_batches and index >= max_batches:
            break
        images,texts = _batch_lines(batch,device)
        negative_texts = negative_metadata = None
        if active_config.negative_dtw_weight:
            nested = batch.get('negative_texts')
            if nested is None:
                raise ValueError('Negative training requires a dataset view configured by build_loaders')
            negative_texts = [sides[0] for sides in nested]
            meta_nested = batch.get('negative_metadata')
            negative_metadata = [sides[0] for sides in meta_nested] if meta_nested is not None else None
            if torch.is_tensor(batch['image2']):
                negative_texts += [sides[1] for sides in nested]
                if negative_metadata is not None:
                    negative_metadata += [sides[1] for sides in meta_nested]
        for sides in batch.get('negative_stats', []):
            for info in sides:
                corruption.update({k: info.get(k, 0) for k in ('requested','generated','shortfall','attempts')})
                corruption.update({'rejected_'+k: v for k,v in info['rejections'].items()})
                corruption.update({'operation_'+k: v for k,v in info['operations'].items()})
        for sides in batch.get('negative_metadata', []):
            for entries in sides:
                corruption['corruption_ratio_sum'] += sum(item['corruption_ratio'] for item in entries)
                corruption['corruption_ratio_count'] += len(entries)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.use_amp and device.type=='cuda'):
                output = model(images)
            loss,stats = compute_loss(output,texts,text_encoder,active_config, negative_texts=negative_texts,
                                     negative_metadata=negative_metadata,
                                     distributed_statistics=distributed,
                                     sketch_seed=None if training else config.seed)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective; no optimizer step taken')
            if training and not stats['global_evaluated']:
                raise ValueError(f'No feasible positive transcripts in the global batch; no step taken: {stats["skip_reasons"]}')
            if training and active_config.negative_dtw_weight and not stats['global_ranking_candidates']:
                raise ValueError('Negative training is enabled, but this batch produced no valid negative alignments; no step taken')
            if training:
                loss.backward()
                gradients = gradient_stats(model)
                gradient_batches.append(gradients)
                if gradients['global_norm'] == 0. and rank == 0:
                    tqdm.write(f'WARNING: global gradient norm is zero at epoch {epoch} batch {index + 1}')
                optimizer.step()
            else:
                gradients = None
            batch_diagnostics, gate_stats = _embedding_diagnostics(output, texts, text_encoder)
            diagnostics.append(batch_diagnostics)
            if gate_stats is not None:
                gates.append(gate_stats)
        tokens = stats['valid_tokens']
        skip_reasons.update(stats['skip_reasons'])
        skip_reasons.update({'negative_'+k: v for k,v in stats['negative_skip_reasons'].items()})
        if stats['skipped'] and rank == 0:
            tqdm.write(f'Skipped {stats["skipped"]} lines: {stats["skip_reasons"]}')
        evaluated = stats['evaluated']
        batch_stats = dict(stats, mean_windows=(sum(t for t, _ in stats['lengths']) / evaluated
                                                 if evaluated else None),
                           mean_letters=(sum(l for _, l in stats['lengths']) / evaluated
                                         if evaluated else None),
                           negative_training_enabled=bool(active_config.negative_dtw_weight),
                           negative_warmup=bool(config.negative_dtw_weight and not active_config.negative_dtw_weight))
        progress.set_postfix(_batch_postfix(batch_stats, optimizer, gradients, device))
        if stats['negative_cost_min'] is not None:
            negative_cost_min = min(negative_cost_min, stats['negative_cost_min'])
            negative_cost_max = max(negative_cost_max, stats['negative_cost_max'])
        if stats['positive_cost_min'] is not None:
            positive_cost_min = min(positive_cost_min, stats['positive_cost_min'])
            positive_cost_max = max(positive_cost_max, stats['positive_cost_max'])
        if stats['hard_cost_min'] is not None:
            hard_cost_min = min(hard_cost_min, stats['hard_cost_min'])
        totals += totals.new_tensor([stats['dtw_sum'],stats['evaluated'],stats['skipped'],
                                      (stats['sigreg'] or 0.)*tokens,tokens,
                                      sum(t for t,l in stats['lengths']),
                                      sum(l for t,l in stats['lengths']),1,
                                      stats['objective_sum'], stats['ranked_lines'], stats['ranking_candidates'],
                                      stats['ranking_correct'], stats['ranking_margin_sum'],
                                      stats['negative_cost_sum'], stats['margin_violations'],
                                      stats['absolute_sum'], stats['ranking_sum'], stats['wrong_sum'],
                                      stats['hard_cost_sum'], stats['hard_count'], stats['target_sum'],
                                      stats['target_success'],
                                      stats['positive_below_reference']])
    progress.close()
    if distributed:
        dist.all_reduce(totals)
        details = [None] * dist.get_world_size()
        dist.all_gather_object(details, (dict(corruption), dict(skip_reasons),
                                         negative_cost_min, negative_cost_max,
                                         positive_cost_min, positive_cost_max, hard_cost_min))
        corruption, skip_reasons = Counter(), Counter()
        negative_cost_min, negative_cost_max = float('inf'), float('-inf')
        positive_cost_min, positive_cost_max, hard_cost_min = float('inf'), float('-inf'), float('inf')
        for c, s, low, high, pos_low, pos_high, hard_low in details:
            corruption.update(c); skip_reasons.update(s)
            negative_cost_min = min(negative_cost_min, low)
            negative_cost_max = max(negative_cost_max, high)
            positive_cost_min = min(positive_cost_min, pos_low)
            positive_cost_max = max(positive_cost_max, pos_high)
            hard_cost_min = min(hard_cost_min, hard_low)
    (dtw_sum,n,skipped,sig_sum,tokens,windows,letters,batches,neg_sum,ranked,
     candidates,correct,margins,negative_cost_sum,margin_violations,absolute_sum,
     ranking_sum,wrong_sum,hard_cost_sum,hard_count,target_sum,target_success,
     positive_below_reference) = totals.tolist()
    positive = dtw_sum/n if n else None
    sigreg = sig_sum/tokens if tokens and config.sigreg_weight else None
    negative = neg_sum/ranked if ranked else None
    total = None if positive is None else (config.positive_dtw_weight*positive + config.sigreg_weight*(sigreg or 0.)
                                           + active_config.negative_dtw_weight*(negative or 0.)
                                           + active_config.ranking_aux_weight*(ranking_sum/ranked if ranked else 0.)
                                           + active_config.wrong_letter_unlikelihood_weight*(wrong_sum/ranked if ranked else 0.))
    result = dict(total=total,total_loss=total,positive_dtw=positive,positive_cost=positive,
                positive_cost_mean=positive,
                positive_cost_min=positive_cost_min if n else None,
                positive_cost_max=positive_cost_max if n else None,
                negative_objective=negative,sigreg=sigreg,
                negative_cost_mean=negative_cost_sum/candidates if candidates else None,
                negative_cost_min=negative_cost_min if candidates else None,
                negative_cost_max=negative_cost_max if candidates else None,
                hard_negative_cost_mean=hard_cost_sum/hard_count if hard_count else None,
                hard_negative_cost_min=hard_cost_min if hard_count else None,
                negative_target_mean=target_sum/candidates if candidates else None,
                negative_target_success_rate=target_success/candidates if candidates else None,
                fraction_negative_above_target=target_success/candidates if candidates else None,
                fraction_positive_below_reference=positive_below_reference/n if n else None,
                absolute_negative_loss=absolute_sum/ranked if ranked else None,
                ranking_loss=ranking_sum/ranked if ranked else None,
                wrong_letter_loss=wrong_sum/ranked if ranked else None,
                hard_negatives=int(margin_violations),
                ranking_margin_violations=int(margin_violations),
                hard_negative_count=int(hard_count),
                negative_training_enabled=bool(active_config.negative_dtw_weight),
                negative_loss_type=config.negative_loss_type,
                dtw_normalization=config.dtw_normalization,
                negative_weight=active_config.negative_dtw_weight, ranked_lines=int(ranked),
                ranking_candidates=int(candidates), ranking_accuracy=correct/candidates if candidates else None,
                negative_gap_mean=margins/candidates if candidates else None,
                negative_minus_positive_cost=margins/candidates if candidates else None,
                corruption=dict(corruption),
                mean_corruption_ratio=(corruption['corruption_ratio_sum'] / corruption['corruption_ratio_count']
                                       if corruption['corruption_ratio_count'] else None),
                generator_severity=config.negative_severity,
                skip_reasons=dict(skip_reasons), alignment_objective=config.alignment_objective,
                weighted_sigreg=config.sigreg_weight*(sigreg or 0.),evaluated=int(n),skipped=int(skipped),
                valid_tokens=int(tokens),batches=int(batches),mean_windows=windows/n if n else None,
                mean_letters=letters/n if n else None,gamma=config.dtw_gamma,
                population='explicit-batch-subset' if max_batches else 'full-split',
                seconds=time.monotonic()-started)
    # Old machine-readable keys remain for ranking-mode consumers only. New
    # progress and summaries always use the unambiguous objective/cost names.
    if config.negative_loss_type == 'ranking':
        result.update(negative_dtw=negative, negative_dtw_mean=result['negative_cost_mean'],
                      negative_dtw_min=result['negative_cost_min'],
                      negative_dtw_max=result['negative_cost_max'], negative_margin_loss=result['ranking_loss'])
    for key in diagnostics[0] if diagnostics else ():
        values = [value for batch in diagnostics for value in batch[key]]
        result[key] = sum(values) / len(values) if values else None
    positive_similarity = result.get('positive_similarity')
    negative_similarity = result.get('negative_similarity')
    result['similarity_margin'] = (positive_similarity - negative_similarity
                                   if positive_similarity is not None and negative_similarity is not None else None)
    result['gate_mean'] = sum(item['mean'] for item in gates) / len(gates) if gates else None
    result['gate_std'] = sum(item['std'] for item in gates) / len(gates) if gates else None
    result['gate_min'] = min((item['min'] for item in gates), default=None)
    result['gate_max'] = max((item['max'] for item in gates), default=None)
    result['learning_rate'] = optimizer.param_groups[0]['lr'] if optimizer else config.learning_rate
    result['gpu_memory_mb'] = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                               if device.type == 'cuda' else None)
    if training:
        def mean(name):
            values = [item[name] for item in gradient_batches if item[name] is not None]
            return sum(values) / len(values) if values else 0.
        result['gradients'] = dict(global_mean=mean('global_norm'),
                                   global_min=min((item['global_norm'] for item in gradient_batches), default=0.),
                                   global_max=max((item['global_norm'] for item in gradient_batches), default=0.),
                                   cnn_mean=mean('cnn'), transformer_mean=mean('transformer'),
                                   fusion_mean=mean('fusion'), grad_max=mean('grad_max'),
                                   grad_mean=mean('grad_mean'), grad_params=mean('grad_params'),
                                   grad_none=mean('grad_none'), grad_zero=mean('grad_zero'),
                                   grad_nonfinite=mean('grad_nonfinite'))
    return result


def train_one_epoch(model,text_encoder,loader,optimizer,config,device,max_batches=0,epoch=None,epochs=None,rank=0):
    return _run_epoch(model,text_encoder,loader,config,torch.device(device),optimizer,max_batches,epoch,epochs,rank)


def validate_one_epoch(model,text_encoder,loader,config,device,max_batches=0,epoch=None,epochs=None,rank=0):
    """Local unwrapped evaluation: no DDP collectives, updates or RNG side effects."""
    if isinstance(model,DistributedDataParallel):
        raise ValueError('Coordinated rank-zero validation requires model.module')
    if loader.dataset.augment:
        raise ValueError('Validation must use an augmentation-free dataset view')
    previous_evaluation_view = getattr(loader.dataset,'evaluation_view',False)
    if hasattr(loader.dataset,'evaluation_view'): loader.dataset.evaluation_view=True
    device = torch.device(device)
    modes = [(module,module.training) for module in list(model.modules())+list(text_encoder.modules())]
    python_rng,numpy_rng = random.getstate(),np.random.get_state()
    loader_rng = loader.generator.get_state() if loader.generator is not None else None
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []), torch.no_grad():
            return _run_epoch(model,text_encoder,loader,config,device,max_batches=max_batches,
                              epoch=epoch,epochs=epochs,rank=rank)
    finally:
        if hasattr(loader.dataset,'evaluation_view'): loader.dataset.evaluation_view=previous_evaluation_view
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
                   letter_evidence_prior=getattr(text_encoder, 'letter_evidence_prior', None),
                   initialization=raw.cnn.initialization,**metadata)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def load_checkpoint(path,device='cpu'):
    """Strict new-format reconstruction. Legacy checkpoints are not migrated."""
    saved = torch.load(path,map_location='cpu',weights_only=False)  # trusted project checkpoints include RNG metadata
    if saved.get('format_version') != 1 or saved.get('architecture_family') != 'simple-alignment-core':
        raise ValueError('Incompatible legacy checkpoint: expected standalone simple-alignment-core format 1')
    config = Config(**saved['config'])
    model = build_model(config,initialize=False).to(device)
    state = dict(saved['model'])
    # Checkpoints saved before the concat projection was named: migrate the old
    # bare fusion Linear keys only when shapes match exactly (same tensors, new names).
    if 'fusion.weight' in state and 'fusion.concat_projection.weight' not in state:
        projection = model.fusion.concat_projection
        if projection is None or tuple(state['fusion.weight'].shape) != tuple(projection.weight.shape):
            raise ValueError('Checkpoint has legacy fusion.weight but no shape-matching concat projection')
        state['fusion.concat_projection.weight'] = state.pop('fusion.weight')
        state['fusion.concat_projection.bias'] = state.pop('fusion.bias')
    model.load_state_dict(state,strict=True)
    model.cnn.initialization = saved['initialization']
    text = OrthogonalCharEmbedding(config.embedding_dim,config.text_vocab_size,config.text_embedding_seed).to(device)
    text.load_state_dict(saved['text_embedding'],strict=True)
    text.letter_evidence_prior = saved.get('letter_evidence_prior')
    if hashlib.sha256(json.dumps(saved['split_ids'],sort_keys=True).encode()).hexdigest() != saved['split_manifest_sha256']:
        raise ValueError('Checkpoint split identity is inconsistent')
    model.eval()
    text.eval()
    return model,text,config,saved


def _print_dataset_summary(args, config, loaders):
    train_loader, val_loader, test_loader = loaders
    print('=' * 60, flush=True)
    print('DATASET', flush=True)
    print('=' * 60, flush=True)
    print(f'Path: {Path(args.dataset).resolve()}', flush=True)
    print(f'Type: {train_loader.dataset.dataset_type}', flush=True)
    print(f'Total samples: {sum(len(loader.dataset) for loader in loaders)}', flush=True)
    for name, loader in (('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)):
        print(f'{name}:', flush=True)
        print(f'  samples: {len(loader.dataset)}', flush=True)
        print(f'  batches: {len(loader)}', flush=True)
    print(f'Split mode: {config.split_mode}', flush=True)
    print(f'Split ratios: {config.train_ratio:.2f} / {config.val_ratio:.2f} / {config.test_ratio:.2f}', flush=True)
    print('Augmentation:', flush=True)
    print(f'  train: {"ON" if train_loader.dataset.augment else "OFF"}', flush=True)
    print(f'  validation: {"ON" if val_loader.dataset.augment else "OFF"}', flush=True)
    print(f'  test: {"ON" if test_loader.dataset.augment else "OFF"}', flush=True)
    print(f'Image size: {config.image_height}x{config.image_width}', flush=True)
    print(f'Window width: {config.window_width}', flush=True)
    print(f'Window stride: {config.window_stride}', flush=True)
    print(f'CNN type: {config.cnn_type}', flush=True)
    if config.cnn_type == 'simple':
        print(f'CNN layers: {config.cnn_layers}', flush=True)
        print(f'CNN channels: {simple_cnn_channels(config.cnn_layers)}', flush=True)
    else:
        print(f'CNN layers: fixed {config.cnn_type} architecture', flush=True)
    print(f'CNN output dimension: {config.embedding_dim}', flush=True)
    print(f'Transformer: {config.transformer_type}', flush=True)
    print(f'Embedding dim: {config.embedding_dim}', flush=True)
    print(f'SIGReg: {"ENABLED (weight=" + str(config.sigreg_weight) + ")" if config.sigreg_weight else "DISABLED"}', flush=True)
    print(f'Positive DTW weight: {config.positive_dtw_weight}', flush=True)
    print(f'Negative DTW weight: {config.negative_dtw_weight}', flush=True)
    print('Negative transcripts: ' + ('ENABLED' if config.negative_dtw_weight else 'DISABLED'), flush=True)
    print(f'Negative count: {config.negative_count}', flush=True)
    print(f'Negative margin: {config.negative_margin}', flush=True)
    print(f'Negative severity: {config.negative_severity}', flush=True)
    print(f'Negative operations: {config.negative_operations}', flush=True)
    print(f'Negative warmup epochs: {config.negative_warmup_epochs}', flush=True)
    print(f'Negative curriculum epochs: {config.negative_curriculum_epochs}', flush=True)
    print(f'Negative seed: {config.negative_seed}', flush=True)
    print(f'Negative loss type: {config.negative_loss_type}', flush=True)
    print(f'Hard negative k: {config.hard_negative_k}', flush=True)
    print(f'DTW normalization: {config.dtw_normalization}', flush=True)
    print(f'Negative target min/max: {config.negative_target_min} / {config.negative_target_max}', flush=True)
    print(f'Negative softness: {config.negative_softness}', flush=True)
    print(f'Ranking auxiliary weight: {config.ranking_aux_weight}', flush=True)
    print(f'Wrong-letter unlikelihood weight: {config.wrong_letter_unlikelihood_weight}', flush=True)
    print(f'Alignment objective: {config.alignment_objective}; cost mode: {config.dtw_cost_mode}; '
          f'temperature: {config.competition_temperature}; gamma: {config.dtw_gamma}; '
          f'position prior: {config.position_prior}; vertical/horizontal: '
          f'{config.vertical_penalty}/{config.horizontal_penalty}', flush=True)
    print('=' * 60, flush=True)


def _print_runtime(device, world, rank, config):
    print('=' * 60, flush=True)
    print('RUNTIME', flush=True)
    print('=' * 60, flush=True)
    print(f'World size: {world}', flush=True)
    print(f'Rank: {rank}', flush=True)
    print(f'CUDA available: {torch.cuda.is_available()}', flush=True)
    print(f'Visible GPUs: {torch.cuda.device_count()}', flush=True)
    print(f'Device: {device}', flush=True)
    print(f'Fusion mode: {config.fusion_mode}', flush=True)
    print(f'Gated fusion: {"enabled" if config.use_gated_fusion else "disabled"}', flush=True)
    for index in range(torch.cuda.device_count()):
        print(f'GPU {index}: {torch.cuda.get_device_name(index)}', flush=True)
    print(f'AMP: {"ON" if config.use_amp and device.type == "cuda" else "OFF"}', flush=True)
    print('=' * 60, flush=True)


def _print_epoch_summary(epoch, epochs, train_stats, val_stats, gradients, previous_best, improved,
                         latest_path, best_path):
    def value(item):
        return 'n/a' if item is None else f'{item:.4f}'
    def objective_details(stats):
        print(f'  positive cost min/max: {value(stats["positive_cost_min"])} / '
              f'{value(stats["positive_cost_max"])}', flush=True)
        print(f'  negative cost mean/min/max: {value(stats["negative_cost_mean"])} / '
              f'{value(stats["negative_cost_min"])} / {value(stats["negative_cost_max"])}', flush=True)
        print(f'  hard negative cost mean/min: {value(stats["hard_negative_cost_mean"])} / '
              f'{value(stats["hard_negative_cost_min"])}', flush=True)
        print(f'  target mean / fraction above: {value(stats["negative_target_mean"])} / '
              f'{value(stats["fraction_negative_above_target"])}', flush=True)
        print(f'  absolute negative loss: {value(stats["absolute_negative_loss"])}', flush=True)
        print(f'  ranking loss/accuracy/gap: {value(stats["ranking_loss"])} / '
              f'{value(stats["ranking_accuracy"])} / {value(stats["negative_gap_mean"])}', flush=True)
        print(f'  positive below target-min reference: {value(stats["fraction_positive_below_reference"])}', flush=True)
        c = stats['corruption']
        print(f'  generation requested/generated/shortfall: '
              f'{c.get("requested", 0)} / {c.get("generated", 0)} / {c.get("shortfall", 0)}', flush=True)
        print(f'  corruption severity / observed ratio: {value(stats["generator_severity"])} / '
              f'{value(stats["mean_corruption_ratio"])}', flush=True)
        print('  operation counts: ' + ', '.join(
            f'{op}={c.get("operation_" + op, 0)}' for op in
            ('substitute', 'adjacent', 'blocks', 'words', 'shift', 'shuffle')), flush=True)
        rejected = {key: amount for key, amount in c.items() if key.startswith('rejected_')}
        if rejected:
            print(f'  rejection reasons: {rejected}', flush=True)
    print('=' * 60, flush=True)
    print(f'EPOCH {epoch}/{epochs} SUMMARY', flush=True)
    print('', flush=True)
    print('TRAIN', flush=True)
    print(f'  total loss:        {value(train_stats["total"])}', flush=True)
    print(f'  positive DTW:      {value(train_stats["positive_dtw"])}', flush=True)
    objective_details(train_stats)
    print(f'  SIGReg:            {value(train_stats["sigreg"]) if train_stats["sigreg"] is not None else "disabled"}', flush=True)
    print(f'  evaluated lines:   {train_stats["evaluated"]}', flush=True)
    print(f'  skipped lines:     {train_stats["skipped"]}', flush=True)
    print(f'  time:              {train_stats["seconds"]:.1f} s', flush=True)
    print('', flush=True)
    print('GRADIENTS', flush=True)
    for label, key in (('mean global norm', 'global_mean'), ('min global norm', 'global_min'),
                       ('max global norm', 'global_max'), ('CNN mean norm', 'cnn_mean'),
                       ('Transformer mean', 'transformer_mean'), ('Fusion mean', 'fusion_mean')):
        print(f'  {label + ":":19}{gradients[key]:.4f}', flush=True)
    print('', flush=True)
    print('VALIDATION', flush=True)
    print(f'  total loss:        {value(val_stats["total"])}', flush=True)
    print(f'  positive DTW:      {value(val_stats["positive_dtw"])}', flush=True)
    objective_details(val_stats)
    print(f'  evaluated lines:   {val_stats["evaluated"]}', flush=True)
    print(f'  skipped lines:     {val_stats["skipped"]}', flush=True)
    print(f'  time:              {val_stats["seconds"]:.1f} s', flush=True)
    print('', flush=True)
    print('BEST VALIDATION', flush=True)
    print(f'  previous:          {"n/a" if previous_best == float("inf") else value(previous_best)}', flush=True)
    print(f'  current:           {value(val_stats["total"])}', flush=True)
    print(f'  improved:          {"YES" if improved else "NO"}', flush=True)
    print('', flush=True)
    print('CHECKPOINT', flush=True)
    print(f'  latest: {latest_path}', flush=True)
    print(f'  best:   {best_path}', flush=True)
    print('=' * 60, flush=True)


def main(argv=None, epoch_callback=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--output-root',default='Weights')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--max-batches',type=int,default=0,help='Explicit smoke cap on BOTH passes; 0=full splits')
    parser.add_argument('--resume',help='New-format checkpoint; config must match (epochs may increase)')
    parser.add_argument('--finetune',help='Load weights and saved splits, fresh optimizer; explicitly allow changed loss settings')
    for field in fields(Config):
        kwargs = dict(default=field.default)
        if isinstance(field.default,bool): kwargs['action']=argparse.BooleanOptionalAction
        else: kwargs['type']=type(field.default)
        parser.add_argument('--'+field.name.replace('_','-'),**kwargs)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    config = Config(**{f.name:getattr(args,f.name) for f in fields(Config)})
    negative_controls = {'--negative-count', '--negative-margin', '--negative-severity',
                         '--negative-operations', '--negative-seed', '--negative-warmup-epochs',
                         '--negative-curriculum-epochs', '--negative-loss-type',
                         '--negative-target-min', '--negative-target-max', '--negative-softness',
                         '--hard-negative-k', '--ranking-aux-weight',
                         '--wrong-letter-unlikelihood-weight'}
    if (config.negative_dtw_weight == 0
            and not any(arg.split('=', 1)[0] == '--negative-dtw-weight' for arg in raw_argv)
            and any(arg.split('=', 1)[0] in negative_controls for arg in raw_argv)):
        raise ValueError('Negative settings were supplied, but negative DTW weight is 0; '
                         'pass --negative-dtw-weight > 0 or explicitly pass 0 for a no-negative control')
    config.fusion_mode, gated = validate_fusion_config(config.fusion_mode, config.use_gated_fusion)
    config.use_gated_fusion = int(gated)
    validate_objective(config)
    if args.resume and args.finetune:
        raise ValueError('--resume and --finetune are mutually exclusive')
    if args.max_batches < 0 or Path(args.run_name).name != args.run_name:
        raise ValueError('Use a simple run name and nonnegative smoke cap')
    rank,world,local_rank = [int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')]]
    if world > 1 and args.device == 'auto' and not torch.cuda.is_available():
        raise RuntimeError('DDP requested multiple ranks, but CUDA is unavailable; refusing CPU fallback')
    if world > 1 and args.device == 'auto' and torch.cuda.device_count() < world:
        raise RuntimeError(f'DDP requested {world} ranks, but only {torch.cuda.device_count()} GPUs are visible')
    device = resolve_device(args.device, local_rank)
    if world > 1 and device.type != 'cuda':
        raise RuntimeError('Multi-rank training requires CUDA; refusing to start DDP on CPU')
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
    if args.resume or args.finetune:
        model,text,saved_config,saved = load_checkpoint(args.resume or args.finetune,device)
        expected,actual = asdict(saved_config),asdict(config)
        expected.pop('epochs')
        actual.pop('epochs')
        if args.resume:
            if expected != actual:
                raise ValueError('Resume configuration differs from checkpoint; use --finetune for changed losses')
            start,best = saved['epoch'],saved['best_val']
        else:
            allowed = {f.name for f in fields(Config) if f.name.startswith(('negative_', 'sigreg_'))}
            allowed |= {'alignment_objective','ctc_blank_logit','alphabet_inventory','position_prior',
                        'positive_dtw_weight','dtw_gamma','vertical_penalty','horizontal_penalty',
                        'competition_temperature','dtw_cost_mode','disable_horizontal_when_feasible',
                        'dtw_normalization','hard_negative_k','ranking_aux_weight',
                        'wrong_letter_unlikelihood_weight'}
            changed = {key for key in expected if expected[key] != actual[key]}
            if changed - allowed:
                raise ValueError(f'Fine-tuning may change losses only, not architecture/data: {sorted(changed-allowed)}')
    else:
        model = build_model(config).to(device)
        text = OrthogonalCharEmbedding(config.embedding_dim,config.text_vocab_size,config.text_embedding_seed).to(device)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loaders = build_loaders(args.dataset,config,saved['split_ids'] if saved else None)
    train_loader,val_loader,_test_loader = loaders  # Never iterate test during training.
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError('Training and validation need nonempty group splits')
    split_ids = {name:[r['sample_id'] for r in loader.dataset.records]
                 for name,loader in zip(('train','val','test'),loaders)}
    if not args.resume:
        text.letter_evidence_prior = fit_letter_prior(train_loader.dataset.records, config.alphabet_inventory)
        text.letter_evidence_prior['train_ids_sha256'] = hashlib.sha256(
            json.dumps(split_ids['train'], sort_keys=True).encode()).hexdigest()
    sampler = None
    if world>1:
        sampler = DistributedSampler(train_loader.dataset,seed=config.seed,shuffle=True)
        train_loader = DataLoader(train_loader.dataset,batch_size=config.batch_size,sampler=sampler,
                                  num_workers=config.num_workers,collate_fn=collate_samples)
        model = DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    if args.resume:
        optimizer.load_state_dict(saved['optimizer'])
        states = saved.get('rng_states', [])
        if len(states) == world:
            state = states[rank]
            random.setstate(state['python']); np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'])
            if device.type == 'cuda': torch.cuda.set_rng_state(state['cuda'], device)
            if train_loader.generator is not None and state['loader'] is not None:
                train_loader.generator.set_state(state['loader'])
        elif rank == 0:
            print('WARNING: legacy checkpoint lacks matching per-rank RNG states; resume is not bit-exact.', flush=True)
    history_path = output_dir/'history.json'
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []
    if rank==0:
        _print_dataset_summary(args, config, loaders)
        _print_runtime(device, world, rank, config)
        print('CONFIG',json.dumps(asdict(config),sort_keys=True),flush=True)
        print('SPLITS',{name:len(ids) for name,ids in split_ids.items()},flush=True)
        (output_dir/'split_manifest.json').write_text(json.dumps(split_ids,indent=2))
    elif world > 1:
        print(f'DDP rank {rank} -> {device}', flush=True)
    for epoch in range(start+1,config.epochs+1):
        if sampler: sampler.set_epoch(epoch)
        if rank == 0:
            print('=' * 60, flush=True)
            print(f'Epoch {epoch} / {config.epochs}', flush=True)
            print('=' * 60, flush=True)
        train_stats = train_one_epoch(model,text,train_loader,optimizer,config,device,args.max_batches,
                                      epoch,config.epochs,rank)
        if world>1: dist.barrier()
        result = [None]
        if rank==0:
            try:
                raw = model.module if world>1 else model
                val_stats = validate_one_epoch(raw,text,val_loader,config,device,args.max_batches,
                                               epoch,config.epochs,rank)
                if val_stats['total'] is None: raise ValueError('Validation has no nonempty cleaned transcripts')
                result[0] = dict(stats=val_stats)
            except Exception as exc:
                result[0] = dict(error=f'{type(exc).__name__}: {exc}')
        if world>1: dist.broadcast_object_list(result,src=0)
        if 'error' in result[0]: raise RuntimeError(result[0]['error'])
        val_stats = result[0]['stats']
        previous_best = best
        improved = val_stats['total'] < best
        best = min(best,val_stats['total'])
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                   loader=train_loader.generator.get_state() if train_loader.generator is not None else None)
        rng_states = [None] * world
        if world > 1: dist.all_gather_object(rng_states, rng)
        else: rng_states[0] = rng
        if rank==0:
            entry = dict(epoch=epoch,train=train_stats,validation=val_stats,
                         gradients=train_stats.get('gradients', {}),
                         model_parameter_count=model_parameter_count)
            history.append(entry)
            history_path.write_text(json.dumps(history,indent=2))
            kwargs = dict(selection_metric='validation.total',selection_population=val_stats['population'],
                          dataset=str(Path(args.dataset).resolve()),max_batches=args.max_batches,metrics=entry,
                          rng_states=rng_states, run_mode='finetune' if args.finetune else 'resume' if args.resume else 'fresh',
                          source_checkpoint=args.finetune or args.resume,
                          source_config=saved['config'] if saved else None)
            print('OBJECTIVE', json.dumps({phase: {k: stats[k] for k in ('negative_training_enabled',
                  'negative_objective','ranked_lines','ranking_candidates','ranking_accuracy',
                  'negative_gap_mean','negative_cost_mean','negative_cost_min',
                  'negative_cost_max','hard_negative_cost_mean','absolute_negative_loss',
                  'ranking_loss','negative_target_success_rate','corruption','skip_reasons')}
                  for phase,stats in [('train',train_stats),('validation',val_stats)]}), flush=True)
            save_checkpoint(output_dir/'checkpoint_latest.pt',model,optimizer,epoch,best,config,text,split_ids,**kwargs)
            if improved: save_checkpoint(output_dir/'checkpoint_best.pt',model,optimizer,epoch,best,config,text,split_ids,**kwargs)
            _print_epoch_summary(epoch, config.epochs, train_stats, val_stats, entry['gradients'],
                                 previous_best,
                                 improved, output_dir/'checkpoint_latest.pt', output_dir/'checkpoint_best.pt')
            if epoch_callback is not None:
                epoch_callback(entry)
        if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()
    return output_dir


if __name__=='__main__':
    main()

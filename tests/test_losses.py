from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from losses import compute_loss, positive_dtw_loss, negative_dtw_margin_loss, sigreg_loss
from parameters import Config
from text_embedding import OrthogonalCharEmbedding


def fixture():
    torch.manual_seed(3)
    raw = torch.randn(2,5,128,requires_grad=True)
    output = dict(fused=F.normalize(raw,dim=-1), fused_pre_l2=raw, token_valid=torch.ones(2,5,dtype=torch.bool))
    config = Config(sigreg_sketch_dim=16, sigreg_min_samples=2)
    return raw, output, OrthogonalCharEmbedding(vocab_size=4096), config


def test_individual_and_combined_backward():
    raw,output,text,config = fixture()
    positive,counts = positive_dtw_loss(output['fused'],['باب','سلام'],text,config)
    negative = negative_dtw_margin_loss(output['fused'],['باب','سلام'],[['سلام'],['باب']],text,config)
    assert torch.isfinite(positive) and torch.isfinite(negative) and counts['evaluated']==2
    loss,stats = compute_loss(output,['باب','سلام'],text,config,sketch_seed=123)
    assert stats['total'] == pytest.approx(stats['positive_dtw'] + .2 * stats['sigreg'])
    assert stats['negative_dtw'] is None
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert all(p.grad is None for p in text.parameters())


def test_sigreg_fixed_seed_valid_population_and_empty_text():
    raw,output,text,config = fixture()
    a = sigreg_loss(raw,sketch_dim=16,min_samples=2,seed=4)
    b = sigreg_loss(raw,sketch_dim=16,min_samples=2,seed=4)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    valid = output['token_valid'].clone();valid[1]=False
    torch.testing.assert_close(sigreg_loss(raw,valid,sketch_dim=16,min_samples=2,seed=4),
                               sigreg_loss(raw[:1],sketch_dim=16,min_samples=2,seed=4))
    loss,stats = compute_loss(output,['','باب'],text,replace(config,sigreg_weight=0))
    assert stats['evaluated']==1 and stats['skipped']==1 and stats['sigreg'] is None
    assert torch.isfinite(loss)


def test_negative_requires_explicit_transcripts():
    _,output,text,config=fixture()
    with pytest.raises(ValueError,match='explicit'):
        compute_loss(output,['باب','سلام'],text,replace(config,negative_dtw_weight=1))

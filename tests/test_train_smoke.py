import pytest
import torch

from dataloader import create_dataloaders
from parameters import Config
from text_embedding import OrthogonalCharEmbedding
from train import build_model, load_checkpoint, main, train_one_epoch, validate_one_epoch
from test_dataset import make_synthetic


def tiny_config():
    return Config(cnn_type='simple',cnn_pretrained=False,image_height=32,image_width=64,
                  batch_size=3,num_workers=0,sigreg_sketch_dim=8,sigreg_min_samples=2)


def test_one_epoch_checkpoint_reload_and_automatic_validation(tmp_path,capsys):
    root=make_synthetic(tmp_path/'data')
    output=main(['--dataset',str(root),'--run-name','tiny','--output-root',str(tmp_path/'weights'),
                 '--epochs','1','--cnn-type','simple','--no-cnn-pretrained','--num-workers','0',
                 '--image-height','32','--image-width','64','--batch-size','3','--sigreg-sketch-dim','8'])
    assert (output/'checkpoint_latest.pt').exists() and (output/'checkpoint_best.pt').exists()
    model,text,config,saved=load_checkpoint(output/'checkpoint_latest.pt')
    assert saved['epoch']==1 and saved['metrics']['validation']['evaluated']==2
    assert model(torch.randn(2,1,32,64))['fused'].shape==(2,3,128)
    console=capsys.readouterr().out
    assert 'Train loss:' in console and 'Validation loss:' in console
    saved['config']['embedding_dim']=64
    torch.save(saved,tmp_path/'bad.pt')
    with pytest.raises(RuntimeError,match='size mismatch'):
        load_checkpoint(tmp_path/'bad.pt')


def test_validation_freezes_parameters_rng_and_counts(tmp_path):
    config=tiny_config()
    loaders=create_dataloaders(make_synthetic(tmp_path/'data'),num_workers=0,batch_size=4,
                               train_ratio=.6,val_ratio=.3,test_ratio=.1,image_size=(32,64))
    model=build_model(config); text=OrthogonalCharEmbedding(vocab_size=4096)
    before={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    train_one_epoch(model,text,loaders[0],optimizer,config,'cpu')
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    before={k:v.clone() for k,v in model.state_dict().items()}; rng=torch.get_rng_state()
    first=validate_one_epoch(model,text,loaders[1],config,'cpu')
    second=validate_one_epoch(model,text,loaders[1],config,'cpu')
    assert first['evaluated']==6 and first['positive_dtw']==second['positive_dtw']
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert model.training
    from pathlib import Path
    Path(loaders[1].dataset.records[0]['sides'][0]['text']).write_text('')
    stats=validate_one_epoch(model,text,loaders[1],config,'cpu')
    assert stats['evaluated']==5 and stats['skipped']==1


def test_resnet_validation_keeps_batchnorm_buffers(tmp_path):
    from dataclasses import replace
    config=replace(tiny_config(),cnn_type='resnet18',image_height=128)
    loaders=create_dataloaders(make_synthetic(tmp_path/'data'),num_workers=0,batch_size=2,image_size=(128,64))
    model=build_model(config).train()
    text=OrthogonalCharEmbedding(vocab_size=4096)
    before={k:v.clone() for k,v in model.named_buffers()}
    validate_one_epoch(model,text,loaders[1],config,'cpu')
    assert before and all(torch.equal(before[k],v) for k,v in model.named_buffers())

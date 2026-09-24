import numpy as np
import pytest
from PIL import Image

from evaluate import shared_regions, source_interval, region_mask, score_mask, evaluate_pair, evaluate_loss
from train import main as train_main
from test_dataset import make_synthetic


def match(c,**kwargs):
    return shared_regions(c,np.arange(c.shape[0])[::-1],np.arange(c.shape[1])[::-1],**kwargs)


def test_off_diagonal_multiple_regions_and_empty_background():
    c=np.zeros((30,30))
    for a,b in [(i,i+4) for i in range(2,8)]+[(i,i+3) for i in range(18,24)]: c[a,b]=.95
    result=match(c)
    assert len(result['regions'])==2
    assert result['regions'][0]['pairs']==[(i,i+4) for i in range(2,8)]
    assert not match(np.full((15,15),.9))['regions']
    assert not match(np.zeros((15,15)))['regions']
    c=np.eye(4)*.9
    assert not match(c)['regions']


def test_gaps_crossing_reuse_and_physical_support():
    c=np.zeros((24,24))
    for i in (1,2,3,5,6,7): c[i,i+2]=.95
    assert len(match(c)['regions'])==1
    assert not match(c,max_gap=0)['regions']
    c=np.zeros((25,25))
    for i in range(5): c[i+1,i+15]=.95;c[i+15,i+1]=.94
    regions=match(c)['regions']
    assert len(regions)==1
    with pytest.raises(ValueError,match='unique'):
        shared_regions(c,[0]*25,np.arange(25))


def test_inverse_geometry_and_full_height_masks(tmp_path):
    geo=dict(source_size=[400,80],crop=[100,10,300,70],scale_x=1024/200,model_size=[1024,128])
    assert source_interval(62,geo,32,16)==[293.75,300.]
    region=dict(supported_physical=[[62],[0]],filled_physical=[[],[]])
    mask=region_mask([region],0,geo,32,16)
    assert mask.shape==(80,400) and mask[:,293:300].all() and not mask[:,:293].any()
    assert not region_mask([],0,geo,32,16).any()
    Image.new('L',(1,1)).save(tmp_path/'wrong.png')
    with pytest.raises(ValueError,match='geometry mismatch'):score_mask(mask,tmp_path/'wrong.png')


def test_checkpoint_loss_and_pair_outputs(tmp_path):
    data=make_synthetic(tmp_path/'data')
    run=train_main(['--dataset',str(data),'--run-name','eval','--output-root',str(tmp_path/'weights'),
                    '--cnn-type','simple','--no-cnn-pretrained','--image-height','32','--image-width','128',
                    '--num-workers','0','--batch-size','4','--epochs','1','--sigreg-sketch-dim','8'])
    checkpoint=run/'checkpoint_best.pt'
    report=evaluate_loss(checkpoint,data,split='val')
    assert report['metrics']['evaluated']==2 and report['metrics']['total'] is not None
    output=tmp_path/'evaluation'
    pair=evaluate_pair(checkpoint,[data/'images/img1_0.png',data/'images/img2_0.png'],output)
    assert np.load(output/'cosine.npy').shape==(7,7)
    assert Image.open(output/'line_a_mask.png').size==(80,32)
    assert (output/'cosine_heatmap.png').exists() and (output/'metadata.json').exists()
    assert pair['metrics'][0]['status']=='unavailable'
    with pytest.raises(FileExistsError):
        evaluate_pair(checkpoint,[data/'images/img1_0.png',data/'images/img2_0.png'],output)

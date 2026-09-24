import json

import pytest
import torch

from dataloader import create_dataloaders
from test_dataset import make_synthetic, make_manifest


@pytest.mark.parametrize('ratios,counts', [((.8,.1,.1),(16,2,2)), ((.6,.2,.2),(12,4,4))])
def test_ratios_groups_and_augmentation(tmp_path, ratios, counts):
    root = make_synthetic(tmp_path)
    loaders = create_dataloaders(root, train_ratio=ratios[0], val_ratio=ratios[1],
                                test_ratio=ratios[2], num_workers=0)
    assert tuple(len(l.dataset) for l in loaders) == counts
    assert [l.dataset.augment for l in loaders] == [True, False, False]
    groups = [{r['group_id'] for r in l.dataset.records} for l in loaders]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    torch.testing.assert_close(loaders[1].dataset[0]['image'], loaders[1].dataset[0]['image'], rtol=0, atol=0)
    assert next(iter(loaders[0]))['image'].ndim == 4


def test_saved_membership_and_bad_ratios(tmp_path):
    root = make_synthetic(tmp_path)
    loaders = create_dataloaders(root, num_workers=0)
    ids = {name: [r['sample_id'] for r in l.dataset.records]
           for name,l in zip(('train','val','test'), loaders)}
    replay = create_dataloaders(root, num_workers=0, split_ids=ids, seed=99)
    assert [r['sample_id'] for r in replay[0].dataset.records] == ids['train']
    with pytest.raises(ValueError, match='sum to 1'):
        create_dataloaders(root, train_ratio=.9)
    ids['val'].append(ids['train'].pop())
    with pytest.raises(ValueError, match='group leakage'):
        create_dataloaders(root, split_ids=ids, num_workers=0)


def test_predefined_requires_labels(tmp_path):
    root = make_manifest(tmp_path)
    with pytest.raises(ValueError, match='predefined split'):
        create_dataloaders(root, split_mode='predefined', num_workers=0)
    manifest = root / 'dataset_manifest.jsonl'
    row = json.loads(manifest.read_text())
    row['split'] = 'valid'
    manifest.write_text(json.dumps(row) + '\n')
    loaders = create_dataloaders(root, split_mode='predefined', num_workers=0)
    assert [len(l.dataset) for l in loaders] == [0,2,0]

import json

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import AlignmentDataset, prepare_image, xml_crop_bounds


def make_synthetic(root, count=10):
    (root / 'images').mkdir(parents=True)
    (root / 'texts').mkdir()
    for i in range(count):
        for side in (1, 2):
            Image.new('L', (80, 32), 100 + i).save(root / f'images/img{side}_{i}.png')
            (root / f'texts/text{side}_{i}.txt').write_text('سلام', encoding='utf-8')
    return root


def make_manifest(root, bridge=False, anchor_index=False):
    root.mkdir(parents=True, exist_ok=True)
    for name in ('foo', 'bar'):
        Image.new('L', (80, 32), 180).save(root / f'{name}.png')
        (root / f'{name}.txt').write_text('سلام', encoding='utf-8')
    Image.new('L', (80, 32), 255).save(root / 'mask.png')
    a = dict(line_image_path='foo.png', text_original_path='bar.txt')
    b = dict(line_image_path='bar.png', text_original_path='foo.txt', alignment_mask_path='mask.png')
    row = dict(pair_id='p1', label_type='medium_match', A=a, B=b)
    if bridge:
        row['bridge'] = dict(anchor_id='anchor-1')
    if anchor_index:
        row = dict(anchor_id='anchor-1', real=dict(image='foo.png', text='bar.txt'),
                   positive=dict(image='bar.png', text='foo.txt', mask='mask.png'))
    path = root / ('anchor_index.jsonl' if anchor_index else 'dataset_manifest.jsonl')
    path.write_text(json.dumps(row) + '\n')
    return root


def make_book_lines(root, pages=10):
    """Small ArabicData layout with linked A/B pages and an uneven book size."""
    for book, lines_per_page in (('A', 3), ('B', 2)):
        side = root / f'Book{book}'
        (side / 'images').mkdir(parents=True)
        (side / 'texts').mkdir()
        (side / 'BoundingBoxes').mkdir()
        rows = []
        for page in range(pages):
            for line in range(lines_per_page):
                line_id = f'{book}_{page:02d}_{line:02d}'
                image = f'Book{book}/images/{line_id}.png'
                text = f'Book{book}/texts/{line_id}.txt'
                Image.new('L', (80, 32), 180).save(root / image)
                if not (book == 'B' and page == 0 and line == 0):
                    (root / text).write_text('سلام', encoding='utf-8')
                rows.append(dict(side=book, writer_id=f'writer-{book}', page_id=f'page-{page}',
                                 line_id=line_id, image_path=image, text_path=text, raw_text='سلام',
                                 bounding_boxes_path=f'Book{book}/BoundingBoxes/{line_id}.json',
                                 source_occurrences=[dict(pair_id=f'pair-{page}')]))
        (side / 'lines.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    return root


def test_synthetic(tmp_path):
    root = make_synthetic(tmp_path / 'synthetic')
    ds = AlignmentDataset(root)
    assert ds.dataset_type == 'synthetic' and len(ds) == 20
    assert ds[0]['image'].shape == (1, 128, 1024)
    assert ds[0]['image2'] is None
    pair = AlignmentDataset(root, paired=True)
    assert len(pair) == 10 and pair[0]['image2'].shape == (1, 128, 1024)


def test_arabic_data_book_lines(tmp_path):
    root = make_book_lines(tmp_path / 'ArabicData')
    ds = AlignmentDataset(root)
    assert ds.dataset_type == 'real' and ds.source_layout == 'book_lines'
    assert len(ds) == 50
    assert {r['book_id'] for r in ds.records} == {'A', 'B'}
    assert len({r['group_id'] for r in ds.records}) == 10
    b = next(i for i, r in enumerate(ds.records) if r['sample_id'] == 'BookB:B_00_00')
    sample = ds[b]
    assert sample['text'] == 'سلام'  # Manifest fallback for the absent text file.
    assert sample['book_id'] == 'B'
    assert sample['image'].shape == (1, 128, 1024)
    assert sample['geometry'][0]['crop'] == [0, 0, 80, 32]
    with pytest.raises(ValueError, match='independent book lines'):
        AlignmentDataset(root, paired=True)


@pytest.mark.parametrize('anchor_index', [False, True])
def test_manifest_and_bridge_common_keys(tmp_path, anchor_index):
    real = AlignmentDataset(make_manifest(tmp_path / 'real'))
    bridge = AlignmentDataset(make_manifest(tmp_path / 'bridge', True, anchor_index))
    assert len(real) == 2 and len(bridge) == 1
    assert bridge.dataset_type == 'real_synthetic'
    assert bridge[0]['group_id'] == bridge[0]['anchor_id'] == 'anchor-1'
    assert bridge[0]['mask'].shape == (1, 128, 1024)
    assert set(real[0]) == set(bridge[0])
    assert real[0]['text'] == 'سلام'  # Explicit different-stem binding retained.


def test_xml_crop_exact_and_normalization(tmp_path):
    side = tmp_path / 'A'
    (side / 'linesImages').mkdir(parents=True)
    Image.new('L', (400, 600), 255).save(side / 'original_image.png')
    (side / 'original.xml').write_text('<Root><DocumentElement><X>100</X><Y>100</Y>'
        '<Width>180</Width><Height>40</Height></DocumentElement></Root>')
    path = side / 'linesImages/line_01.png'
    Image.new('L', (400, 60), 255).save(path)
    assert xml_crop_bounds(path, (400, 60)) == (98, 8, 282, 52)
    pil, tensor, geometry = prepare_image(path)
    assert pil.mode == 'L' and pil.size == (1024, 128)
    assert geometry['crop'] == [98, 8, 282, 52]
    torch.testing.assert_close(tensor, torch.full_like(tensor, (1 - .449) / .226))


def test_rgb_and_explicit_bbox(tmp_path):
    path = tmp_path / 'line.png'
    Image.new('RGB', (100, 30), 'white').save(path)
    _, tensor, geo = prepare_image(path, grayscale=False, bbox=(10, 2, 90, 28))
    assert tensor.shape == (3, 128, 1024) and geo['crop'] == [10, 2, 90, 28]
    with pytest.raises(ValueError, match='outside source'):
        prepare_image(path, bbox=(-1, 0, 20, 30))


def test_missing_mask_geometry_fails(tmp_path):
    ds = AlignmentDataset(make_manifest(tmp_path), paired=True)
    Image.new('L', (5, 5)).save(tmp_path / 'mask.png')
    with pytest.raises(ValueError, match='Mask/source size'):
        ds[0]


def test_copied_pages_keep_distinct_transcripts_in_same_group(tmp_path):
    rows = []
    for copy in ('one', 'two'):
        sides = {}
        for side_name in ('A', 'B'):
            side = tmp_path / copy / side_name
            (side / 'linesImages').mkdir(parents=True)
            (side / 'text/final/original').mkdir(parents=True)
            Image.new('L', (80, 32), 255).save(side / 'original_image.png')
            image = side / 'linesImages/line_01.png'
            text = side / 'text/final/original/line_01.txt'
            Image.new('L', (80, 32), 150).save(image)
            text.write_text('سلام' if copy == 'one' else 'باب', encoding='utf-8')
            sides[side_name] = dict(line_image_path=str(image), text_original_path='ignored-native-mismatch.txt')
        rows.append(dict(pair_id=copy, **sides))
    (tmp_path / 'dataset_manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    ds = AlignmentDataset(tmp_path)
    assert len(ds) == 4
    assert len({r['sample_id'] for r in ds.records}) == 4
    assert len({r['group_id'] for r in ds.records}) == 1

import hashlib
import json

from PIL import Image

from scripts.data.complete_book_pair_assets import complete


def test_complete_catalogs_preserves_annotations_and_partitions_all_books(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    books = {}
    for side in 'ABC':
        rows = []
        for number in range(2):
            line_id = f'{side}_{number}'
            paths = {key: f'Book{side}/{kind}/{line_id}{ext}' for key, kind, ext in (
                ('image_path', 'images', '.png'), ('text_path', 'texts', '.txt'),
                ('bounding_boxes_path', 'BoundingBoxes', '.json'))}
            for rel in paths.values():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.new('L', (30, 10), 40 + number).save(root / paths['image_path'])
            (root / paths['text_path']).write_text('نص أصلي' if side != 'C' else 'نصأصلي')
            image_hash = hashlib.sha256((root / paths['image_path']).read_bytes()).hexdigest()
            box = dict(line_id=line_id, image_path=paths['image_path'], image_sha256=image_hash,
                       image_size=[30, 10], crop_status='SOURCE_LINE_IMAGE',
                       subwords=[dict(id='original', text='نص', line_bbox_xyxy=[2, 1, 20, 9])])
            (root / paths['bounding_boxes_path']).write_text(json.dumps(box))
            rows.append(dict(side=side, line_id=line_id, image_sha256=image_hash, **paths))
        (root / f'Book{side}/lines.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        books[side] = rows

    def copy_side(folder, side, row, stem, mask=True):
        result = dict(line_id=row['line_id'])
        for key, kind, ext in [('image_path', 'images', '.png'), ('text_path', 'texts', '.txt'),
                               ('bounding_boxes_path', 'BoundingBoxes', '.json')]:
            rel = f'{folder}/{kind}{side}/{stem}{ext}'
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes((root / row[key]).read_bytes())
            result[key] = rel
        if mask:
            rel = f'{folder}/masks{side}/{stem}.png'
            (root / rel).parent.mkdir()
            Image.new('L', (30, 10), 255).save(root / rel)
            result['mask_path'] = rel
        return result

    old = dict(pair_id='legacy')
    for side in 'AB':
        old[side] = copy_side('alignedPairs', side, books[side][0], 'legacy')
    (root / 'alignedPairs/pairs.jsonl').write_text(json.dumps(old) + '\n')
    for group in ('A_B', 'A_C', 'B_C'):
        row = dict(pair_id='legacy' if group == 'A_B' else group)
        for side in group.split('_'):
            row[side] = copy_side('alignedPairs/' + group, side, books[side][0], row['pair_id'])
            row[side]['bounding_boxes_path'] = old[side]['bounding_boxes_path'] if group == 'A_B' else books[side][0]['bounding_boxes_path']
            row[side].pop('text_path')
        folder = root / 'alignedPairs' / group
        (folder / 'pairs.jsonl').write_text(json.dumps(row) + '\n')
        (folder / 'summary.json').write_text(json.dumps(dict(number_of_pairs=1)))
        (folder / 'README.md').write_text('original')
    unmatched = []
    for side in 'AB':
        copied = copy_side('Unalignedlines', side, books[side][1], f'{side}_1', mask=False)
        unmatched.append(dict(books[side][1], **{k: v for k, v in copied.items() if k != 'line_id'}))
    (root / 'Unalignedlines/lines.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in unmatched))
    (root / 'metadata').mkdir()
    (root / 'metadata/book_pair_catalog.json').write_text(json.dumps(dict(groups={g: {} for g in ('A_B', 'A_C', 'B_C')})))
    (root / 'README.md').write_text('source README')
    (root / 'export_summary.json').write_text('{}')

    report = complete(root)
    assert report['legacy_pairs_preserved'] == 1
    assert report['legacy_duplicate_files_removed'] == 8
    assert report['unmatched'] == dict(A=1, B=1, C=1)
    assert {p.name for p in (root / 'alignedPairs').iterdir()} == {'A_B', 'A_C', 'B_C'}
    for group in ('A_B', 'A_C', 'B_C'):
        row = json.loads((root / 'alignedPairs' / group / 'pairs.jsonl').read_text())
        for side in group.split('_'):
            assert row[side]['text_path'].startswith('alignedPairs/' + group + '/')
            box = json.loads((root / row[side]['bounding_boxes_path']).read_text())
            assert box['image_path'] == row[side]['image_path']
            assert box['subwords'] == [dict(id='original', text='نص', line_bbox_xyxy=[2, 1, 20, 9])]
    assert (root / 'metadata/catalog_layout/source_files/alignedPairs/pairs.jsonl').read_text() == json.dumps(old) + '\n'

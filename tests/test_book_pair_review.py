import json

import pytest

from scripts.eval.review_book_pairs import GROUPS, merged_intervals, prepare, topology


def test_topology_uses_connected_regions_in_both_supplied_masks():
    assert merged_intervals([[20, 40], [0, 10], [10, 22]]) == [[0, 40]]
    row = dict(A={'mask_regions_x': [[0, 10], [10, 30]]}, B={'mask_regions_x': [[5, 20]]})
    assert topology(row, ['A', 'B']) == 'consecutive'
    row['A']['mask_regions_x'] = [[0, 10], [20, 30]]
    assert topology(row, ['A', 'B']) == 'asymmetric_or_empty'
    row['B']['mask_regions_x'] = [[0, 10], [11, 20]]
    assert topology(row, ['A', 'B']) == 'non_consecutive'


def test_stratified_selection_is_fixed_unique_and_excludes_training_lines(tmp_path):
    root = tmp_path / 'dataset'
    ids = dict(train=[], val=[], test=[])
    for group in GROUPS:
        folder = root / 'alignedPairs' / group
        folder.mkdir(parents=True)
        rows = []
        for kind in ('consecutive', 'non_consecutive'):
            for i in range(7):
                row = dict(pair_id=f'{group}_{kind}_{i}')
                for book in group.split('_'):
                    line = f'{group}_{kind}_{book}_{i}'
                    row[book] = dict(line_id=line, mask_regions_x=[[1, 20]] if kind == 'consecutive' else [[1, 8], [12, 20]])
                    if book != 'C':
                        ids['train' if i == 0 else 'test'].append(f'Book{book}:' + line)
                rows.append(row)
        (folder / 'pairs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    membership = tmp_path / 'split.json'
    membership.write_text(json.dumps(ids))
    selection = prepare(root, membership, tmp_path / 'out')
    assert selection == prepare(root, membership, tmp_path / 'out')
    assert len(selection['samples']) == len({r['review_id'] for r in selection['samples']}) == 30
    assert all('train' not in r['saved_line_splits'].values() for r in selection['samples'])
    for group in GROUPS:
        for kind in ('consecutive', 'non_consecutive'):
            rows = [r for r in selection['samples'] if r['book_pair'] == group and r['path_type'] == kind]
            assert len(rows) == 5
            for book in group.split('_'):
                assert len({r[book]['line_id'] for r in rows}) == 5
    ids['test'] = []
    membership.write_text(json.dumps(ids))
    with pytest.raises(ValueError, match='held-out eligible'):
        prepare(root, membership, tmp_path / 'out')


def test_catalog_verification_reuses_exact_bytes_and_rejects_bad_masks(tmp_path):
    import hashlib
    import numpy as np
    from PIL import Image
    from scripts.data.validate_book_pair_folders import validate

    root = tmp_path / 'dataset'
    for group in GROUPS:
        folder = root / 'alignedPairs' / group
        folder.mkdir(parents=True)
        row = dict(pair_id='example')
        for book in group.split('_'):
            image_path = folder / ('images' + book) / 'example.png'
            mask_path = folder / ('masks' + book) / 'example.png'
            image_path.parent.mkdir()
            mask_path.parent.mkdir()
            Image.new('RGB', (30, 10), 'white').save(image_path)
            pixels = np.zeros((10, 30), dtype=np.uint8)
            pixels[:, 2:20] = 255
            Image.fromarray(pixels).save(mask_path)
            row[book] = dict(line_id='line', image_path=image_path.relative_to(root).as_posix(),
                             mask_path=mask_path.relative_to(root).as_posix(), mask_regions_x=[[2, 20]])
        (folder / 'pairs.jsonl').write_text(json.dumps(row))
        (folder / 'summary.json').write_text(json.dumps(dict(number_of_pairs=1)))
    entries = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size,
               sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob('*') if p.is_file()}
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(dict(files=entries)))
    assert validate(root, snapshot)['verified_files'] == 18
    mask = root / 'alignedPairs/A_B/masksA/example.png'
    mask.unlink()
    with pytest.raises(FileNotFoundError):
        validate(root, snapshot)
    assert validate(root, snapshot, reuse=True)['reused_exact_copies'] == 1
    Image.new('L', (30, 10), 100).save(mask)
    rel = mask.relative_to(root).as_posix()
    entries[rel].update(bytes=mask.stat().st_size, sha256=hashlib.sha256(mask.read_bytes()).hexdigest())
    snapshot.write_text(json.dumps(dict(files=entries)))
    with pytest.raises(ValueError, match='binary'):
        validate(root, snapshot)

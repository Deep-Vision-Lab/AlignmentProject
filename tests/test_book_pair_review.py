import json

import pytest

from scripts.eval.review_book_pairs import (GROUPS, load_evaluation_setup, load_settings,
                                           merged_intervals, prepare, reuse_selection, topology)


def test_topology_uses_connected_regions_in_both_supplied_masks():
    assert merged_intervals([[20, 40], [0, 10], [10, 22]]) == [[0, 40]]
    row = dict(A={'mask_regions_x': [[0, 10], [10, 30]]}, B={'mask_regions_x': [[5, 20]]})
    assert topology(row, ['A', 'B']) == 'consecutive'
    row['A']['mask_regions_x'] = [[0, 10], [20, 30]]
    assert topology(row, ['A', 'B']) == 'asymmetric_or_empty'
    row['B']['mask_regions_x'] = [[0, 10], [11, 20]]
    assert topology(row, ['A', 'B']) == 'non_consecutive'


def test_review_reads_current_notebook_options_without_executing_setup(tmp_path):
    notebook = tmp_path / 'settings.ipynb'
    cells = [dict(source=[
        '# MODEL/DATA SETUP.\n', "EVALUATION_CROP = 'content'\n", "DEVICE = 'mps'\n",
        "REPRESENTATION = 'fused'\n", 'RANDOM_SEED = 42\n',
        "raise RuntimeError('Notebook setup must not execute')\n"]), dict(source=[
        '# EVALUATION-ONLY MATCHING SETTINGS.\n', 'from evaluate import resolved_match_settings\n',
        'REPEAT_PENALTY = 0.2\n', 'GAP_OPEN = 0.5940936607241908\n', 'ALLOW_SHORT_REGIONS = False\n']),
        dict(source=[
            'def current_match_settings(threshold=None):\n',
            '    return resolved_match_settings(dict(repeat_penalty=REPEAT_PENALTY, gap_open=GAP_OPEN, allow_short_regions=ALLOW_SHORT_REGIONS))\n',
            "raise RuntimeError('Session mutations and display must not execute')\n"])]
    notebook.write_text(json.dumps(dict(cells=cells)))
    assert load_evaluation_setup(notebook) == dict(EVALUATION_CROP='content', DEVICE='mps',
                                                 REPRESENTATION='fused', RANDOM_SEED=42)
    settings = load_settings(notebook)
    assert settings['repeat_penalty'] == 0.2
    assert settings['gap_open'] == 0.5940936607241908
    assert settings['allow_short_regions'] is False
    cells[1]['source'][2] = 'REPEAT_PENALTY = 0.7\n'
    notebook.write_text(json.dumps(dict(cells=cells)))
    assert load_settings(notebook)['repeat_penalty'] == 0.7


@pytest.mark.parametrize('saved_books', [('A', 'B'), ('C',)])
def test_stratified_selection_is_fixed_unique_and_excludes_training_lines(tmp_path, saved_books):
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
                    if book in saved_books:
                        ids['train' if i == 0 else 'test'].append(f'Book{book}:' + line)
                rows.append(row)
        (folder / 'pairs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    membership = tmp_path / 'split.json'
    membership.write_text(json.dumps(ids))
    selection = prepare(root, membership, tmp_path / 'out')
    assert selection == prepare(root, membership, tmp_path / 'out')
    assert len(selection['samples']) == len({r['review_id'] for r in selection['samples']}) == 30
    assert all('train' not in r['saved_line_splits'].values() for r in selection['samples'])
    assert all(split == ('test' if book in saved_books else 'unseen')
               for row in selection['samples'] for book, split in row['saved_line_splits'].items())
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


def test_reused_examples_keep_identity_and_refresh_finetuning_split_labels(tmp_path):
    source = tmp_path / 'selection.json'
    original = dict(split_manifest_sha256='original-hash', inventory={}, samples=[dict(
        review_id='A_C_consecutive_01', book_pair='A_C',
        A=dict(line_id='a'), C=dict(line_id='c'), saved_line_splits=dict(A='test', C='unseen'))])
    source.write_text(json.dumps(original))
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train=['BookC:c'], val=[], test=[])))
    selection = reuse_selection(source, split, tmp_path / 'output')
    row = selection['samples'][0]
    assert row['review_id'] == original['samples'][0]['review_id']
    assert row['original_saved_line_splits'] == dict(A='test', C='unseen')
    assert row['saved_line_splits'] == dict(A='unseen', C='train')
    assert selection['original_split_manifest_sha256'] == 'original-hash'
    assert selection['split_manifest_sha256'] != 'original-hash'
    assert 'not a held-out performance estimate' in selection['policy']
    assert json.loads(source.read_text()) == original


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

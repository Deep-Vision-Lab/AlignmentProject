"""Make the three aligned catalogs self-contained and refresh unmatched inventories.

Run after importing the complete BookA/BookB/BookC source inventories. Original
Drive manifests are retained under metadata/catalog_layout/source_files. Images,
masks, transcriptions, and annotation coordinates are never regenerated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

GROUPS = ('A_B', 'A_C', 'B_C')
KINDS = ('images', 'masks', 'texts', 'BoundingBoxes')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise ValueError(f'Existing copy differs: {target}')
    else:
        shutil.copyfile(source, target)


def complete(root):
    root = root.resolve()
    archive = root / 'metadata/catalog_layout/source_files'
    report_path = root / 'metadata/catalog_layout/report.json'
    if report_path.exists():
        raise ValueError('Layout migration already completed; run validate_book_pair_folders.py')

    def backup(path):
        copy_exact(path, archive / path.relative_to(root))

    books = {s: {r['line_id']: r for r in read_rows(root / f'Book{s}/lines.jsonl')} for s in 'ABC'}
    catalogs = {g: read_rows(root / 'alignedPairs' / g / 'pairs.jsonl') for g in GROUPS}
    old = read_rows(root / 'alignedPairs/pairs.jsonl')
    ab_ids = {r['pair_id'] for r in catalogs['A_B']}
    if not {r['pair_id'] for r in old} <= ab_ids:
        raise ValueError('A_B does not contain all legacy pairs')

    # Never discard unmatched copies carrying a distinct image/text/annotation.
    for row in read_rows(root / 'Unalignedlines/lines.jsonl'):
        book = books[row['side']][row['line_id']]
        for key in ('image_path', 'text_path'):
            if (root / row[key]).read_bytes() != (root / book[key]).read_bytes():
                raise ValueError('Distinct unmatched asset: ' + row[key])
        copied = json.loads((root / row['bounding_boxes_path']).read_text())
        canonical = json.loads((root / book['bounding_boxes_path']).read_text())
        for key in ('image_path', 'export_record_id'):
            copied.pop(key, None)
            canonical.pop(key, None)
        if copied != canonical:
            raise ValueError('Distinct unmatched annotation: ' + row['bounding_boxes_path'])

    # Establish every source and its image identity before altering manifests.
    plan, used = [], {s: set() for s in 'ABC'}
    for group, rows in catalogs.items():
        folder = root / 'alignedPairs' / group
        if len(rows) != len({r['pair_id'] for r in rows}):
            raise ValueError('Duplicate pair IDs: ' + group)
        for row in rows:
            stem = row['pair_id']
            if Path(stem).name != stem:
                raise ValueError('Unsafe pair ID')
            for s in group.split('_'):
                side = row[s]
                book = books[s][side['line_id']]
                image = root / side['image_path']
                if digest(image) != book['image_sha256']:
                    raise ValueError('Pair/book image differs: ' + str(image))
                text = root / side.get('text_path', book['text_path'])
                boxes = root / side['bounding_boxes_path']
                box = json.loads(boxes.read_text())
                if box['line_id'] != side['line_id'] or box['image_sha256'] != digest(image):
                    raise ValueError('Annotation/image identity differs: ' + str(boxes))
                if text.read_bytes() != (root / book['text_path']).read_bytes():
                    raise ValueError('Pair/book transcription differs: ' + str(text))
                target_text = folder / ('texts' + s) / (stem + '.txt')
                target_boxes = folder / ('BoundingBoxes' + s) / (stem + '.json')
                plan.append((side, text, boxes, box, target_text, target_boxes))
                used[s].add(side['line_id'])

    for path in [root / 'README.md', root / 'export_summary.json', root / 'Unalignedlines/lines.jsonl',
                 root / 'metadata/book_pair_catalog.json']:
        backup(path)
    for group in GROUPS:
        for name in ('pairs.jsonl', 'summary.json', 'README.md'):
            backup(root / 'alignedPairs' / group / name)

    for side, text, boxes, box, target_text, target_boxes in plan:
        copy_exact(text, target_text)
        box['image_path'] = side['image_path']
        # Keep all source rectangles, subword IDs/text, warnings and versions.
        write_json(target_boxes, box)
        side['text_path'] = target_text.relative_to(root).as_posix()
        side['bounding_boxes_path'] = target_boxes.relative_to(root).as_posix()
        side['text_sha256'] = digest(target_text)
        side['bounding_boxes_sha256'] = digest(target_boxes)

    catalog = json.loads((root / 'metadata/book_pair_catalog.json').read_text())
    for group, rows in catalogs.items():
        folder = root / 'alignedPairs' / group
        write_rows(folder / 'pairs.jsonl', rows)
        summary = json.loads((folder / 'summary.json').read_text())
        summary.update(text_count=2 * len(rows), bounding_box_file_count=2 * len(rows),
                       assets_self_contained=True)
        write_json(folder / 'summary.json', summary)
        (folder / 'README.md').write_text(
            f'# {group}: {len(rows):,} aligned pairs\n\n'
            'Each book side has images, supplied binary masks, texts, and full subword '
            'bounding-box JSON files inside this folder. Matching pair filename stems '
            'connect the files. `pairs.jsonl` uses paths relative to the dataset root.\n\n'
            'Transcriptions and annotation coordinates are preserved. Book C has no '
            'source word spaces; none are inferred. Supplied masks are unchanged. '
            'Automatic pairs retain their original provenance and review status. '
            'Original Drive manifests are preserved under '
            '`metadata/catalog_layout/source_files/`.\n')
        catalog['groups'][group].update(texts=2 * len(rows), bounding_box_files=2 * len(rows),
                                       assets_self_contained=True)
    write_json(root / 'metadata/book_pair_catalog.json', catalog)

    # Rebuild inventories from the full books, excluding every saved partner in
    # every catalog. These are independent lines, not verified negative pairs.
    staged = root / '.Unalignedlines_complete'
    staged.mkdir(exist_ok=True)
    unmatched = []
    for s in 'ABC':
        for kind in ('images', 'texts', 'BoundingBoxes'):
            (staged / (kind + s)).mkdir(exist_ok=True)
        for line_id, book in books[s].items():
            if line_id in used[s]:
                continue
            row = dict(book, reason='No saved partner in A_B, A_C, or B_C')
            for key, kind in [('image_path', 'images'), ('text_path', 'texts'),
                              ('bounding_boxes_path', 'BoundingBoxes')]:
                source = root / book[key]
                target = staged / (kind + s) / source.name
                row[key] = 'Unalignedlines/' + target.relative_to(staged).as_posix()
                if kind == 'BoundingBoxes':
                    box = json.loads(source.read_text())
                    box['image_path'] = row['image_path']
                    write_json(target, box)
                else:
                    copy_exact(source, target)
            unmatched.append(row)
    write_rows(staged / 'lines.jsonl', unmatched)
    write_json(staged / 'summary.json', dict(mode='independent-unmatched-lines',
        counts=dict(Counter(row['side'] for row in unmatched)),
        excluded_catalogs=list(GROUPS), definition='Book lines without a saved partner in any catalog; not verified negatives'))

    # Check each legacy file against its destination before deleting duplicates.
    legacy_files = []
    for s in 'AB':
        for kind in KINDS:
            folder = root / 'alignedPairs' / (kind + s)
            for source in folder.iterdir():
                if source.name == '.DS_Store':
                    continue
                target = root / 'alignedPairs/A_B' / (kind + s) / source.name
                if kind == 'BoundingBoxes':
                    a, b = (json.loads(p.read_text()) for p in (source, target))
                    a['image_path'] = b['image_path']
                    same = a == b
                else:
                    same = source.read_bytes() == target.read_bytes()
                if not same:
                    raise ValueError('Legacy file not preserved: ' + str(source))
                legacy_files.append(dict(source=source.relative_to(root).as_posix(),
                    destination=target.relative_to(root).as_posix(), original_sha256=digest(source)))
    write_json(root / 'metadata/catalog_layout/legacy_file_mapping.json', legacy_files)
    backup(root / 'alignedPairs/pairs.jsonl')
    old_unmatched = root / '.Unalignedlines_previous'
    (root / 'Unalignedlines').rename(old_unmatched)
    staged.rename(root / 'Unalignedlines')
    for s in 'AB':
        for kind in KINDS:
            shutil.rmtree(root / 'alignedPairs' / (kind + s))
    (root / 'alignedPairs/pairs.jsonl').unlink()
    shutil.rmtree(old_unmatched)
    report = dict(catalogs={g: len(rows) for g, rows in catalogs.items()},
        book_lines={s: len(rows) for s, rows in books.items()},
        unmatched=dict(Counter(row['side'] for row in unmatched)),
        legacy_pairs_preserved=len(old), legacy_duplicate_files_removed=len(legacy_files),
        source_files='metadata/catalog_layout/source_files',
        checks=['Pair images match book SHA256', 'Pair texts equal original book text bytes',
                'Full source boxes match line/image identity', 'Legacy assets preserved before cleanup'])
    write_json(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('DataSet/ArabicData'))
    print(json.dumps(complete(parser.parse_args().dataset), indent=2))

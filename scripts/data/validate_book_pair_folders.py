"""Verify the downloaded A_B/A_C/B_C catalog and its Drive snapshot checksums."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def validate(root, snapshot, reuse=False):
    import numpy as np
    from PIL import Image

    groups = ('A_B', 'A_C', 'B_C')
    remote = json.loads(snapshot.read_text())['files']
    entries = {rel: data for rel, data in remote.items()
               if rel.startswith(tuple('alignedPairs/' + group + '/' for group in groups))}
    if not entries:
        raise ValueError('Snapshot has no book pair files')
    digest_cache = {}
    originals = root / 'metadata/catalog_layout/source_files'

    def snapshot_path(rel):
        # Layout enrichment changes only these three catalog metadata files.
        archived = originals / rel
        if Path(rel).name in ('pairs.jsonl', 'summary.json', 'README.md') and archived.is_file():
            return archived
        return root / rel

    def digest(path):
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if key not in digest_cache:
            digest_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest_cache[key]

    hashes = {}
    for rel, data in entries.items():
        path = snapshot_path(rel)
        if path.exists():
            if path.stat().st_size != data['bytes'] or digest(path) != data['sha256']:
                raise ValueError('Size/checksum differs from Drive snapshot: ' + rel)
            hashes[data['sha256']] = path
    reused = 0
    for rel, data in entries.items():
        path = snapshot_path(rel)
        if not path.exists():
            source = hashes.get(data['sha256'])
            if not reuse or source is None:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
            if digest(path) != data['sha256']:
                raise ValueError('Local reuse checksum mismatch')
            reused += 1

    catalog = {}
    for group in groups:
        folder = root / 'alignedPairs' / group
        rows = [json.loads(line) for line in (folder / 'pairs.jsonl').read_text().splitlines() if line.strip()]
        summary = json.loads((folder / 'summary.json').read_text())
        if len(rows) != summary['number_of_pairs'] or len({r['pair_id'] for r in rows}) != len(rows):
            raise ValueError('Pair count/uniqueness differs: ' + group)
        original_manifest = originals / 'alignedPairs' / group / 'pairs.jsonl'
        if original_manifest.is_file():
            original_rows = [json.loads(line) for line in original_manifest.read_text().splitlines() if line.strip()]
            original_by_id = {row['pair_id']: row for row in original_rows}
            if set(original_by_id) != {row['pair_id'] for row in rows}:
                raise ValueError('Pair identities changed: ' + group)
            for row in rows:
                original = original_by_id[row['pair_id']]
                for key, value in original.items():
                    if key in group.split('_'):
                        for field, before in value.items():
                            if field not in ('text_path', 'bounding_boxes_path') and row[key].get(field) != before:
                                raise ValueError('Source pair annotation changed: ' + group + '/' + key + '/' + field)
                    elif row.get(key) != value:
                        raise ValueError('Source pair field changed: ' + group + '/' + key)
        seen_lines, mask_origins = set(), Counter()
        asset_counts = Counter()
        for row in rows:
            for book in group.split('_'):
                side = row[book]
                image_path, mask_path = (root / side[key] for key in ('image_path', 'mask_path'))
                for path in (image_path, mask_path):
                    if not path.resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Pair path outside its book folder')
                    if path.relative_to(root).as_posix() not in entries:
                        raise ValueError('Pair asset absent from Drive snapshot')
                with Image.open(image_path) as image:
                    image.load()
                    size = image.size
                if summary.get('assets_self_contained') or 'text_path' in side and 'bounding_boxes_sha256' in side:
                    text_path, box_path = (root / side[key] for key in ('text_path', 'bounding_boxes_path'))
                    for path, hash_key in ((text_path, 'text_sha256'), (box_path, 'bounding_boxes_sha256')):
                        if not path.resolve().is_relative_to(folder.resolve()):
                            raise ValueError('Text/box path outside its book pair folder')
                        if digest(path) != side[hash_key]:
                            raise ValueError('Enriched asset checksum differs: ' + str(path))
                    box = json.loads(box_path.read_text())
                    if (box['line_id'] != side['line_id'] or box['image_path'] != side['image_path']
                            or box['image_sha256'] != digest(image_path) or box['image_size'] != list(size)):
                        raise ValueError('Box/image identity differs: ' + str(box_path))
                    asset_counts.update(texts=1, bounding_box_files=1)
                with Image.open(mask_path) as image:
                    mask = np.asarray(image.convert('L'))
                if mask.shape != size[::-1] or not np.isin(mask, [0, 255]).all():
                    raise ValueError('Mask size/binary values: ' + str(mask_path))
                if not np.all(mask == mask[0]):
                    raise ValueError('Mask is not full-height: ' + str(mask_path))
                foreground = (mask[0] > 0)
                actual = np.flatnonzero(np.diff(np.r_[False, foreground, False])).reshape(-1, 2).tolist()
                intervals = []
                for left, right in sorted(side['mask_regions_x']):
                    if not 0 <= left < right <= size[0]:
                        raise ValueError('Mask interval out of bounds')
                    if intervals and left <= intervals[-1][1]:
                        intervals[-1][1] = max(right, intervals[-1][1])
                    else:
                        intervals.append([left, right])
                if intervals != actual:
                    raise ValueError('Mask pixels differ from manifest intervals: ' + str(mask_path))
                seen_lines.add((book, side['line_id']))
                mask_origins[side.get('mask_status', 'unspecified')] += 1
        catalog[group] = dict(pairs=len(rows), images=2 * len(rows), masks=2 * len(rows),
                              unique_book_lines=len(seen_lines), mask_origins=dict(mask_origins), **asset_counts)
    unmatched_counts = {}
    if (root / 'metadata/catalog_layout/report.json').is_file():
        used = {book: set() for book in 'ABC'}
        for group in groups:
            for line in (root / 'alignedPairs' / group / 'pairs.jsonl').read_text().splitlines():
                row = json.loads(line)
                for book in group.split('_'):
                    used[book].add(row[book]['line_id'])
        unmatched_rows = [json.loads(line) for line in (root / 'Unalignedlines/lines.jsonl').read_text().splitlines()]
        for book in 'ABC':
            canonical = {row['line_id']: row for row in map(json.loads,
                (root / f'Book{book}/lines.jsonl').read_text().splitlines())}
            actual = {row['line_id']: row for row in unmatched_rows if row['side'] == book}
            if set(actual) != set(canonical) - used[book]:
                raise ValueError('Unmatched inventory overlaps or omits book lines: ' + book)
            if len(actual) != sum(row['side'] == book for row in unmatched_rows):
                raise ValueError('Duplicate unmatched lines: ' + book)
            for line_id, row in actual.items():
                original = canonical[line_id]
                for key in ('image_path', 'text_path'):
                    if digest(root / row[key]) != digest(root / original[key]):
                        raise ValueError('Unmatched asset differs from book: ' + row[key])
                copied = json.loads((root / row['bounding_boxes_path']).read_text())
                source = json.loads((root / original['bounding_boxes_path']).read_text())
                source['image_path'] = row['image_path']
                if copied != source:
                    raise ValueError('Unmatched annotation differs from book: ' + row['bounding_boxes_path'])
            unmatched_counts[book] = len(actual)
    return dict(source_snapshot=str(snapshot), snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                verified_files=len(entries), bytes=sum(v['bytes'] for v in entries.values()),
                reused_exact_copies=reused, catalog=catalog, unmatched=unmatched_counts,
                checks=['Drive snapshot SHA256 and byte sizes', 'Unique pair IDs and summary counts',
                        'Every pair image decoded', 'Binary full-height masks at exact image size',
                        'Mask pixels match manifest intervals',
                        'Enriched text/box checksums and exact image identity',
                        'Unmatched inventories equal full books minus every saved partner'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('DataSet/ArabicData'))
    parser.add_argument('--snapshot', type=Path, default=Path('DataSet/ArabicData/metadata/drive_sync_2026-11-07/workspace_manifest.json'))
    parser.add_argument('--reuse-local', action='store_true')
    args = parser.parse_args()
    report = validate(args.dataset, args.snapshot, args.reuse_local)
    target = args.snapshot.parent / 'verification.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

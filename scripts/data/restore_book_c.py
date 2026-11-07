"""Restore Book C from its extracted original archive and Drive catalog metadata.

Requires BookC/lines.jsonl, book.json, README.md, the original archive, and the
first BoundingBoxes JSON template downloaded from Drive. Every output must match
its Drive SHA256 and byte length; mismatches are never written.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path, default=Path('DataSet/ArabicData'))
parser.add_argument('--source-dir', type=Path, required=True, help='Directory containing extracted KuficQuran/')
args = parser.parse_args()
root = args.dataset
snapshot = json.loads((root / 'metadata/drive_sync_2026-11-07/workspace_manifest.json').read_text())['files']
template = json.loads((root / 'BookC/BoundingBoxes/KuficQuran__0010__line_001.json').read_text())
rows = [json.loads(line) for line in (root / 'BookC/lines.jsonl').read_text().splitlines()]
mismatches = []
def save_checked(rel, data):
    expected = snapshot[rel]
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        mismatches.append(rel)
        return
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Existing differs: ' + rel)
    path.write_bytes(data)

for row in rows:
    source = row['source_occurrences'][0]
    image = args.source_dir / source['image_member']
    annotation = args.source_dir / source['annotation_member']
    data = annotation.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['source_annotation_sha256']
    payload = json.loads(data)
    save_checked(row['image_path'], image.read_bytes())
    save_checked(row['text_path'], row['raw_text'].encode())
    box = copy.deepcopy(template)
    for key in ['line_id', 'image_path', 'image_sha256', 'source_annotation_sha256']:
        box[key] = row[key]
    box['export_record_id'] = row['line_id']
    with Image.open(image) as im:
        width, height = im.size
    box['image_size'] = [width, height]
    box['source_occurrence'] = source
    box['source_annotation'] = payload
    box['warnings'] = row.get('warnings', [])
    box['subwords'] = []
    for i, subword in enumerate(payload['text']):
        x, y, w, h = subword['bbox']
        valid = 0 <= x < x+w <= width and 0 <= y < y+h <= height
        box['subwords'].append(dict(id=f'source_index_{i:04d}', source_index=i,
            text=subword['label'], source_bbox_xywh=subword['bbox'],
            line_bbox_xyxy=[x,y,x+w,y+h], page_bbox_xyxy=None,
            status='SOURCE_LINE_COORDINATES' if valid else 'SOURCE_BOX_OUTSIDE_LINE'))
    save_checked(row['bounding_boxes_path'], (json.dumps(box, ensure_ascii=False, indent=2)+'\n').encode())

if mismatches:
    raise ValueError(f'{len(mismatches)} files differ from Drive snapshot: {mismatches[:10]}')
verified = []
for rel, expected in snapshot.items():
    if not rel.startswith('BookC/'):
        continue
    data = (root / rel).read_bytes()
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError('Book C snapshot mismatch: ' + rel)
    verified.append(rel)
report = dict(lines=len(rows), files_verified=len(verified), mismatches=0,
              method='Original archive images, full_transcription text, lossless annotation conversion; all outputs match Drive SHA256 and byte lengths')
(root / 'metadata/book_c_import_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

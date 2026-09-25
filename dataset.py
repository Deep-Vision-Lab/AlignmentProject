"""One image/text dataset for synthetic pairs, Arabic manifests and Bridge V3.

Native line crops reproduce the dataset builder's XML envelope and safety margin.
No heuristic foreground crop, geometric augmentation, padding, or polarity change.
Bridge items always retain the real/positive relationship (including its mask).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=512)
def _xml_lines(side_dir):
    side = Path(side_dir)
    boxes = []
    for elem in ET.parse(side / 'original.xml').getroot().findall('.//DocumentElement'):
        values = [elem.findtext(k) for k in ('X', 'Y', 'Width', 'Height')]
        if None not in values:
            x, y, w, h = [int(float(v)) for v in values]
            boxes.append((x, y, x + w, y + h, h, y + h / 2))
    if not boxes:
        raise ValueError(f'No XML boxes in {side}')
    heights = sorted(b[4] for b in boxes if b[4] > 0)
    centers = sorted(b[5] for b in boxes)
    gaps = sorted(b - a for a, b in zip(centers, centers[1:]) if b > a)
    median_h = heights[len(heights) // 2] if heights else 40
    median_gap = gaps[len(gaps) // 2] if gaps else median_h
    threshold = max(int(max(.6 * median_h, .6 * median_gap)), 25)
    lines = []
    for box in sorted(boxes, key=lambda b: b[5]):
        if lines and abs(box[5] - lines[-1][-1][5]) < threshold:
            lines[-1].append(box)
        else:
            lines.append([box])
    lines.sort(key=lambda line: min(b[1] for b in line))
    originals = sorted(side.glob('original_image.*'))
    if not originals:
        raise FileNotFoundError(f'XML mapping requires original_image.* in {side}')
    with Image.open(originals[0]) as image:
        page_size = image.size
    return lines, page_size


def xml_crop_bounds(image_path, image_size, margin_ratio=.05, minimum_margin=2):
    """Four-sided crop in *saved line* pixels, including builder's 25% y padding."""
    path = Path(image_path)
    match = re.fullmatch(r'line_(\d+)', path.stem)
    if path.parent.name != 'linesImages' or not match:
        raise ValueError(f'Not a native linesImages/line_N image: {path}')
    lines, (page_w, page_h) = _xml_lines(str(path.parent.parent))
    index = int(match[1]) - 1
    if not 0 <= index < len(lines):
        raise ValueError(f'Line {index + 1} outside XML with {len(lines)} lines: {path}')
    line = lines[index]
    x0, y0 = min(b[0] for b in line), min(b[1] for b in line)
    x1, y1 = max(b[2] for b in line), max(b[3] for b in line)
    pad = int(.25 * max(1, y1 - y0))
    saved_y0, saved_y1 = max(0, y0 - pad), min(page_h, y1 + pad)
    w, h = image_size
    sx, sy = w / page_w, h / max(1, saved_y1 - saved_y0)
    margin = max(minimum_margin, round(max(1., (y1 - y0) * sy) * margin_ratio))
    bounds = (max(0, round(x0 * sx) - margin),
              max(0, round((y0 - saved_y0) * sy) - margin),
              min(w, round(x1 * sx) + margin),
              min(h, round((y1 - saved_y0) * sy) + margin))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(f'Invalid XML crop {bounds}: {path}')
    return bounds


def scan_augment(image):
    """Current scan-only defaults; intensity changes never move image geometry."""
    if random.random() >= .95:
        return image
    image = ImageEnhance.Brightness(image).enhance(random.uniform(.88, 1.12))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(.78, 1.22))
    if random.random() < .55:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(.35, 1.4)))
    values = np.array(image, dtype=np.float32)
    if random.random() < .75:
        noise = np.random.normal(0, random.uniform(5., 16.), values.shape[:2])
        values += noise if values.ndim == 2 else noise[..., None]
    if random.random() < .30:
        count = round(values.shape[0] * values.shape[1] * .001)
        ys = np.random.randint(values.shape[0], size=count)
        xs = np.random.randint(values.shape[1], size=count)
        dust = np.random.choice([0, 255], size=count)
        values[ys, xs] = dust if values.ndim == 2 else dust[:, None]
    return Image.fromarray(values.clip(0, 255).astype(np.uint8))


def prepare_image(path, image_size=(128, 1024), grayscale=True, crop='auto',
                  bbox=None, augment=False, binarize=False):
    """Return prepared PIL, normalized tensor and invertible direct-resize geometry.

    auto: strict XML for native lines, explicit bbox if supplied, otherwise full
    source. xml requires native XML; none is an explicit full-source override.
    """
    if crop not in {'auto', 'xml', 'none'}:
        raise ValueError('crop must be auto, xml or none')
    path = Path(path)
    with Image.open(path) as source:
        source_size = source.size
        bounds = (0, 0, *source_size)
        if crop != 'none':
            if bbox is not None:
                bounds = tuple(map(int, bbox))
            elif crop == 'xml' or path.parent.name == 'linesImages':
                bounds = xml_crop_bounds(path, source_size)
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
            raise ValueError(f'Crop {bounds} outside source {source_size}: {path}')
        image = source.crop(bounds).convert('L' if grayscale else 'RGB')
    if augment:
        image = scan_augment(image)
    h, w = map(int, image_size)
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    if binarize:
        image = image.point(lambda p: 255 if p >= 180 else 0)
    pixels = np.array(image, dtype=np.float32) / 255.
    pixels = pixels[..., None] if pixels.ndim == 2 else pixels
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
    mean, std = ((.449,), (.226,)) if grayscale else ((.485, .456, .406), (.229, .224, .225))
    tensor = (tensor - tensor.new_tensor(mean)[:, None, None]) / tensor.new_tensor(std)[:, None, None]
    geometry = dict(source_size=list(source_size), crop=list(bounds), model_size=[w, h],
                    scale_x=w / (x1 - x0), scale_y=h / (y1 - y0), padding=False)
    return image, tensor, geometry


class AlignmentDataset(Dataset):
    def __init__(self, root, dataset_type='auto', split=None, augment=False, paired=False,
                 image_size=(128, 1024), grayscale=True, crop='auto', binarize=False):
        self.root = Path(root).expanduser().resolve()
        self.augment = bool(augment and split not in {'val', 'valid', 'test'})
        self.paired = bool(paired)
        self.image_size, self.grayscale, self.crop, self.binarize = image_size, grayscale, crop, binarize
        manifest = self.root if self.root.is_file() else self.root / 'dataset_manifest.jsonl'
        if self.root.is_file():
            self.root = self.root.parent
        if not manifest.exists() and (self.root / 'anchor_index.jsonl').exists():
            manifest = self.root / 'anchor_index.jsonl'
        rows = []
        if manifest.exists():
            rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
        if dataset_type == 'auto':
            dataset_type = ('real_synthetic' if rows and ('anchor_id' in rows[0] or rows[0].get('bridge'))
                            else 'real' if rows else 'synthetic')
        if dataset_type not in {'synthetic', 'real', 'real_synthetic'}:
            raise ValueError(f'Unknown dataset_type {dataset_type}')
        self.dataset_type = dataset_type
        self.records = []
        page_hashes = {}

        def resolve(value):
            p = Path(value).expanduser()
            p = p if p.is_absolute() else self.root / p
            # Manifests may have been produced on another machine and store
            # absolute paths from that checkout. Rebase paths rooted at this
            # dataset directory onto the dataset currently being loaded.
            if not p.is_file() and Path(value).expanduser().is_absolute():
                parts = p.parts
                anchors = [i for i, part in enumerate(parts) if part == self.root.name]
                if anchors:
                    rebased = self.root.joinpath(*parts[anchors[-1] + 1:])
                    if rebased.is_file():
                        p = rebased
            if not p.is_file():
                raise FileNotFoundError(p)
            return str(p.resolve())

        def side_record(side, group):
            image = resolve(side.get('line_image_path') or side['image'])
            p = Path(image)
            if p.parent.name == 'linesImages':
                # Native own-side association is authoritative, on BOTH sides.
                text = resolve(p.parent.parent / 'text/final/original' / (p.stem + '.txt'))
                originals = sorted(p.parent.parent.glob('original_image.*'))
                if originals:
                    key = str(originals[0])
                    if key not in page_hashes:
                        page_hashes[key] = file_hash(key)
                    group = 'page:' + page_hashes[key]
            else:
                text = resolve(side.get('text_original_path') or side['text'])
                group = str(side.get('group_id') or side.get('page_id') or side.get('page_dir') or group)
            mask = side.get('alignment_mask_path') or side.get('mask')
            return dict(image=image, text=text, mask=resolve(mask) if mask else None,
                        group_id=group, bbox=side.get('bbox'))

        if dataset_type == 'synthetic':
            for path in sorted((self.root / 'images').glob('img1_*.png')):
                idx = path.stem.split('_', 1)[1]
                sides = [side_record({'image': f'images/img{n}_{idx}.png',
                                      'text': f'texts/text{n}_{idx}.txt'}, idx) for n in (1, 2)]
                self._append(sides, f'synthetic:{idx}', None, None, paired)
        else:
            if not rows:
                raise FileNotFoundError(f'Missing/empty manifest: {manifest}')
            for number, row in enumerate(rows):
                anchor = str(row.get('anchor_id') or (row.get('bridge') or {}).get('anchor_id') or '')
                if dataset_type == 'real_synthetic' and row.get('label_type') == 'no_shared_content':
                    continue  # This view is explicitly real + POSITIVE synthetic.
                if dataset_type == 'real_synthetic' and not anchor:
                    raise ValueError('Bridge row missing anchor_id')
                group = anchor or str(row.get('pair_id', number))
                sides = [side_record(row.get('A') or row['real'], group),
                         side_record(row.get('B') or row['positive'], group)]
                if anchor:
                    for side in sides:
                        side['group_id'] = anchor
                self._append(sides, f'{group}:{number}', row.get('label_type'),
                             row.get('split'), paired or bool(anchor), anchor)
        # Deduplicate repeated manifest associations, not page/line-name aliases:
        # copies of one page can carry different annotations. Keep those records
        # together in one group without discarding either annotation.
        unique = {}
        for record in self.records:
            first = record['sides'][0]
            key = record['sample_id'] if len(record['sides']) == 2 else (first['image'], first['text'])
            if key in unique:
                previous = unique[key]
                if (Path(previous['sides'][0]['text']).read_bytes() != Path(first['text']).read_bytes()
                        or previous['split'] != record['split']):
                    raise ValueError(f'Conflicting transcript/split for {key}')
                continue
            unique[key] = record
        self.records = list(unique.values())
        # A paired record links pages. Connected components prevent a page from
        # reaching multiple splits through different pairs.
        parent = {}
        def find(key):
            parent.setdefault(key, key)
            if parent[key] != key:
                parent[key] = find(parent[key])
            return parent[key]
        for record in self.records:
            groups = [s['group_id'] for s in record['sides']]
            for group in groups[1:]:
                parent[find(group)] = find(groups[0])
        for record in self.records:
            record['group_id'] = find(record['sides'][0]['group_id'])
        if split is not None:
            wanted = 'val' if split == 'valid' else split
            self.records = [r for r in self.records if r['split'] == wanted]
            if not self.records:
                raise ValueError(f'No predefined {wanted} records; use create_dataloaders for generated splits')
        if not self.records:
            raise ValueError(f'No {dataset_type} records in {self.root}')

    def _append(self, sides, sample_id, label, split, paired, anchor=''):
        for index, selected in enumerate([sides] if paired else [[s] for s in sides]):
            identity = []
            for key in ('image', 'text'):
                path = Path(selected[0][key])
                identity.append(str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path))
            sid = sample_id if paired else hashlib.sha256('\0'.join(identity).encode()).hexdigest()
            self.records.append(dict(sides=selected, sample_id=sid, label=label,
                                     split='val' if split == 'valid' else split, anchor_id=anchor))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images, texts, geometries, masks = [], [], [], []
        for side in record['sides']:
            _, tensor, geometry = prepare_image(side['image'], self.image_size, self.grayscale,
                                                self.crop, side['bbox'], self.augment, self.binarize)
            images.append(tensor)
            texts.append(Path(side['text']).read_text(encoding='utf-8-sig').strip())
            geometries.append(geometry)
            mask = None
            if side['mask']:
                with Image.open(side['mask']) as source:
                    if list(source.size) != geometry['source_size']:
                        raise ValueError(f'Mask/source size mismatch: {side["mask"]}')
                    pixels = np.array(source.convert('L').crop(geometry['crop']).resize(
                        tuple(geometry['model_size']), Image.Resampling.NEAREST))
                mask = torch.from_numpy((pixels >= 128).astype(np.float32))[None]
            masks.append(mask)
        return dict(image=images[0], text=texts[0], image2=images[1] if len(images) == 2 else None,
                    text2=texts[1] if len(texts) == 2 else None, mask=masks[-1],
                    sample_id=record['sample_id'], group_id=record['group_id'], label=record['label'],
                    anchor_id=record['anchor_id'], geometry=geometries,
                    image_paths=[s['image'] for s in record['sides']])

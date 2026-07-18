#!/usr/bin/env python3
"""Audit and repair ArabicDataset line transcripts against trusted Quran text.

The script is deliberately conservative:

* image files and raw OCR files are never modified;
* dry-run is the default;
* only lines whose best sequential reference match reaches ``--min-confidence``
  are proposed/applied;
* every existing file changed by ``--apply`` is copied into the timestamped
  report directory first;
* repaired files keep the existing manifest paths, so the manifest does not
  need to be rewritten.

The expected dataset layout is documented in ``DATASET_README.md``.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ARABIC_DIACRITICS_RE = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
ARABIC_PUNCTUATION_RE = re.compile(r"[\u0600-\u0605\u060c\u061b\u061f\u06d4]")
NON_WORD_RE = re.compile(r"[^\w\u0600-\u06ff]+", re.UNICODE)
LINE_NUMBER_RE = re.compile(r"(\d+)")
REFERENCE_LINE_RE = re.compile(
    r"^\s*(\d+)\s*(?:\||\t|:|,)\s*(\d+)\s*(?:\||\t|:|,)\s*(.+?)\s*$"
)

SURAH_KEYS = {
    "surah",
    "surah_number",
    "surah_no",
    "sura",
    "sura_number",
    "chapter",
    "chapter_number",
}
AYAH_START_KEYS = {
    "ayah_start",
    "start_ayah",
    "first_ayah",
    "ayah_from",
    "verse_start",
    "start_verse",
}
AYAH_END_KEYS = {
    "ayah_end",
    "end_ayah",
    "last_ayah",
    "ayah_to",
    "verse_end",
    "end_verse",
}
AYAH_VALUE_KEYS = {
    "ayah",
    "ayahs",
    "ayah_ids",
    "ayah_numbers",
    "verse",
    "verses",
    "verse_ids",
    "verse_numbers",
}


@dataclass(frozen=True, order=True)
class VerseKey:
    surah: int
    ayah: int


@dataclass
class Verse:
    key: VerseKey
    clean: str
    tashkeel: str


@dataclass
class Token:
    clean: str
    tashkeel: str
    key: VerseKey


@dataclass
class SourceCandidate:
    kind: str
    path: Path
    text: str


@dataclass
class MatchResult:
    source: SourceCandidate
    start: int
    end: int
    confidence: float
    repaired_original: str
    repaired_tashkeel: str


@dataclass
class AuditRow:
    pair_id: str
    side: str
    line_idx: int
    status: str
    confidence: float | None
    source_kind: str | None
    source_path: str | None
    original_target: str
    tashkeel_target: str
    before_original: str
    before_tashkeel: str
    proposed_original: str
    proposed_tashkeel: str
    reference_start: str | None
    reference_end: str | None
    reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and repair ArabicDataset final line text using trusted clean "
            "and tashkeel Quran references. Dry-run is the default."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("DataSet/ArabicDataset"),
        help="ArabicDataset root (default: DataSet/ArabicDataset).",
    )
    parser.add_argument(
        "--reference-clean",
        type=Path,
        required=True,
        help="Trusted Quran text without tashkeel.",
    )
    parser.add_argument(
        "--reference-tashkeel",
        type=Path,
        required=True,
        help="Trusted Quran text with tashkeel.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Only process this pair id (repeatable, for example pair_000001).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.78,
        help="Minimum match confidence required to repair a line (default: 0.78).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional deterministic cap, useful for a small audit first.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report directory. Default: DATASET_ROOT/text_repair_reports/<timestamp>.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write accepted repairs. Without this flag, only reports/previews are written.",
    )
    return parser.parse_args(argv)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def collapse_spaces(text: str) -> str:
    return " ".join(str(text).replace("\ufeff", " ").split())


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", unicodedata.normalize("NFKC", text))


def normalize_arabic(text: str, *, keep_spaces: bool = False) -> str:
    text = strip_diacritics(text)
    text = text.replace("ـ", "")
    text = ARABIC_PUNCTUATION_RE.sub(" ", text)
    # Matching-only normalization. Output always comes from the trusted reference.
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    text = NON_WORD_RE.sub(" ", text)
    text = collapse_spaces(text)
    return text if keep_spaces else text.replace(" ", "")


def extract_ints(value: Any) -> list[int]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, float) and value.is_integer():
        return [int(value)]
    if isinstance(value, str):
        return [int(item) for item in re.findall(r"\d+", value)]
    if isinstance(value, Mapping):
        result: list[int] = []
        for item in value.values():
            result.extend(extract_ints(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = []
        for item in value:
            result.extend(extract_ints(item))
        return result
    return []


def first_int(value: Any) -> int | None:
    values = extract_ints(value)
    return values[0] if values else None


def parse_reference_json(data: Any) -> dict[VerseKey, str]:
    result: dict[VerseKey, str] = {}

    def visit(node: Any, inherited_surah: int | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, inherited_surah)
            return
        if not isinstance(node, dict):
            return

        lowered = {str(key).lower(): value for key, value in node.items()}
        surah = inherited_surah
        for key in SURAH_KEYS:
            if key in lowered:
                surah = first_int(lowered[key]) or surah
                break

        ayah = None
        for key in ("ayah", "ayah_number", "verse", "verse_number", "number"):
            if key in lowered:
                ayah = first_int(lowered[key])
                if ayah is not None:
                    break

        text = None
        for key in ("text", "content", "verse_text", "ayah_text", "arabic"):
            value = lowered.get(key)
            if isinstance(value, str) and value.strip():
                text = value
                break

        if surah is not None and ayah is not None and text is not None:
            result[VerseKey(int(surah), int(ayah))] = collapse_spaces(text)

        for raw_key, value in node.items():
            key_text = str(raw_key)
            key_match = re.fullmatch(r"\s*(\d+)\s*[:|_-]\s*(\d+)\s*", key_text)
            if key_match and isinstance(value, str):
                result[VerseKey(int(key_match.group(1)), int(key_match.group(2)))] = collapse_spaces(value)
                continue

            child_surah = surah
            if inherited_surah is None and key_text.isdigit() and isinstance(value, (dict, list)):
                child_surah = int(key_text)
            visit(value, child_surah)

    visit(data)
    return result


def parse_reference_file(path: Path) -> dict[VerseKey, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            parsed = parse_reference_json(rows)
        else:
            parsed = parse_reference_json(json.loads(path.read_text(encoding="utf-8")))
        if parsed:
            return parsed

    result: dict[VerseKey, str] = {}
    sequential: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        match = REFERENCE_LINE_RE.match(line)
        if match:
            key = VerseKey(int(match.group(1)), int(match.group(2)))
            result[key] = collapse_spaces(match.group(3))
        else:
            sequential.append(collapse_spaces(line))

    if result:
        return result
    if sequential:
        # Fallback for one-verse-per-line files without explicit ids. Surah 0 is
        # intentionally synthetic and is aligned by order with the companion file.
        return {VerseKey(0, index): text for index, text in enumerate(sequential, start=1)}
    raise ValueError(f"No verse text could be parsed from {path}")


def load_reference(clean_path: Path, tashkeel_path: Path) -> dict[VerseKey, Verse]:
    clean = parse_reference_file(clean_path)
    tashkeel = parse_reference_file(tashkeel_path)
    common = sorted(set(clean) & set(tashkeel))

    if not common and len(clean) == len(tashkeel):
        clean_items = sorted(clean.items())
        tash_items = sorted(tashkeel.items())
        return {
            clean_key: Verse(clean_key, clean_text, tash_text)
            for (clean_key, clean_text), (_, tash_text) in zip(clean_items, tash_items)
        }

    if not common:
        raise ValueError(
            "The clean and tashkeel references have no matching verse ids. "
            "Use files with surah|ayah|text rows, compatible JSON, or equal-length "
            "one-verse-per-line files."
        )

    missing_clean = sorted(set(tashkeel) - set(clean))
    missing_tashkeel = sorted(set(clean) - set(tashkeel))
    if missing_clean or missing_tashkeel:
        print(
            "warning: reference key mismatch; only shared verses will be used "
            f"(missing_clean={len(missing_clean)}, missing_tashkeel={len(missing_tashkeel)})",
            file=sys.stderr,
        )

    return {
        key: Verse(key=key, clean=clean[key], tashkeel=tashkeel[key])
        for key in common
    }


def align_verse_tokens(verse: Verse) -> list[Token]:
    clean_tokens = collapse_spaces(verse.clean).split()
    tash_tokens = collapse_spaces(verse.tashkeel).split()
    if len(clean_tokens) == len(tash_tokens):
        return [Token(clean, tash, verse.key) for clean, tash in zip(clean_tokens, tash_tokens)]

    clean_norm = [normalize_arabic(token) for token in clean_tokens]
    tash_norm = [normalize_arabic(token) for token in tash_tokens]
    matcher = difflib.SequenceMatcher(a=clean_norm, b=tash_norm, autojunk=False)
    mapping: dict[int, str] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                mapping[i1 + offset] = tash_tokens[j1 + offset]
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for offset in range(i2 - i1):
                mapping[i1 + offset] = tash_tokens[j1 + offset]

    return [
        Token(clean, mapping.get(index, clean), verse.key)
        for index, clean in enumerate(clean_tokens)
    ]


def recursive_find_first_int(node: Any, accepted_keys: set[str]) -> int | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in accepted_keys:
                found = first_int(value)
                if found is not None:
                    return found
        for value in node.values():
            found = recursive_find_first_int(value, accepted_keys)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = recursive_find_first_int(value, accepted_keys)
            if found is not None:
                return found
    return None


def collect_explicit_ayahs(node: Any) -> list[int]:
    values: list[int] = []
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if lowered in AYAH_VALUE_KEYS or "ayah" in lowered or "verse" in lowered:
                values.extend(extract_ints(value))
            elif isinstance(value, (dict, list)):
                values.extend(collect_explicit_ayahs(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(collect_explicit_ayahs(value))
    return values


def line_number_from_key(value: Any) -> int | None:
    match = LINE_NUMBER_RE.search(str(value))
    return int(match.group(1)) if match else None


def collect_line_ayahs(node: Any, output: dict[int, set[int]]) -> None:
    if isinstance(node, list):
        for item in node:
            collect_line_ayahs(item, output)
        return
    if not isinstance(node, dict):
        return

    lowered = {str(key).lower(): value for key, value in node.items()}
    line_idx = None
    for key in ("line_idx", "line_index", "line_number", "line_no", "line"):
        if key in lowered:
            line_idx = first_int(lowered[key])
            if line_idx is not None:
                break
    if line_idx is not None:
        ayahs: list[int] = []
        for key in AYAH_VALUE_KEYS:
            if key in lowered:
                ayahs.extend(extract_ints(lowered[key]))
        if ayahs:
            output.setdefault(int(line_idx), set()).update(ayahs)

    for key, value in node.items():
        lowered_key = str(key).lower()
        if "line" in lowered_key and isinstance(value, dict):
            for maybe_line, mapping in value.items():
                idx = line_number_from_key(maybe_line)
                ayahs = collect_explicit_ayahs(mapping)
                if idx is not None and ayahs:
                    output.setdefault(idx, set()).update(ayahs)
        collect_line_ayahs(value, output)


def load_json_if_present(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read metadata {path}: {exc}", file=sys.stderr)
        return {}


def page_reference_keys(
    side_dir: Path,
    pair_dir: Path,
    reference: Mapping[VerseKey, Verse],
) -> tuple[list[VerseKey], dict[int, set[int]], str]:
    side_meta = load_json_if_present(side_dir / "page_meta.json")
    pair_meta = load_json_if_present(pair_dir / "pair_meta.json")
    metadata = [side_meta, pair_meta]

    surah = None
    for item in metadata:
        surah = recursive_find_first_int(item, SURAH_KEYS)
        if surah is not None:
            break

    line_ayahs: dict[int, set[int]] = {}
    for item in metadata:
        collect_line_ayahs(item, line_ayahs)

    ayah_start = None
    ayah_end = None
    for item in metadata:
        ayah_start = ayah_start or recursive_find_first_int(item, AYAH_START_KEYS)
        ayah_end = ayah_end or recursive_find_first_int(item, AYAH_END_KEYS)

    all_line_ayahs = sorted({ayah for values in line_ayahs.values() for ayah in values})
    if ayah_start is None and all_line_ayahs:
        ayah_start = min(all_line_ayahs)
    if ayah_end is None and all_line_ayahs:
        ayah_end = max(all_line_ayahs)

    if surah is None and len({key.surah for key in reference if key.surah != 0}) == 1:
        surah = next(key.surah for key in reference if key.surah != 0)

    if surah is None and all(key.surah == 0 for key in reference):
        # Ordered fallback references cannot be safely connected to arbitrary page
        # metadata, but they can still be used when ayah indexes are present.
        surah = 0

    if surah is None:
        return [], line_ayahs, "surah number was not found in page_meta.json or pair_meta.json"

    keys = [key for key in sorted(reference) if key.surah == surah]
    if ayah_start is not None:
        keys = [key for key in keys if key.ayah >= ayah_start]
    if ayah_end is not None:
        keys = [key for key in keys if key.ayah <= ayah_end]

    if not keys and all_line_ayahs:
        wanted = set(all_line_ayahs)
        keys = [key for key in sorted(reference) if key.surah == surah and key.ayah in wanted]

    if not keys:
        return [], line_ayahs, (
            f"no reference verses found for surah={surah}, "
            f"ayah_start={ayah_start}, ayah_end={ayah_end}"
        )
    return keys, line_ayahs, ""


def line_index(path: Path) -> int | None:
    return line_number_from_key(path.stem)


def discover_line_indices(side_dir: Path) -> list[int]:
    candidates: set[int] = set()
    roots = [
        side_dir / "linesImages",
        side_dir / "text" / "raw",
        side_dir / "text" / "final" / "original",
        side_dir / "text" / "final" / "tashkeel",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("line_*.*"):
            idx = line_index(path)
            if idx is not None:
                candidates.add(idx)
    return sorted(candidates)


def find_line_file(directory: Path, idx: int) -> Path:
    direct = directory / f"line_{idx:02d}.txt"
    if direct.exists():
        return direct
    alternatives = sorted(directory.glob(f"line_*{idx}*.txt")) if directory.is_dir() else []
    for path in alternatives:
        if line_index(path) == idx:
            return path
    return direct


def source_candidates(side_dir: Path, idx: int) -> list[SourceCandidate]:
    text_root = side_dir / "text"
    preferred = [
        ("final_original", find_line_file(text_root / "final" / "original", idx)),
        ("raw", find_line_file(text_root / "raw", idx)),
    ]

    seen: set[Path] = set()
    output: list[SourceCandidate] = []
    for kind, path in preferred:
        if path in seen:
            continue
        seen.add(path)
        text = read_text(path)
        if text:
            output.append(SourceCandidate(kind, path, text))

    if text_root.is_dir():
        for path in sorted(text_root.rglob("*.txt")):
            if path in seen or line_index(path) != idx:
                continue
            lowered = "/".join(part.lower() for part in path.parts)
            if "tashkeel" in lowered or path.name.startswith("full_"):
                continue
            kind = "llm_or_other" if "llm" in lowered else "other"
            text = read_text(path)
            if text:
                output.append(SourceCandidate(kind, path, text))
                seen.add(path)
    return output


def token_offsets(tokens: Sequence[Token]) -> dict[VerseKey, tuple[int, int]]:
    offsets: dict[VerseKey, tuple[int, int]] = {}
    for index, token in enumerate(tokens):
        if token.key not in offsets:
            offsets[token.key] = (index, index + 1)
        else:
            offsets[token.key] = (offsets[token.key][0], index + 1)
    return offsets


def candidate_bounds(
    tokens: Sequence[Token],
    offsets: Mapping[VerseKey, tuple[int, int]],
    surah: int,
    line_ayahs: set[int] | None,
    cursor: int,
) -> tuple[int, int]:
    if line_ayahs:
        ranges = [offsets[VerseKey(surah, ayah)] for ayah in sorted(line_ayahs) if VerseKey(surah, ayah) in offsets]
        if ranges:
            return max(0, min(start for start, _ in ranges) - 3), min(
                len(tokens), max(end for _, end in ranges) + 3
            )
    return max(0, cursor - 3), min(len(tokens), cursor + 45)


def score_text(source: str, candidate: str) -> float:
    source_norm = normalize_arabic(source)
    candidate_norm = normalize_arabic(candidate)
    if not source_norm or not candidate_norm:
        return 0.0
    ratio = difflib.SequenceMatcher(a=source_norm, b=candidate_norm, autojunk=False).ratio()
    length_ratio = min(len(source_norm), len(candidate_norm)) / max(len(source_norm), len(candidate_norm))
    spaced_ratio = difflib.SequenceMatcher(
        a=normalize_arabic(source, keep_spaces=True),
        b=normalize_arabic(candidate, keep_spaces=True),
        autojunk=False,
    ).ratio()
    # Character identity dominates because a common defect is a missing space that
    # collapses several Quran words into one OCR token.
    return max(0.0, min(1.0, 0.72 * ratio + 0.18 * length_ratio + 0.10 * spaced_ratio))


def best_line_match(
    sources: Sequence[SourceCandidate],
    tokens: Sequence[Token],
    offsets: Mapping[VerseKey, tuple[int, int]],
    surah: int,
    line_ayahs: set[int] | None,
    cursor: int,
) -> MatchResult | None:
    if not sources or not tokens:
        return None

    lower, upper = candidate_bounds(tokens, offsets, surah, line_ayahs, cursor)
    best: MatchResult | None = None
    for source in sources:
        source_chars = max(1, len(normalize_arabic(source.text)))
        for start in range(lower, upper):
            candidate_parts: list[str] = []
            for end in range(start + 1, min(len(tokens), upper + 30) + 1):
                candidate_parts.append(tokens[end - 1].clean)
                candidate_text = " ".join(candidate_parts)
                candidate_chars = len(normalize_arabic(candidate_text))
                if candidate_chars < max(1, int(source_chars * 0.45)):
                    continue
                if candidate_chars > int(source_chars * 1.65) + 18:
                    break
                confidence = score_text(source.text, candidate_text)
                if best is None or confidence > best.confidence:
                    best = MatchResult(
                        source=source,
                        start=start,
                        end=end,
                        confidence=confidence,
                        repaired_original=candidate_text,
                        repaired_tashkeel=" ".join(token.tashkeel for token in tokens[start:end]),
                    )
    return best


def relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    normalized_lines = [collapse_spaces(line) for line in str(text).splitlines()]
    normalized = "\n".join(line for line in normalized_lines if line)
    temp.write_text(normalized + "\n", encoding="utf-8")
    temp.replace(path)


def write_preview(report_dir: Path, dataset_root: Path, target: Path, text: str) -> None:
    preview = report_dir / "preview" / relative_to_root(target, dataset_root)
    atomic_write(preview, text)


def backup_and_write(report_dir: Path, dataset_root: Path, target: Path, text: str) -> None:
    if target.exists():
        backup = report_dir / "backup" / relative_to_root(target, dataset_root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    atomic_write(target, text)


def pair_directories(dataset_root: Path, requested: Sequence[str], max_pairs: int | None) -> list[Path]:
    root = dataset_root / "DatasetPairs" / "page_pairs"
    if not root.is_dir():
        raise FileNotFoundError(f"Page-pair directory not found: {root}")
    wanted = {item.strip() for item in requested if item.strip()}
    pairs = [path for path in sorted(root.glob("pair_*")) if path.is_dir()]
    if wanted:
        pairs = [path for path in pairs if path.name in wanted]
        missing = sorted(wanted - {path.name for path in pairs})
        if missing:
            print(f"warning: requested pair ids not found: {', '.join(missing)}", file=sys.stderr)
    if max_pairs is not None:
        pairs = pairs[: max(0, max_pairs)]
    return pairs


def process_side(
    pair_dir: Path,
    side_name: str,
    dataset_root: Path,
    report_dir: Path,
    reference: Mapping[VerseKey, Verse],
    min_confidence: float,
    apply: bool,
) -> list[AuditRow]:
    side_dir = pair_dir / side_name
    if not side_dir.is_dir():
        return []

    keys, per_line_ayahs, reference_error = page_reference_keys(side_dir, pair_dir, reference)
    line_indices = discover_line_indices(side_dir)
    if not line_indices:
        return []

    text_root = side_dir / "text"
    rows: list[AuditRow] = []
    if reference_error:
        for idx in line_indices:
            original_target = find_line_file(text_root / "final" / "original", idx)
            tashkeel_target = find_line_file(text_root / "final" / "tashkeel", idx)
            rows.append(
                AuditRow(
                    pair_id=pair_dir.name,
                    side=side_name,
                    line_idx=idx,
                    status="skipped_no_reference",
                    confidence=None,
                    source_kind=None,
                    source_path=None,
                    original_target=str(relative_to_root(original_target, dataset_root)),
                    tashkeel_target=str(relative_to_root(tashkeel_target, dataset_root)),
                    before_original=read_text(original_target),
                    before_tashkeel=read_text(tashkeel_target),
                    proposed_original="",
                    proposed_tashkeel="",
                    reference_start=None,
                    reference_end=None,
                    reason=reference_error,
                )
            )
        return rows

    tokens: list[Token] = []
    for key in keys:
        tokens.extend(align_verse_tokens(reference[key]))
    offsets = token_offsets(tokens)
    surah = keys[0].surah
    cursor = 0
    page_original: dict[int, str] = {}
    page_tashkeel: dict[int, str] = {}
    accepted_count = 0

    for idx in line_indices:
        original_target = find_line_file(text_root / "final" / "original", idx)
        tashkeel_target = find_line_file(text_root / "final" / "tashkeel", idx)
        before_original = read_text(original_target)
        before_tashkeel = read_text(tashkeel_target)
        sources = source_candidates(side_dir, idx)
        match = best_line_match(
            sources,
            tokens,
            offsets,
            surah,
            per_line_ayahs.get(idx),
            cursor,
        )

        if match is None:
            status = "skipped_no_source"
            reason = "no non-empty final/raw/LLM transcript was found for the line"
            confidence = None
            proposed_original = ""
            proposed_tashkeel = ""
            source_kind = None
            source_path = None
            reference_start = None
            reference_end = None
        else:
            confidence = match.confidence
            source_kind = match.source.kind
            source_path = str(relative_to_root(match.source.path, dataset_root))
            proposed_original = collapse_spaces(match.repaired_original)
            proposed_tashkeel = collapse_spaces(match.repaired_tashkeel)
            reference_start = f"{tokens[match.start].key.surah}:{tokens[match.start].key.ayah}"
            reference_end = f"{tokens[match.end - 1].key.surah}:{tokens[match.end - 1].key.ayah}"

            if confidence >= min_confidence:
                changed = (
                    collapse_spaces(before_original) != proposed_original
                    or collapse_spaces(before_tashkeel) != proposed_tashkeel
                )
                status = "applied" if apply and changed else "proposed" if changed else "already_correct"
                reason = "accepted sequential reference match"
                accepted_count += 1
                page_original[idx] = proposed_original
                page_tashkeel[idx] = proposed_tashkeel
                cursor = max(cursor, match.end)
                if changed:
                    write_preview(report_dir, dataset_root, original_target, proposed_original)
                    write_preview(report_dir, dataset_root, tashkeel_target, proposed_tashkeel)
                    if apply:
                        backup_and_write(report_dir, dataset_root, original_target, proposed_original)
                        backup_and_write(report_dir, dataset_root, tashkeel_target, proposed_tashkeel)
            else:
                status = "skipped_low_confidence"
                reason = f"best match {confidence:.4f} is below --min-confidence {min_confidence:.4f}"
                # A moderately plausible span still helps the next sequential line
                # without allowing this line to be written.
                if confidence >= max(0.45, min_confidence * 0.70):
                    cursor = max(cursor, match.end)

        rows.append(
            AuditRow(
                pair_id=pair_dir.name,
                side=side_name,
                line_idx=idx,
                status=status,
                confidence=confidence,
                source_kind=source_kind,
                source_path=source_path,
                original_target=str(relative_to_root(original_target, dataset_root)),
                tashkeel_target=str(relative_to_root(tashkeel_target, dataset_root)),
                before_original=before_original,
                before_tashkeel=before_tashkeel,
                proposed_original=proposed_original,
                proposed_tashkeel=proposed_tashkeel,
                reference_start=reference_start,
                reference_end=reference_end,
                reason=reason,
            )
        )

    if accepted_count:
        for idx in line_indices:
            page_original.setdefault(
                idx, read_text(find_line_file(text_root / "final" / "original", idx))
            )
            page_tashkeel.setdefault(
                idx, read_text(find_line_file(text_root / "final" / "tashkeel", idx))
            )
        full_original = text_root / "full_final_original.txt"
        full_tashkeel = text_root / "full_final_tashkeel.txt"
        rebuilt_original = "\n".join(page_original[idx] for idx in line_indices if page_original[idx])
        rebuilt_tashkeel = "\n".join(page_tashkeel[idx] for idx in line_indices if page_tashkeel[idx])
        write_preview(report_dir, dataset_root, full_original, rebuilt_original)
        write_preview(report_dir, dataset_root, full_tashkeel, rebuilt_tashkeel)
        if apply:
            backup_and_write(report_dir, dataset_root, full_original, rebuilt_original)
            backup_and_write(report_dir, dataset_root, full_tashkeel, rebuilt_tashkeel)

    return rows


def write_reports(
    report_dir: Path,
    args: argparse.Namespace,
    rows: Sequence[AuditRow],
    pair_count: int,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / "lines.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    csv_path = report_dir / "lines.csv"
    fieldnames = list(AuditRow.__dataclass_fields__)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    confidence_values = [row.confidence for row in rows if row.confidence is not None]
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "dataset_root": str(args.dataset_root),
        "reference_clean": str(args.reference_clean),
        "reference_tashkeel": str(args.reference_tashkeel),
        "min_confidence": args.min_confidence,
        "pair_count": pair_count,
        "line_count": len(rows),
        "status_counts": counts,
        "mean_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
        "manifest_rewrite_required": False,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# ArabicDataset text repair audit",
        "",
        f"- Mode: **{summary['mode']}**",
        f"- Dataset root: `{args.dataset_root}`",
        f"- Processed page pairs: **{pair_count}**",
        f"- Audited lines: **{len(rows)}**",
        f"- Minimum confidence: **{args.min_confidence:.3f}**",
        "- Raw OCR and images modified: **no**",
        "- Manifest rewrite required: **no** (the repaired files keep their paths)",
        "",
        "## Status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(counts.items()):
        markdown.append(f"| `{status}` | {count} |")
    markdown.extend(
        [
            "",
            "## Outputs",
            "",
            "- `lines.csv`: spreadsheet-friendly audit.",
            "- `lines.jsonl`: complete machine-readable audit.",
            "- `summary.json`: run configuration and counts.",
            "- `preview/`: proposed final text tree.",
            "- `backup/`: originals saved before `--apply` writes.",
            "",
            "Low-confidence rows are intentionally left unchanged. Inspect them in "
            "`lines.csv`, correct the metadata/reference source if necessary, and rerun.",
            "",
        ]
    )
    (report_dir / "AUDIT.md").write_text("\n".join(markdown), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")

    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.reference_clean = args.reference_clean.expanduser().resolve()
    args.reference_tashkeel = args.reference_tashkeel.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        args.report_dir.expanduser().resolve()
        if args.report_dir is not None
        else args.dataset_root / "text_repair_reports" / timestamp
    )

    reference = load_reference(args.reference_clean, args.reference_tashkeel)
    pairs = pair_directories(args.dataset_root, args.pair, args.max_pairs)
    if not pairs:
        raise ValueError("No page-pair directories matched the request")

    rows: list[AuditRow] = []
    for pair_dir in pairs:
        for side in ("A", "B"):
            rows.extend(
                process_side(
                    pair_dir=pair_dir,
                    side_name=side,
                    dataset_root=args.dataset_root,
                    report_dir=report_dir,
                    reference=reference,
                    min_confidence=args.min_confidence,
                    apply=args.apply,
                )
            )

    write_reports(report_dir, args, rows, len(pairs))
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"pairs={len(pairs)} lines={len(rows)}")
    print(f"report={report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

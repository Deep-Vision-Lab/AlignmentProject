#!/usr/bin/env python3
"""Build Quran references from ArabicDataset/Surahs and run text repair.

This wrapper removes the need for external clean/tashkeel reference files. It
extracts verses from the dataset's existing ``Surahs/surah_*.json`` files,
derives clean text by removing Arabic diacritics when necessary, writes
deterministic reference files, and invokes ``repair_alignment_texts.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ARABIC_DIACRITICS_RE = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
SURAH_FILENAME_RE = re.compile(r"surah[_\-\s]*(\d+)", re.IGNORECASE)
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")

AYAH_NUMBER_KEYS = (
    "number_in_surah",
    "numberinsurah",
    "ayah_number",
    "ayahnumber",
    "ayah_no",
    "ayahno",
    "aya_number",
    "ayanumber",
    "aya_no",
    "ayano",
    "verse_number",
    "versenumber",
    "verse_no",
    "verseno",
    "index",
    "id",
    "number",
)

CLEAN_TEXT_KEYS = (
    "text_clean",
    "clean_text",
    "text_simple",
    "simple_text",
    "text_original",
    "original_text",
    "without_tashkeel",
    "without_diacritics",
    "clean",
    "simple",
    "original",
)

TASHKEEL_TEXT_KEYS = (
    "text_tashkeel",
    "tashkeel_text",
    "text_uthmani",
    "uthmani_text",
    "text_uthmanic",
    "uthmanic_text",
    "with_tashkeel",
    "with_diacritics",
    "tashkeel",
    "uthmani",
    "uthmanic",
)

GENERIC_TEXT_KEYS = (
    "text",
    "arabic_text",
    "arabic",
    "content",
    "verse_text",
    "ayah_text",
    "aya_text",
)

VERSE_CONTAINER_KEYS = (
    "ayahs",
    "ayat",
    "verses",
    "ayas",
    "data",
    "items",
)


@dataclass
class VerseText:
    clean: str
    tashkeel: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use DataSet/ArabicDataset/Surahs/surah_*.json as the Quran "
            "reference, then audit or apply ArabicDataset line-text repairs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("DataSet/ArabicDataset"),
        help="ArabicDataset root (default: DataSet/ArabicDataset).",
    )
    parser.add_argument(
        "--surahs-dir",
        type=Path,
        default=None,
        help="Surah JSON directory (default: DATASET_ROOT/Surahs).",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Only process this pair id; repeat for multiple pairs.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.78,
        help="Minimum confidence accepted by the repair tool (default: 0.78).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional deterministic cap for a small smoke test.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional report directory passed to the repair tool.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply accepted repairs. Without this flag the run is audit-only.",
    )
    parser.add_argument(
        "--references-only",
        action="store_true",
        help="Only generate the two derived reference files; do not run repair.",
    )
    return parser.parse_args(argv)


def collapse_spaces(text: str) -> str:
    return " ".join(str(text).replace("\ufeff", " ").split())


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return ARABIC_DIACRITICS_RE.sub("", text).replace("ـ", "")


def is_arabic_text(value: Any) -> bool:
    return isinstance(value, str) and bool(ARABIC_RE.search(value))


def normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def first_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        integer = int(value)
        return integer if integer > 0 else None
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            integer = int(match.group())
            return integer if integer > 0 else None
    return None


def infer_surah_number(path: Path, data: Any) -> int:
    match = SURAH_FILENAME_RE.search(path.stem)
    if match:
        return int(match.group(1))

    if isinstance(data, Mapping):
        normalized = {normalize_key(key): value for key, value in data.items()}
        for key in (
            "surahnumber",
            "suranumber",
            "chapternumber",
            "surahno",
            "surano",
            "number",
            "id",
        ):
            number = first_positive_int(normalized.get(key))
            if number is not None:
                return number

    raise ValueError(
        f"Could not infer a surah number from filename or JSON metadata: {path}"
    )


def find_text(node: Mapping[str, Any], keys: Iterable[str]) -> str:
    normalized = {normalize_key(key): value for key, value in node.items()}
    for key in keys:
        value = normalized.get(normalize_key(key))
        if is_arabic_text(value):
            return collapse_spaces(value)
    return ""


def find_ayah_number(node: Mapping[str, Any]) -> int | None:
    normalized = {normalize_key(key): value for key, value in node.items()}
    for key in AYAH_NUMBER_KEYS:
        number = first_positive_int(normalized.get(normalize_key(key)))
        if number is not None:
            return number
    return None


def choose_texts(node: Mapping[str, Any]) -> VerseText | None:
    clean = find_text(node, CLEAN_TEXT_KEYS)
    tashkeel = find_text(node, TASHKEEL_TEXT_KEYS)
    generic = find_text(node, GENERIC_TEXT_KEYS)

    if not clean and not tashkeel and not generic:
        return None

    if not tashkeel:
        tashkeel = generic or clean
    if not clean:
        clean = strip_diacritics(generic or tashkeel)

    clean = collapse_spaces(strip_diacritics(clean))
    tashkeel = collapse_spaces(tashkeel)
    if not clean or not tashkeel:
        return None
    return VerseText(clean=clean, tashkeel=tashkeel)


def verse_quality(verse: VerseText) -> tuple[int, int]:
    """Prefer a vocalized tashkeel form, then the longer non-empty text."""
    diacritics = len(ARABIC_DIACRITICS_RE.findall(verse.tashkeel))
    return diacritics, len(verse.tashkeel)


def collect_verses(data: Any, surah_number: int) -> dict[int, VerseText]:
    verses: dict[int, VerseText] = {}

    def add(ayah_number: int | None, verse: VerseText | None) -> None:
        if ayah_number is None or verse is None:
            return
        current = verses.get(ayah_number)
        if current is None or verse_quality(verse) > verse_quality(current):
            verses[ayah_number] = verse

    def visit(node: Any, fallback_ayah: int | None = None) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node, start=1):
                visit(item, index)
            return

        if not isinstance(node, Mapping):
            return

        add(find_ayah_number(node) or fallback_ayah, choose_texts(node))

        # Support maps such as {"1": "verse text", "2": "verse text"}.
        for raw_key, value in node.items():
            mapped_ayah = first_positive_int(raw_key)
            if mapped_ayah is not None and is_arabic_text(value):
                text = collapse_spaces(value)
                add(
                    mapped_ayah,
                    VerseText(
                        clean=collapse_spaces(strip_diacritics(text)),
                        tashkeel=text,
                    ),
                )

        normalized_items = {
            normalize_key(key): value for key, value in node.items()
        }
        visited_ids: set[int] = set()

        # Visit likely verse containers first so list indexes can serve as a
        # reliable fallback when a verse object omits its local ayah number.
        for key in VERSE_CONTAINER_KEYS:
            child = normalized_items.get(normalize_key(key))
            if isinstance(child, (list, Mapping)):
                visited_ids.add(id(child))
                visit(child)

        for child in node.values():
            if isinstance(child, (list, Mapping)) and id(child) not in visited_ids:
                visit(child)

    visit(data)
    if not verses:
        raise ValueError(
            f"No verse records were detected for surah {surah_number}. "
            "The JSON must contain Arabic verse text and either local ayah "
            "numbers or an ordered verse list."
        )
    return verses


def load_surahs(surahs_dir: Path) -> dict[tuple[int, int], VerseText]:
    files = sorted(surahs_dir.glob("surah_*.json"))
    if not files:
        files = sorted(surahs_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No Surah JSON files found in: {surahs_dir}")

    output: dict[tuple[int, int], VerseText] = {}
    failures: list[str] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            surah_number = infer_surah_number(path, data)
            verses = collect_verses(data, surah_number)
            for ayah_number, verse in verses.items():
                output[(surah_number, ayah_number)] = verse
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"{path.name}: {exc}")

    if not output:
        rendered = "\n  - ".join(failures)
        raise ValueError(
            "No Quran verses could be extracted from the Surahs directory."
            + (f"\n  - {rendered}" if rendered else "")
        )

    if failures:
        print(
            "warning: some Surah files were skipped:\n  - "
            + "\n  - ".join(failures),
            file=sys.stderr,
        )
    return output


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_references(
    dataset_root: Path,
    verses: Mapping[tuple[int, int], VerseText],
) -> tuple[Path, Path]:
    reference_dir = dataset_root / "text_repair_references"
    clean_path = reference_dir / "quran_from_surahs_clean.txt"
    tashkeel_path = reference_dir / "quran_from_surahs_tashkeel.txt"

    clean_lines: list[str] = []
    tashkeel_lines: list[str] = []
    for (surah, ayah), verse in sorted(verses.items()):
        clean_lines.append(f"{surah}|{ayah}|{verse.clean}")
        tashkeel_lines.append(f"{surah}|{ayah}|{verse.tashkeel}")

    atomic_write(clean_path, "\n".join(clean_lines) + "\n")
    atomic_write(tashkeel_path, "\n".join(tashkeel_lines) + "\n")
    return clean_path, tashkeel_path


def run_repair(args: argparse.Namespace, clean: Path, tashkeel: Path) -> int:
    repair_script = Path(__file__).with_name("repair_alignment_texts.py")
    if not repair_script.is_file():
        raise FileNotFoundError(f"Repair script not found: {repair_script}")

    command = [
        sys.executable,
        str(repair_script),
        "--dataset-root",
        str(args.dataset_root),
        "--reference-clean",
        str(clean),
        "--reference-tashkeel",
        str(tashkeel),
        "--min-confidence",
        str(args.min_confidence),
    ]
    for pair_id in args.pair:
        command.extend(["--pair", pair_id])
    if args.max_pairs is not None:
        command.extend(["--max-pairs", str(args.max_pairs)])
    if args.report_dir is not None:
        command.extend(["--report-dir", str(args.report_dir)])
    if args.apply:
        command.append("--apply")

    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.surahs_dir = (
        args.surahs_dir.expanduser().resolve()
        if args.surahs_dir is not None
        else args.dataset_root / "Surahs"
    )
    if args.report_dir is not None:
        args.report_dir = args.report_dir.expanduser().resolve()

    verses = load_surahs(args.surahs_dir)
    clean, tashkeel = write_references(args.dataset_root, verses)

    print(f"verses={len(verses)}")
    print(f"reference_clean={clean}")
    print(f"reference_tashkeel={tashkeel}")

    if args.references_only:
        return 0
    return run_repair(args, clean, tashkeel)


if __name__ == "__main__":
    raise SystemExit(main())

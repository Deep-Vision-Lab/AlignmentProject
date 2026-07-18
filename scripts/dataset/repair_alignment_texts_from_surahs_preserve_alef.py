#!/usr/bin/env python3
"""Repair ArabicDataset text with a verified standard-Imlaei Quran reference.

The local ``Surahs/surah_*.json`` files may contain only Uthmani text. Uthmani
superscript/dagger alef cannot be converted mechanically into modern spelling:
``بَعَثْنَـٰكُمْ`` must become ``بعثناكم``, while ``هَٰذَا`` must remain ``هذا``.

This wrapper therefore downloads Tanzil's verified ``simple-clean`` (Imlaei)
Quran text once, caches it under ``DATASET_ROOT/text_repair_references/``, and
uses it as the clean reference. The local Surah JSON remains the source for the
vocalized/tashkeel reference. Subsequent runs use the cached Tanzil file.

It accepts the same command-line arguments as
``repair_alignment_texts_from_surahs.py``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

EXPECTED_VERSE_COUNT = 6236
TANZIL_DOWNLOAD_URLS = (
    "https://tanzil.net/pub/download/download.php",
    "http://tanzil.net/pub/download/download.php",
)
CACHE_FILENAME = "tanzil_quran_simple_clean.txt"
NOTICE_FILENAME = "TANZIL_SOURCE_AND_LICENSE.txt"

TANZIL_NOTICE = """Tanzil Quran Text source and license notice

Source: https://tanzil.net/
Download page: https://tanzil.net/download/
Text type: Simple Clean (Imlaei, without diacritics or symbols)

This Quran text is distributed under the Creative Commons Attribution 3.0
License. Permission is granted to copy and distribute verbatim copies of the
text, but changing the downloaded reference text is not allowed. When this text
is used in an application or dataset, Tanzil.net must be identified as the
source and linked so users can track text updates.

The cached file in this directory is retained verbatim. It is used as a
reference to repair line-level dataset transcripts; it is not rewritten.
"""


def _load_base_module():
    base_path = Path(__file__).with_name("repair_alignment_texts_from_surahs.py")
    if not base_path.is_file():
        raise FileNotFoundError(f"Base Surahs repair wrapper not found: {base_path}")

    module_name = "_alignment_repair_from_surahs_base"
    spec = importlib.util.spec_from_file_location(module_name, base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the base Surahs repair wrapper: {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_tanzil_simple_clean(text: str) -> dict[tuple[int, int], str]:
    verses: dict[tuple[int, int], str] = {}
    for line_number, raw_line in enumerate(text.lstrip("\ufeff").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        surah_raw, ayah_raw, verse_text = parts
        try:
            surah = int(surah_raw.strip())
            ayah = int(ayah_raw.strip())
        except ValueError:
            continue

        verse_text = " ".join(verse_text.split())
        if not verse_text:
            raise ValueError(
                f"Empty Tanzil verse at downloaded line {line_number}: {surah}:{ayah}"
            )
        verses[(surah, ayah)] = verse_text

    if len(verses) != EXPECTED_VERSE_COUNT:
        raise ValueError(
            "Invalid Tanzil Simple Clean reference: expected "
            f"{EXPECTED_VERSE_COUNT} verses, found {len(verses)}."
        )

    expected_samples = {
        (1, 1): "بسم الله الرحمن الرحيم",
        (2, 56): "بعثناكم",
    }
    for key, expected_fragment in expected_samples.items():
        actual = verses.get(key, "")
        if expected_fragment not in actual:
            raise ValueError(
                "Tanzil reference validation failed for "
                f"{key[0]}:{key[1]}; expected to find {expected_fragment!r}, "
                f"got {actual!r}."
            )
    return verses


def _download_tanzil_text() -> str:
    # These parameters match Tanzil's official download form. False optional
    # switches are omitted so pause/rub/sajdah marks are not requested.
    payload = urllib.parse.urlencode(
        {
            "quranType": "simple-clean",
            "outType": "txt-2",
            "agree": "true",
        }
    ).encode("ascii")

    errors: list[str] = []
    for url in TANZIL_DOWNLOAD_URLS:
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "User-Agent": "AlignmentProject-ArabicDataset-TextRepair/1.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            text = content.decode("utf-8-sig")
            _parse_tanzil_simple_clean(text)
            return text
        except (
            OSError,
            UnicodeDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            ValueError,
        ) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Could not download the Tanzil Simple Clean Quran reference. "
        "Check that this machine has outbound internet access. Attempts:\n  - "
        + "\n  - ".join(errors)
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_verified_imlaei(dataset_root: Path) -> dict[tuple[int, int], str]:
    reference_dir = dataset_root / "text_repair_references"
    cache_path = reference_dir / CACHE_FILENAME
    notice_path = reference_dir / NOTICE_FILENAME
    refresh = os.environ.get("REFRESH_TANZIL_QURAN", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if cache_path.is_file() and not refresh:
        try:
            verses = _parse_tanzil_simple_clean(
                cache_path.read_text(encoding="utf-8")
            )
            print(f"Using cached Tanzil Imlaei reference: {cache_path}", flush=True)
            if not notice_path.is_file():
                _atomic_write(notice_path, TANZIL_NOTICE)
            return verses
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(
                f"warning: cached Tanzil reference is invalid and will be refreshed: {exc}",
                file=sys.stderr,
            )

    print(
        "Downloading Tanzil Simple Clean Quran text for standard Imlaei spelling...",
        flush=True,
    )
    downloaded = _download_tanzil_text()
    verses = _parse_tanzil_simple_clean(downloaded)
    _atomic_write(cache_path, downloaded)
    _atomic_write(notice_path, TANZIL_NOTICE)
    print(f"Cached Tanzil Imlaei reference: {cache_path}", flush=True)
    return verses


def main(argv: Sequence[str] | None = None) -> int:
    base_module = _load_base_module()
    original_load_surahs = base_module.load_surahs

    def load_surahs_with_verified_imlaei(surahs_dir: Path):
        local_verses = original_load_surahs(surahs_dir)
        dataset_root = surahs_dir.resolve().parent
        imlaei = _load_verified_imlaei(dataset_root)

        missing = sorted(set(local_verses) - set(imlaei))
        if missing:
            rendered = ", ".join(f"{surah}:{ayah}" for surah, ayah in missing[:20])
            raise ValueError(
                "The local Surah data contains verse keys missing from the Tanzil "
                f"reference ({len(missing)} missing; first keys: {rendered})."
            )

        for key, verse in local_verses.items():
            verse.clean = imlaei[key]
            setattr(verse, "_spelling_source", "tanzil_simple_clean")

        sample = local_verses.get((2, 56))
        if sample is None or "بعثناكم" not in sample.clean:
            raise ValueError(
                "Final Imlaei validation failed: verse 2:56 does not contain بعثناكم."
            )

        print(
            "Standard-spelling references: "
            f"tanzil_imlaei={len(local_verses)} local_tashkeel={len(local_verses)}",
            flush=True,
        )
        print(f"Validation 2:56: {sample.clean}", flush=True)
        return local_verses

    base_module.load_surahs = load_surahs_with_verified_imlaei

    print(
        "Verified Imlaei mode: clean spelling comes from Tanzil Simple Clean; "
        "vocalized text comes from the local Surah JSON.",
        flush=True,
    )
    return int(base_module.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

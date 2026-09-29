"""The evaluation dataset: a folder of WAVs plus ``manifest.jsonl``.

One manifest line per utterance::

    {"id": "00", "file": "00.vi_short.wav", "text": "anh em", "lang": "vi",
     "category": "vi_short", "en_words": [], "duration_s": 1.41}

``dataset_hash`` covers the manifest text and every WAV's bytes so a fixture
edit can never be confused with a model change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "eval" / "stt-fixtures"
MANIFEST_NAME = "manifest.jsonl"


@dataclass(frozen=True)
class Utterance:
    id: str
    file: str
    text: str
    lang: str            # "vi" | "en"
    category: str        # e.g. "vi_short"
    en_words: tuple[str, ...]
    duration_s: float


def load(fixtures_dir: Path) -> list[Utterance]:
    path = Path(fixtures_dir) / MANIFEST_NAME
    utts: list[Utterance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        utts.append(Utterance(
            id=str(row["id"]),
            file=row["file"],
            text=row["text"],
            lang=row["lang"],
            category=row["category"],
            en_words=tuple(row.get("en_words") or ()),
            duration_s=float(row["duration_s"]),
        ))
    return utts


def audio_bytes(fixtures_dir: Path, utt: Utterance) -> bytes:
    return (Path(fixtures_dir) / utt.file).read_bytes()


def dataset_hash(fixtures_dir: Path) -> str:
    root = Path(fixtures_dir)
    h = hashlib.sha256()
    h.update((root / MANIFEST_NAME).read_bytes())
    for utt in load(root):
        h.update(utt.file.encode("utf-8"))
        h.update((root / utt.file).read_bytes())
    return f"sha256:{h.hexdigest()}"

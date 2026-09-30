"""The TTS sentence set: ``manifest.jsonl`` plus the reference clip every cloning system copies.

One manifest line per sentence::

    {"id": "mx-02", "text": "Bé ơi, dog nghĩa là con chó đó.", "lang": "vi",
     "category": "mix", "en_words": ["dog"]}

``alt_texts`` (optional) lists other correct readings, e.g. numbers spelled out; a judge's
transcript is scored against whichever reading it is closest to. ``reference.wav`` and
``reference.json`` (its transcript and provenance) sit next to the manifest. ``dataset_hash``
covers all three so a sentence or reference edit can never be confused with a model change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SENTENCES_DIR = Path(__file__).resolve().parents[4] / "eval" / "tts-sentences"
MANIFEST_NAME = "manifest.jsonl"
REFERENCE_WAV = "reference.wav"
REFERENCE_META = "reference.json"


@dataclass(frozen=True)
class Sentence:
    id: str
    text: str
    lang: str            # request language: "vi" (also for code-switched sentences) | "en"
    category: str        # e.g. "vi_short", "mix"
    en_words: tuple[str, ...]
    alt_texts: tuple[str, ...] = ()

    @property
    def readings(self) -> tuple[str, ...]:
        return (self.text, *self.alt_texts)


@dataclass(frozen=True)
class Reference:
    wav: bytes
    text: str


def load(sentences_dir: Path) -> list[Sentence]:
    out: list[Sentence] = []
    for line in (Path(sentences_dir) / MANIFEST_NAME).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(Sentence(
            id=str(row["id"]),
            text=row["text"],
            lang=row["lang"],
            category=row["category"],
            en_words=tuple(row.get("en_words") or ()),
            alt_texts=tuple(row.get("alt_texts") or ()),
        ))
    ids = [s.id for s in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate sentence ids in {sentences_dir}/{MANIFEST_NAME}")
    return out


def reference(sentences_dir: Path) -> Reference:
    root = Path(sentences_dir)
    meta = json.loads((root / REFERENCE_META).read_text(encoding="utf-8"))
    return Reference(wav=(root / REFERENCE_WAV).read_bytes(), text=meta["text"])


def dataset_hash(sentences_dir: Path) -> str:
    root = Path(sentences_dir)
    h = hashlib.sha256()
    for name in (MANIFEST_NAME, REFERENCE_META, REFERENCE_WAV):
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return f"sha256:{h.hexdigest()}"

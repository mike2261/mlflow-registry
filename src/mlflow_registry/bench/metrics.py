"""Pooled STT metrics. Every rate is Σ edits / Σ reference units, never a mean of rates."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import jiwer

from mlflow_registry.bench.manifest import Utterance
from mlflow_registry.bench.normalize import normalize, tokens


def score_text(ref: str, hyp: str) -> tuple[int, int, int, int, bool]:
    """Return (word_edits, ref_words, char_edits, ref_chars, exact) after normalization."""
    r, h = normalize(ref), normalize(hyp)
    words = jiwer.process_words(r, h)
    word_edits = words.substitutions + words.deletions + words.insertions
    ref_words = words.substitutions + words.deletions + words.hits
    rc, hc = r.replace(" ", ""), h.replace(" ", "")
    if hc:
        chars = jiwer.process_characters(rc, hc)
        char_edits = chars.substitutions + chars.deletions + chars.insertions
    else:  # jiwer rejects an empty hypothesis; every reference char is a deletion
        char_edits = len(rc)
    return word_edits, ref_words, char_edits, len(rc), r == h


def en_recall_counts(en_words: Sequence[str], hyp: str) -> tuple[int, int]:
    if not en_words:
        return 0, 0
    present = set(tokens(hyp))
    return sum(1 for w in en_words if normalize(w) in present), len(en_words)


@dataclass(frozen=True)
class Scored:
    utt: Utterance
    hyp: str | None
    error: str | None
    word_edits: int
    ref_words: int
    char_edits: int
    ref_chars: int
    exact: bool
    en_hits: int
    en_total: int
    latency_s: float | None

    @property
    def failed(self) -> bool:
        return self.error is not None


def scored(utt: Utterance, hyp: str | None, error: str | None, latency_s: float | None) -> Scored:
    if error is None and hyp is not None and normalize(hyp):
        word_edits, ref_words, char_edits, ref_chars, exact = score_text(utt.text, hyp)
        en_hits, en_total = en_recall_counts(utt.en_words, hyp)
        return Scored(utt, hyp, None, word_edits, ref_words, char_edits, ref_chars, exact,
                      en_hits, en_total, latency_s)
    reason = error or "empty transcript"
    return Scored(utt, hyp, reason, 0, 0, 0, 0, False, 0, 0, latency_s)


def median(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def p90(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(0.9 * len(s)) - 1)]


@dataclass(frozen=True)
class Aggregate:
    n: int
    failures: int
    word_edits: int
    ref_words: int
    char_edits: int
    ref_chars: int
    exact: int
    en_hits: int
    en_total: int
    latency_median_s: float | None
    latency_p90_s: float | None
    audio_s: float
    latency_sum_s: float

    @property
    def wer(self) -> float | None:
        return self.word_edits / self.ref_words if self.ref_words else None

    @property
    def cer(self) -> float | None:
        return self.char_edits / self.ref_chars if self.ref_chars else None

    @property
    def exact_rate(self) -> float | None:
        return self.exact / self.n if self.n else None

    @property
    def en_recall(self) -> float | None:
        return self.en_hits / self.en_total if self.en_total else None

    @property
    def rtf(self) -> float | None:
        return self.latency_sum_s / self.audio_s if self.audio_s else None


def aggregate(rows: Iterable[Scored]) -> Aggregate:
    rows = list(rows)
    ok = [r for r in rows if not r.failed]
    lat = [r.latency_s for r in rows if r.latency_s is not None]
    return Aggregate(
        n=len(rows),
        failures=len(rows) - len(ok),
        word_edits=sum(r.word_edits for r in ok),
        ref_words=sum(r.ref_words for r in ok),
        char_edits=sum(r.char_edits for r in ok),
        ref_chars=sum(r.ref_chars for r in ok),
        exact=sum(1 for r in ok if r.exact),
        en_hits=sum(r.en_hits for r in ok),
        en_total=sum(r.en_total for r in ok),
        latency_median_s=median(lat),
        latency_p90_s=p90(lat),
        audio_s=sum(r.utt.duration_s for r in rows if r.latency_s is not None),
        latency_sum_s=sum(lat),
    )

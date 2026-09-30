"""Turn a TTS run directory into per-system results and a Markdown report.

Intelligibility reuses the STT scoring: each judge's transcript is scored against the sentence
with ``score_text`` (whichever reading in ``Sentence.readings`` it is closest to), pooled into an
``Aggregate``. Unlike STT, a failed synthesis or an empty transcript is not dropped: it counts as
every reference word deleted, so a system cannot improve its WER by failing.

Systems cover different languages (kokoro English only, vieneu Vietnamese only), so headline
tables are per language and systems are only ranked against systems that speak it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mlflow_registry.bench.collect import RUN_META
from mlflow_registry.bench.manifest import Utterance
from mlflow_registry.bench.metrics import Aggregate, Scored, aggregate, en_recall_counts, median, score_text
from mlflow_registry.bench.normalize import NORMALIZER_VERSION, normalize
from mlflow_registry.bench.score import DatasetMismatch, _cell, _frac, _pct, _sec
from mlflow_registry.bench.tts.backends import CLONING, GOOGLE_NAME
from mlflow_registry.bench.tts.judge import SPEAKER_MODEL
from mlflow_registry.bench.tts.manifest import Sentence

CHIRP3_HD_USD_PER_CHAR = 30 / 1_000_000
PRICE_CHECKED = "2026-09-30"
PRICE_URL = "https://cloud.google.com/text-to-speech/pricing"
DURATION_FLAG_HIGH = 2.0
DURATION_FLAG_LOW = 0.5
WORST_LIMIT = 10


@dataclass
class Row:
    sentence: Sentence
    error: str | None                        # synthesis failure (pass 1)
    latency_s: float | None                  # median over passes
    duration_s: float | None
    audio: str | None
    hyps: dict[str, str | None] = field(default_factory=dict)       # judge -> transcript
    scored: dict[str, Scored] = field(default_factory=dict)         # judge -> scored
    utmos: float | None = None
    spk_sim: float | None = None
    duration_flag: str | None = None

    def wer(self, judges: list[str]) -> float | None:
        vals = [self.scored[j].wer_utt for j in judges if j in self.scored and self.scored[j].wer_utt is not None]
        return sum(vals) / len(vals) if vals else None


@dataclass
class Result:
    system: str
    rows: list[Row]
    backend: dict
    passes: int
    requests: int
    failed_requests: int

    @property
    def failures(self) -> int:
        return sum(1 for r in self.rows if r.error is not None)


# --- loading -----------------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check(path: Path, rows: list[dict], dataset_hash: str) -> list[dict]:
    for r in rows:
        if r.get("dataset_hash") != dataset_hash:
            raise DatasetMismatch(f"{path} was produced on dataset {r.get('dataset_hash')}, "
                                  f"but {RUN_META} pins {dataset_hash}")
    return rows


@dataclass
class Run:
    meta: dict
    synth: dict[str, list[dict]]                       # system -> records (all passes)
    asr: dict[str, dict[str, dict[str, dict]]]         # judge -> system -> sent -> record
    quality: dict[str, dict[str, dict]]                # system -> sent -> record

    @property
    def judges(self) -> list[str]:
        return sorted(self.asr)


def load_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / RUN_META).read_text(encoding="utf-8"))
    h = meta["dataset_hash"]
    synth = {p.stem: _check(p, _read_jsonl(p), h) for p in sorted(run_dir.glob("*.jsonl"))}
    asr: dict[str, dict[str, dict[str, dict]]] = defaultdict(dict)
    for p in sorted((run_dir / "asr").glob("*/*.jsonl")):
        asr[p.parent.name][p.stem] = {r["sent"]: r for r in _check(p, _read_jsonl(p), h)}
    quality = {p.stem: {r["sent"]: r for r in _check(p, _read_jsonl(p), h)}
               for p in sorted((run_dir / "quality").glob("*.jsonl"))}
    return Run(meta, synth, dict(asr), quality)


# --- scoring -----------------------------------------------------------------------------------

def _utt(s: Sentence, text: str, duration_s: float) -> Utterance:
    return Utterance(id=s.id, file="", text=text, lang=s.lang, category=s.category,
                     en_words=s.en_words, duration_s=duration_s)


def score_sentence(s: Sentence, hyp: str | None, latency_s: float | None, duration_s: float) -> Scored:
    """Score against the closest reading; a missing or empty transcript deletes every word."""
    hyp = hyp if hyp and normalize(hyp) else ""
    best = None
    for reading in s.readings:
        ref_n = normalize(reading)
        if hyp:
            we, rw, ce, rc, exact = score_text(reading, hyp)
        else:
            rw = len(ref_n.split())
            we, ce, rc, exact = rw, len(ref_n.replace(" ", "")), len(ref_n.replace(" ", "")), False
        if best is None or we / max(rw, 1) < best[0] / max(best[1], 1):
            best = (we, rw, ce, rc, exact, reading)
    we, rw, ce, rc, exact, reading = best
    en_hits, en_total = en_recall_counts(s.en_words, hyp) if s.en_words else (0, 0)
    return Scored(_utt(s, reading, duration_s), hyp, None, we, rw, ce, rc, exact, en_hits, en_total, latency_s)


def build_results(run: Run, sentences: list[Sentence]) -> list[Result]:
    by_id = {s.id: s for s in sentences}
    results = []
    for system, records in sorted(run.synth.items()):
        by_sent: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_sent[r["sent"]].append(r)
        rows = []
        for sid, recs in by_sent.items():
            s = by_id[sid]
            first = min(recs, key=lambda r: r["pass"])
            lat = median([r["latency_s"] for r in recs if r["latency_s"] is not None and r["error"] is None])
            row = Row(s, first["error"], lat, first["duration_s"], first["audio"])
            for judge in run.judges:
                a = run.asr[judge].get(system, {}).get(sid)
                hyp = None if a is None or a["error"] else a["text"]
                row.hyps[judge] = hyp
                if a is not None:
                    row.scored[judge] = score_sentence(s, None if row.error else hyp, lat,
                                                       row.duration_s or 0.0)
            q = run.quality.get(system, {}).get(sid)
            if q:
                row.utmos, row.spk_sim = q["utmos"], q["spk_sim"]
            rows.append(row)
        rows.sort(key=lambda r: [x.id for x in sentences].index(r.sentence.id))
        results.append(Result(system, rows, records[0].get("backend", {}),
                              passes=max(r["pass"] for r in records), requests=len(records),
                              failed_requests=sum(1 for r in records if r["error"] is not None)))
    flag_durations(results)
    return results


def flag_durations(results: list[Result]) -> None:
    """Flag audio far longer or shorter per character than the median of the other systems."""
    per_char: dict[str, dict[str, float]] = defaultdict(dict)
    for res in results:
        for row in res.rows:
            chars = len(normalize(row.sentence.text).replace(" ", ""))
            if row.error is None and row.duration_s and chars:
                per_char[row.sentence.id][res.system] = row.duration_s / chars
    for res in results:
        for row in res.rows:
            values = per_char.get(row.sentence.id, {})
            mine = values.get(res.system)
            others = [v for k, v in values.items() if k != res.system]
            if mine is None or not others:
                continue
            ratio = mine / median(others)
            if ratio > DURATION_FLAG_HIGH:
                row.duration_flag = f"{ratio:.1f}x longer"
            elif ratio < DURATION_FLAG_LOW:
                row.duration_flag = f"{ratio:.1f}x shorter"


def agg(rows: list[Row], judge: str) -> Aggregate:
    return aggregate(r.scored[judge] for r in rows if judge in r.scored)


def mean_of(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def wer(rows: list[Row], judges: list[str]) -> float | None:
    """Mean over judges of each judge's pooled WER."""
    return mean_of([agg(rows, j).wer for j in judges])


def en_recall(rows: list[Row], judges: list[str]) -> float | None:
    return mean_of([agg(rows, j).en_recall for j in judges])


def cost_per_1k(system: str, sentences: list[Sentence]) -> float | None:
    if system != GOOGLE_NAME or not sentences:
        return None
    chars = sum(len(s.text) for s in sentences) / len(sentences)
    return chars * 1000 * CHIRP3_HD_USD_PER_CHAR


def rows_for(res: Result, lang: str | None = None, category: str | None = None) -> list[Row]:
    return [r for r in res.rows
            if (lang is None or r.sentence.lang == lang) and (category is None or r.sentence.category == category)]


def speaks(res: Result, lang: str) -> bool:
    """Whether the system was collected on that language (collect skips unsupported ones)."""
    return any(r.sentence.lang == lang for r in res.rows)


# --- report ------------------------------------------------------------------------------------

def _num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _judge_cols(judges: list[str]) -> str:
    return " | ".join(f"WER ({j})" for j in judges)


def _language_table(results: list[Result], judges: list[str], lang: str) -> list[str]:
    ranked = [r for r in results if speaks(r, lang)]
    ranked.sort(key=lambda r: (wer(rows_for(r, lang), judges) is None, wer(rows_for(r, lang), judges) or 0))
    out = [f"| # | System | Mean WER | {_judge_cols(judges)} | CER | EN recall (mix) | UTMOS | Speaker sim | "
           "Duration flags | Failed | Median latency | p95 | RTF | $/1k sentences |",
           "|" + "---|" * (13 + len(judges))]
    for i, res in enumerate(ranked, 1):
        rows = rows_for(res, lang)
        mix = rows_for(res, lang, "mix")
        first = agg(rows, judges[0]) if judges else None
        cer = mean_of([agg(rows, j).cer for j in judges])
        cost = cost_per_1k(res.system, [r.sentence for r in rows])
        out.append("| " + " | ".join([
            str(i), res.system, _frac(wer(rows, judges)),
            *[_frac(agg(rows, j).wer) for j in judges],
            _frac(cer),
            _frac(en_recall(mix, judges)) if mix else "—",
            _num(mean_of([r.utmos for r in rows])),
            _num(mean_of([r.spk_sim for r in rows])) if res.system in CLONING else "n/a",
            str(sum(1 for r in rows if r.duration_flag)),
            _pct(sum(1 for r in rows if r.error), len(rows)),
            _sec(first.latency_median_s if first else None),
            _sec(first.latency_p95_s if first else None),
            _num(first.rtf if first else None),
            "—" if cost is None else f"${cost:.2f}",
        ]) + " |")
    return out


def _category_table(results: list[Result], judges: list[str], sentences: list[Sentence]) -> list[str]:
    cats = list(dict.fromkeys(s.category for s in sentences))
    out = ["| System | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for res in results:
        cells = []
        for c in cats:
            rows = rows_for(res, category=c)
            cells.append(_frac(wer(rows, judges)) if rows else "—")
        out.append(f"| {res.system} | " + " | ".join(cells) + " |")
    return out


def category_winners(results: list[Result], judges: list[str], sentences: list[Sentence]) -> dict[str, list[str]]:
    """Lowest mean WER per category among systems that speak it; on ``mix`` EN recall comes first."""
    winners: dict[str, list[str]] = {}
    for c in dict.fromkeys(s.category for s in sentences):
        scored = []
        for res in results:
            rows = rows_for(res, category=c)
            w = wer(rows, judges)
            if not rows or w is None:
                continue
            key = (-(en_recall(rows, judges) or 0), w) if c == "mix" else (w,)
            scored.append((key, res.system))
        if scored:
            best = min(k for k, _ in scored)
            winners[c] = [s for k, s in scored if k == best]
    return winners


def _latency_table(results: list[Result], judges: list[str]) -> list[str]:
    out = ["| System | mean | median | p95 | min | max | audio / sentence |", "|---|---|---|---|---|---|---|"]
    for res in results:
        a = agg(res.rows, judges[0]) if judges else None
        dur = mean_of([r.duration_s for r in res.rows if r.error is None])
        out.append(f"| {res.system} | " + " | ".join(_sec(getattr(a, f, None)) for f in (
            "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_min_s", "latency_max_s"))
            + f" | {_sec(dur)} |")
    return out


def _worst(res: Result, judges: list[str], root: str) -> list[str]:
    rows = sorted(res.rows, key=lambda r: -(r.wer(judges) if r.wer(judges) is not None else 9))[:WORST_LIMIT]
    rows = [r for r in rows if (r.wer(judges) or 0) > 0 or r.error]
    if not rows:
        return ["No errors from any judge."]
    out = [f"| Sentence | WER | Reference | {' | '.join(judges)} | Note | Audio |",
           "|---|---|---|" + "---|" * len(judges) + "---|---|"]
    for r in rows:
        note = r.error or r.duration_flag or ""
        out.append(f"| {r.sentence.id} | {_frac(r.wer(judges))} | {_cell(r.sentence.text)} | "
                   + " | ".join(_cell(r.hyps.get(j) or "") for j in judges)
                   + f" | {_cell(note)} | {f'`{root}/{r.audio}`' if r.audio else '—'} |")
    return out


def render_report(run: Run, sentences: list[Sentence], results: list[Result], run_root: str) -> str:
    judges = run.judges
    meta = run.meta
    langs = [lang for lang in ("vi", "en") if any(s.lang == lang for s in sentences)]
    lines = [
        f"# TTS evaluation: {meta['run_id']}",
        "",
        f"- Dataset: `{meta['dataset']}` `{meta['dataset_hash']}` ({len(sentences)} sentences)",
        f"- Created {meta['created']}, harness `{meta.get('harness_git_sha') or 'unknown'}`",
        f"- ASR judges: {', '.join(f'`{j}`' for j in judges) or 'none'}; normalizer `{NORMALIZER_VERSION}`",
        "- Round-trip WER: each judge transcribes the pass-1 audio with the sentence language as hint; "
        "pooled over sentences, scored against the closest reading (numbers may be digits or words). "
        "A failed synthesis counts as every word deleted. Mean WER is the mean of the judges' WERs.",
        "- UTMOS (UTMOS22 strong) is trained on English speech: compare systems with it, "
        "do not read it as an absolute Vietnamese MOS.",
        f"- Speaker sim: cosine of `{SPEAKER_MODEL}` x-vectors to the reference clip, cloning systems only.",
        f"- Duration flags: audio per character more than {DURATION_FLAG_HIGH:g}x or under "
        f"{DURATION_FLAG_LOW:g}x the median of the other systems for that sentence (runaway or truncated).",
        "- Latency: one HTTP request for the whole clip, median over passes per sentence; "
        "RTF = Σ latency / Σ audio.",
        f"- Google price: Chirp 3 HD US$30 per 1M characters (checked {PRICE_CHECKED}, {PRICE_URL}); "
        "the first 1M characters a month are free.",
        "",
    ]
    for lang in langs:
        n = sum(1 for s in sentences if s.lang == lang)
        title = "Vietnamese (incl. code-switched and numbers)" if lang == "vi" else "English"
        lines += [f"## {title}: {n} sentences", "", *_language_table(results, judges, lang), ""]
    winners = category_winners(results, judges, sentences)
    lines += ["## Mean WER by category", "", *_category_table(results, judges, sentences), "",
              "## Category winners", "", "Lowest mean WER; on `mix` the highest English-word recall first.", "",
              "| Category | Winner |", "|---|---|",
              *[f"| {c} | {', '.join(w)} |" for c, w in winners.items()], "",
              "## Latency", "", *_latency_table(results, judges), ""]
    flagged = [(res.system, r) for res in results for r in res.rows if r.duration_flag]
    lines += ["## Duration flags", ""]
    lines += ([f"- `{s}` {r.sentence.id}: {r.duration_flag} ({_sec(r.duration_s)})" for s, r in flagged]
              or ["None."])
    lines.append("")
    for res in results:
        lines += [f"## Worst sentences: {res.system}", "", *_worst(res, judges, run_root), ""]
    return "\n".join(lines)


def write_report(run_dir: Path, text: str) -> Path:
    path = Path(run_dir) / "report.md"
    path.write_text(text, encoding="utf-8")
    return path

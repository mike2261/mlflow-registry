"""Turn a run directory of raw records into metrics, slices and a Markdown report."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from mlflow_registry.bench.backends import CHIRP_NAME, SYSTEM_LANGS
from mlflow_registry.bench.collect import RUN_META
from mlflow_registry.bench.manifest import Utterance
from mlflow_registry.bench.metrics import Aggregate, Scored, aggregate, median, scored
from mlflow_registry.bench.normalize import NORMALIZER_VERSION

CHIRP3_USD_PER_MIN = 0.016
PRICE_CHECKED = "2026-09-29"
PRICE_URL = "https://cloud.google.com/speech-to-text/pricing"


class MissingBaseline(RuntimeError):
    """No chirp_3.jsonl in the run directory; pass allow_missing_baseline to override."""


class DatasetMismatch(RuntimeError):
    """A records file was collected on different fixtures than the run's ``run.json`` pins."""


@dataclass(frozen=True)
class Cell:
    system: str
    condition: str
    rows: list[Scored]                      # one per utterance: pass-1 text, median latency
    backend: dict = field(default_factory=dict)
    passes: int = 0
    requests: int = 0                       # every request over every pass
    failed_requests: int = 0                # errors or empty text in any pass
    languages: dict[str, str | None] = field(default_factory=dict)   # utt id -> pass-1 detected language


def load_run(run_dir: Path, allow_missing_baseline: bool = False) -> tuple[dict, list[dict]]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / RUN_META).read_text(encoding="utf-8"))
    if not allow_missing_baseline and not (run_dir / f"{CHIRP_NAME}.jsonl").exists():
        raise MissingBaseline(f"{run_dir} has no {CHIRP_NAME}.jsonl; the Google row must be present")
    records: list[dict] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("dataset_hash") != meta["dataset_hash"]:
                raise DatasetMismatch(
                    f"{path.name} was collected on dataset {rec.get('dataset_hash') or 'unknown'}, "
                    f"but {RUN_META} pins {meta['dataset_hash']}; recollect that leg on the same fixtures"
                )
            records.append(rec)
    return meta, records


def build_cells(records: list[dict], utts: list[Utterance]) -> list[Cell]:
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[(r["system"], r["condition"])][r["utt"]].append(r)

    cells: list[Cell] = []
    for (system, condition), per_utt in sorted(grouped.items()):
        rows: list[Scored] = []
        languages: dict[str, str | None] = {}
        for utt in utts:
            passes = sorted(per_utt.get(utt.id, []), key=lambda r: r["pass"])
            if not passes:
                continue
            first = passes[0]
            latencies = [p["latency_s"] for p in passes if p["latency_s"] is not None]
            rows.append(scored(utt, first["text"], first["error"], median(latencies)))
            languages[utt.id] = first.get("language")
        cell_records = [r for recs in per_utt.values() for r in recs]
        cells.append(Cell(
            system, condition, rows,
            backend=dict(cell_records[0].get("backend") or {}),
            passes=max(r["pass"] for r in cell_records),
            requests=len(cell_records),
            failed_requests=sum(1 for r in cell_records if r["error"] is not None),
            languages=languages,
        ))
    return cells


def nondeterministic(records: list[dict]) -> list[tuple[str, str, str, list[str]]]:
    seen: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for r in sorted(records, key=lambda r: (r["system"], r["condition"], r["utt"], r["pass"])):
        key = (r["system"], r["condition"], r["utt"])
        if r["error"] is None and r["text"] not in seen[key]:
            seen[key].append(r["text"])
    return [(s, c, u, v) for (s, c, u), v in sorted(seen.items()) if len(v) > 1]


def in_domain(cell: Cell) -> list[Scored]:
    langs = SYSTEM_LANGS.get(cell.system, frozenset())
    return [r for r in cell.rows if r.utt.lang in langs]


def slices(cell: Cell) -> dict[str, Aggregate]:
    out = {"all": aggregate(cell.rows), "in_domain": aggregate(in_domain(cell))}
    for lang in sorted({r.utt.lang for r in cell.rows}):
        out[f"lang:{lang}"] = aggregate(r for r in cell.rows if r.utt.lang == lang)
    for cat in sorted({r.utt.category for r in cell.rows}):
        out[f"cat:{cat}"] = aggregate(r for r in cell.rows if r.utt.category == cat)
    return out


def cost_per_min(system: str) -> float | None:
    return CHIRP3_USD_PER_MIN if system == CHIRP_NAME else None


# --- report ------------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}% ({num}/{den})" if den else "n/a"


def _sec(x: float | None) -> str:
    return f"{x:.2f}s" if x is not None else "n/a"


def _rate(x: float | None) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _summary_table(cells: list[Cell]) -> list[str]:
    lines = ["| System | In-domain WER | CER | Exact | EN recall | Median latency | p90 | RTF | "
             "Failed utts (pass 1) | $/min | Failed requests (all passes) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        sl = slices(cell)
        d, a = sl["in_domain"], sl["all"]
        cost = cost_per_min(cell.system)
        lines.append(
            f"| {cell.system} | {_pct(d.word_edits, d.ref_words)} | {_pct(d.char_edits, d.ref_chars)} | "
            f"{_pct(d.exact, d.n)} | {_pct(a.en_hits, a.en_total)} | {_sec(a.latency_median_s)} | "
            f"{_sec(a.latency_p90_s)} | {_rate(a.rtf)} | {a.failures}/{a.n} | "
            f"{'$' + format(cost, '.3f') if cost is not None else '—'} | "
            f"{cell.failed_requests}/{cell.requests} |"
        )
    return lines


def _slice_table(cells: list[Cell], prefix: str, title: str) -> list[str]:
    keys = sorted({k for c in cells for k in slices(c) if k.startswith(prefix)})
    if not keys:
        return []
    lines = [f"### WER by {title}", "", "| System | " + " | ".join(k.split(":", 1)[1] for k in keys) + " |",
             "|---|" + "---|" * len(keys)]
    for cell in cells:
        sl = slices(cell)
        vals = [(_pct(sl[k].word_edits, sl[k].ref_words) if k in sl else "n/a") for k in keys]
        lines.append(f"| {cell.system} | " + " | ".join(vals) + " |")
    return lines + [""]


def render_report(meta: dict, utts: list[Utterance], cells: list[Cell],
                  nondet: list[tuple[str, str, str, list[str]]]) -> str:
    words = sum(len(u.text.split()) for u in utts)
    audio = sum(u.duration_s for u in utts)
    caveat = (f"This dataset has {words} reference words. It **cannot rank models**; it proves the "
              "harness and catches gross failures. Every percentage carries its raw counts.")
    out: list[str] = [
        "# STT evaluation",
        "",
        f"- Run: `{meta['run_id']}` created {meta['created']} (harness {meta.get('harness_git_sha') or 'unknown'})",
        f"- Dataset: `{meta['dataset']}` `{meta['dataset_hash']}`: {len(utts)} utterances, "
        f"{words} words, {audio:.1f}s of audio",
        f"- Normalizer: `{NORMALIZER_VERSION}`; text metrics from pass 1, latency = median over passes",
        f"- Chirp 3 price: ${CHIRP3_USD_PER_MIN}/min, list price checked {PRICE_CHECKED} at {PRICE_URL}",
        "",
        f"> {caveat}",
        "",
    ]
    for condition in ("hinted", "auto"):
        group = [c for c in cells if c.condition == condition]
        if not group:
            continue
        out += [f"## Summary: {condition}", "",
                "In-domain = utterances in a language the system claims to support. "
                "Latency, RTF and failures are over all utterances.", ""]
        out += _summary_table(group) + [""]
        out += _slice_table(group, "lang:", "language")
        out += _slice_table(group, "cat:", "category")

    for cell in cells:
        out += [f"## Detail: {cell.system} ({cell.condition})", "",
                "| Utt | Ref | Hyp | Lang | WER | Latency |", "|---|---|---|---|---|---|"]
        for r in cell.rows:
            hyp = f"⚠ {r.error}" if r.failed else (r.hyp or "")
            wer = "n/a" if r.failed else _pct(r.word_edits, r.ref_words)
            lang = cell.languages.get(r.utt.id) or "—"
            out.append(f"| {r.utt.id} | {_cell(r.utt.text)} | {_cell(hyp)} | {lang} | {wer} | "
                       f"{_sec(r.latency_s)} |")
        out.append("")

    out += ["## Non-deterministic outputs", ""]
    if nondet:
        out += ["| System | Condition | Utt | Variants |", "|---|---|---|---|"]
        out += [f"| {s} | {c} | {u} | {_cell(' / '.join(repr(v) for v in variants))} |"
                for s, c, u, variants in nondet]
    else:
        out.append("None: every system produced identical text across passes.")
    out.append("")
    return "\n".join(out)


def write_report(run_dir: Path, text: str) -> Path:
    path = Path(run_dir) / "report.md"
    path.write_text(text, encoding="utf-8")
    return path

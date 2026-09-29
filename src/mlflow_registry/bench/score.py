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


def _frac(x: float | None) -> str:
    return f"{100 * x:.1f}%" if x is not None else "n/a"


def _summary_table(cells: list[Cell]) -> list[str]:
    lines = ["| System | In-domain WER | Mean WER | CER | Exact | EN recall | CS pass | Median latency | p95 | "
             "RTF | Failed utts (pass 1) | $/min | Failed requests (all passes) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in cells:
        sl = slices(cell)
        d, a = sl["in_domain"], sl["all"]
        cost = cost_per_min(cell.system)
        lines.append(
            f"| {cell.system} | {_pct(d.word_edits, d.ref_words)} | {_frac(d.wer_mean)} | "
            f"{_pct(d.char_edits, d.ref_chars)} | {_pct(d.exact, d.n)} | {_pct(a.en_hits, a.en_total)} | "
            f"{_pct(a.cs_pass, a.cs_n)} | {_sec(a.latency_median_s)} | {_sec(a.latency_p95_s)} | "
            f"{_rate(a.rtf)} | {a.failures}/{a.n} | "
            f"{'$' + format(cost, '.3f') if cost is not None else '—'} | "
            f"{cell.failed_requests}/{cell.requests} |"
        )
    return lines


def _latency_table(cells: list[Cell], condition: str) -> list[str]:
    lines = [f"### Latency: {condition}", "",
             "Per utterance, median over passes; statistics as in robo-be's bench_tts.py.", "",
             "| System | mean | median | p95 | min | max |", "|---|---|---|---|---|---|"]
    for cell in cells:
        a = slices(cell)["all"]
        lines.append(f"| {cell.system} | {_sec(a.latency_mean_s)} | {_sec(a.latency_median_s)} | "
                     f"{_sec(a.latency_p95_s)} | {_sec(a.latency_min_s)} | {_sec(a.latency_max_s)} |")
    return lines + [""]


def category_winners(cells: list[Cell]) -> dict[str, list[str]]:
    """robo-be bench_stt's ``_category_winner``, generalised from two systems to N.

    Per category, only systems that claim the category's language compete. If any of them
    passes the code-switch check on every utterance and others do not, only the passers stay
    in ("code-switch dominates"). Then the lowest mean per-utterance WER wins; equal values tie.
    """
    out: dict[str, list[str]] = {}
    categories = sorted({r.utt.category for c in cells for r in c.rows})
    for cat in categories:
        entries = []
        for cell in cells:
            rows = [r for r in in_domain(cell) if r.utt.category == cat]
            if not rows:
                continue
            agg = aggregate(rows)
            if agg.wer_mean is not None:
                entries.append((cell.system, agg.cs_pass_rate, agg.wer_mean))
        if any(cs == 1.0 for _, cs, _ in entries) and any(cs is not None and cs < 1.0 for _, cs, _ in entries):
            entries = [e for e in entries if e[1] == 1.0]
        if not entries:
            out[cat] = []
            continue
        best = min(w for _, _, w in entries)
        out[cat] = sorted(s for s, _, w in entries if w == best)
    return out


def _winner_table(cells: list[Cell], condition: str) -> list[str]:
    lines = [f"### Category winners: {condition}", "",
             "robo-be rule: among systems that support the category's language, code-switch pass "
             "dominates, then lower mean per-utterance WER; equal values tie.", "",
             "| Category | Winner |", "|---|---|"]
    for cat, names in category_winners(cells).items():
        label = " = ".join(names) if names else "—"
        lines.append(f"| {cat} | {label}{' (tie)' if len(names) > 1 else ''} |")
    return lines + [""]


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
                "In-domain = utterances in a language the system claims to support. In-domain WER is "
                "pooled (as robo-be bench_wer.py); Mean WER is the mean of per-utterance WER (as "
                "bench_stt.py). CS pass = every expected English word present, vacuously true for "
                "utterances without English (as bench_stt.py). EN recall, CS pass, latency, RTF and "
                "failures are over all utterances.", ""]
        out += _summary_table(group) + [""]
        out += _latency_table(group, condition)
        out += _winner_table(group, condition)
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


def _pp(a: Aggregate, b: Aggregate) -> str:
    if a.wer is None or b.wer is None:
        return "n/a"
    return f"{100 * (b.wer - a.wer):+.1f} pp"


def render_comparison(runs: list[tuple[str, dict, list[Cell]]]) -> str:
    """Side-by-side in-domain metrics for the same systems on several datasets.

    ``runs`` is ``[(label, run_meta, cells), ...]``; the first is the baseline for the Δ column.
    """
    labels = [label for label, _, _ in runs]
    out = [f"# STT evaluation: {' vs '.join(labels)}", ""]
    out += [f"- `{label}`: dataset `{meta['dataset']}` `{meta['dataset_hash']}`" for label, meta, _ in runs]
    out += ["", f"Δ WER is each dataset's pooled in-domain WER minus `{labels[0]}`'s.", ""]
    for condition in ("hinted", "auto"):
        per_run = [{c.system: c for c in cells if c.condition == condition} for _, _, cells in runs]
        systems = sorted(set().union(*per_run))
        if not systems:
            continue
        head = ["System"]
        for label in labels:
            head += [f"{label} WER", f"{label} Mean WER", f"{label} CS pass", f"{label} median latency"]
        head += [f"Δ WER ({label})" for label in labels[1:]]
        out += [f"## {condition}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for system in systems:
            aggs = [slices(r[system]) if system in r else None for r in per_run]
            vals = [system]
            for sl in aggs:
                if sl is None:
                    vals += ["n/a"] * 4
                    continue
                d, a = sl["in_domain"], sl["all"]
                vals += [_pct(d.word_edits, d.ref_words), _frac(d.wer_mean), _pct(a.cs_pass, a.cs_n),
                         _sec(a.latency_median_s)]
            for sl in aggs[1:]:
                vals.append(_pp(aggs[0]["in_domain"], sl["in_domain"]) if aggs[0] and sl else "n/a")
            out.append("| " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out)


def write_report(run_dir: Path, text: str) -> Path:
    path = Path(run_dir) / "report.md"
    path.write_text(text, encoding="utf-8")
    return path

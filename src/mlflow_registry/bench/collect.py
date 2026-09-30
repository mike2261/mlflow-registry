"""Call one backend over the manifest and write raw records to ``<run_dir>/<system>.jsonl``.

A run directory is shared by every leg of one evaluation (Google leg on the laptop,
self-hosted leg on the devserver). ``run.json`` pins the dataset hash so two legs can never
use different fixtures. Records are raw: no normalization, no metrics; ``score`` does that.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from mlflow_registry.bench import manifest
from mlflow_registry.bench.backends import Backend, BackendError, Hypothesis

RUN_META = "run.json"
CONDITIONS = ("hinted", "auto")


class RunMismatch(RuntimeError):
    """The run directory was created from different fixtures than the ones on disk."""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def ensure_run(run_dir: Path, fixtures_dir: Path, run_id: str | None = None,
               git_sha: str | None = None, hasher: Callable[[Path], str] = manifest.dataset_hash) -> dict:
    run_dir = Path(run_dir)
    meta_path = run_dir / RUN_META
    current_hash = hasher(fixtures_dir)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["dataset_hash"] != current_hash:
            raise RunMismatch(
                f"{meta_path} was created from dataset {meta['dataset_hash']} "
                f"but {fixtures_dir} hashes to {current_hash}"
            )
        return meta
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id or run_dir.name,
        "dataset": str(Path(fixtures_dir).name),
        "dataset_hash": current_hash,
        "created": _now(),
        "harness_git_sha": git_sha if git_sha is not None else _git_sha(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def record(system: str, condition: str, pass_no: int, utt: manifest.Utterance,
           lang_hint: str | None, hyp: Hypothesis | None, error: str | None,
           ts: str, backend_meta: dict, dataset_hash: str) -> dict:
    return {
        "dataset_hash": dataset_hash,
        "system": system,
        "condition": condition,
        "pass": pass_no,
        "utt": utt.id,
        "lang_hint": lang_hint,
        "text": None if hyp is None else hyp.text,
        "language": None if hyp is None else hyp.language,
        "latency_s": None if hyp is None else hyp.latency_s,
        "error": error,
        "ts": ts,
        "backend": backend_meta,
    }


def _one(backend: Backend, wav: bytes, lang: str | None) -> tuple[Hypothesis | None, str | None]:
    try:
        hyp = backend.transcribe(wav, lang)
    except BackendError as e:
        return None, f"{type(e).__name__}: {e}"
    if not hyp.text.strip():
        return hyp, "empty transcript"
    return hyp, None


def collect(backend: Backend, fixtures_dir: Path, run_dir: Path, passes: int = 3, concurrency: int = 1,
            retry_errors: bool = False, now: Callable[[], str] = _now,
            log: Callable[..., None] = print) -> Path:
    fixtures_dir, run_dir = Path(fixtures_dir), Path(run_dir)
    # Every record carries the hash of the fixtures it was collected on, so score can refuse a
    # leg collected elsewhere on different fixtures even when that machine made its own run.json.
    dataset_hash = ensure_run(run_dir, fixtures_dir)["dataset_hash"]
    utts = manifest.load(fixtures_dir)
    conditions = [c for c in CONDITIONS if c == "hinted" or backend.auto_detect]
    meta = backend.meta()
    out = run_dir / f"{backend.name}.jsonl"

    def call(utt: manifest.Utterance, hint: str | None) -> tuple[Hypothesis | None, str | None, str]:
        # audio is read per request so a large dataset is never held in memory at once
        hyp, error = _one(backend, manifest.audio_bytes(fixtures_dir, utt), hint)
        return hyp, error, now()

    if retry_errors:
        return _retry_errors(out, {u.id: u for u in utts}, call, backend.name, meta, dataset_hash,
                             concurrency, log)

    log(f"[{backend.name}] warm-up")
    call(utts[0], utts[0].lang)

    # concurrency > 1 overlaps requests (useful for a cloud API); records are still written in
    # manifest order. Keep it at 1 for self-hosted models so latency is not measured under load.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool, out.open("w", encoding="utf-8") as fh:
        for condition in conditions:
            for pass_no in range(1, passes + 1):
                hints = [utt.lang if condition == "hinted" else None for utt in utts]
                results = pool.map(call, utts, hints)
                for i, (utt, hint, (hyp, error, ts)) in enumerate(zip(utts, hints, results), 1):
                    row = record(backend.name, condition, pass_no, utt, hint, hyp, error, ts, meta,
                                 dataset_hash)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    status = error or f"{hyp.latency_s:.2f}s {hyp.text[:60]!r}"
                    log(f"[{backend.name}] {condition} p{pass_no} {i}/{len(utts)} {utt.id}: {status}")
                fh.flush()
    return out


def _retry_errors(out: Path, by_id: dict[str, manifest.Utterance], call, system: str, meta: dict,
                  dataset_hash: str, concurrency: int, log: Callable[..., None]) -> Path:
    """Re-request only the records of ``out`` that have an error; keep every other record as is."""
    if not out.exists():
        raise FileNotFoundError(f"{out} does not exist; run collect without --retry-errors first")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [i for i, r in enumerate(rows) if r["error"] is not None]
    log(f"[{system}] retrying {len(todo)} of {len(rows)} records with errors")
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = pool.map(lambda i: call(by_id[rows[i]["utt"]], rows[i]["lang_hint"]), todo)
        for n, (i, (hyp, error, ts)) in enumerate(zip(todo, results), 1):
            old = rows[i]
            rows[i] = record(system, old["condition"], old["pass"], by_id[old["utt"]], old["lang_hint"],
                             hyp, error, ts, meta, dataset_hash)
            log(f"[{system}] retry {n}/{len(todo)} {old['utt']}: {error or 'ok'}")
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(out)
    return out

"""Synthesize every sentence a system supports; write ``<system>.jsonl`` and pass-1 WAVs.

Layout of a TTS run directory (shared by the Google leg and the self-hosted leg, like STT)::

    run.json                        dataset hash, created, harness sha
    <system>.jsonl                  one record per sentence per pass
    audio/<system>/<sentence>.wav   pass-1 audio, the only audio that is judged
    asr/<judge>/<system>.jsonl      written by judge asr
    quality/<system>.jsonl          written by judge quality

Pass 1 is the judged draw. Passes 2..N are timed only (their audio is discarded) so the
median latency is not a single sample.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from mlflow_registry.bench.backends import BackendError
from mlflow_registry.bench.collect import _now, ensure_run
from mlflow_registry.bench.tts import manifest
from mlflow_registry.bench.tts.backends import Synthesis, TtsBackend

MIN_AUDIO_S = 0.2


def audio_path(run_dir: Path, system: str, sentence_id: str) -> Path:
    return Path(run_dir) / "audio" / system / f"{sentence_id}.wav"


def _one(backend: TtsBackend, s: manifest.Sentence) -> tuple[Synthesis | None, str | None]:
    try:
        syn = backend.synthesize(s.text, s.lang)
    except BackendError as e:
        return None, f"{type(e).__name__}: {e}"
    if syn.duration_s < MIN_AUDIO_S:
        return syn, f"audio shorter than {MIN_AUDIO_S}s ({syn.duration_s:.2f}s)"
    return syn, None


def record(system: str, pass_no: int, s: manifest.Sentence, syn: Synthesis | None, error: str | None,
           audio: str | None, ts: str, backend_meta: dict, dataset_hash: str) -> dict:
    return {
        "dataset_hash": dataset_hash,
        "system": system,
        "pass": pass_no,
        "sent": s.id,
        "lang": s.lang,
        "latency_s": None if syn is None else syn.latency_s,
        "sample_rate": None if syn is None else syn.sample_rate,
        "duration_s": None if syn is None else syn.duration_s,
        "audio": audio,
        "error": error,
        "ts": ts,
        "backend": backend_meta,
    }


def collect(backend: TtsBackend, sentences_dir: Path, run_dir: Path, passes: int = 3,
            now: Callable[[], str] = _now, log: Callable[..., None] = print) -> Path:
    sentences_dir, run_dir = Path(sentences_dir), Path(run_dir)
    dataset_hash = ensure_run(run_dir, sentences_dir, hasher=manifest.dataset_hash)["dataset_hash"]
    todo = [s for s in manifest.load(sentences_dir) if s.lang in backend.languages]
    if not todo:
        raise ValueError(f"{backend.name} supports none of the sentence languages")
    meta = backend.meta()
    out = run_dir / f"{backend.name}.jsonl"
    audio_dir = run_dir / "audio" / backend.name
    audio_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{backend.name}] warm-up")
    _one(backend, todo[0])

    with out.open("w", encoding="utf-8") as fh:
        for pass_no in range(1, passes + 1):
            for i, s in enumerate(todo, 1):
                syn, error = _one(backend, s)
                audio = None
                if pass_no == 1 and syn is not None:
                    path = audio_path(run_dir, backend.name, s.id)
                    path.write_bytes(syn.wav)
                    audio = str(path.relative_to(run_dir))
                row = record(backend.name, pass_no, s, syn, error, audio, now(), meta, dataset_hash)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = error or f"{syn.latency_s:.2f}s -> {syn.duration_s:.2f}s audio"
                log(f"[{backend.name}] p{pass_no} {i}/{len(todo)} {s.id}: {status}")
            fh.flush()
    return out

"""Judge pass-1 audio: ASR round trip (intelligibility) and quality models (naturalness, voice).

``asr`` sends every WAV to an STT backend from ``bench.backends`` (a serving container or
Chirp 3) with the sentence's language as the hint and writes ``asr/<judge>/<system>.jsonl``.
``quality`` runs a UTMOS predictor and a speaker-embedding model over the same WAVs and writes
``quality/<system>.jsonl``. Each judge writes its own files, so judges can run on different
machines and the run directory is rsynced together afterwards, like the STT legs.
"""
from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mlflow_registry.bench.backends import Backend, BackendError
from mlflow_registry.bench.collect import RUN_META, _now
from mlflow_registry.bench.tts import manifest

JUDGE_SR = 16_000


def pass1_records(run_dir: Path, system: str) -> list[dict]:
    path = Path(run_dir) / f"{system}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["pass"] == 1]


def systems_in(run_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(run_dir).glob("*.jsonl"))


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return path


def _dataset_hash(run_dir: Path) -> str:
    return json.loads((Path(run_dir) / RUN_META).read_text(encoding="utf-8"))["dataset_hash"]


# --- ASR round trip ----------------------------------------------------------------------------

def asr(stt: Backend, run_dir: Path, systems: list[str] | None = None,
        log: Callable[..., None] = print) -> list[Path]:
    run_dir = Path(run_dir)
    dataset_hash = _dataset_hash(run_dir)
    written = []
    for system in systems or systems_in(run_dir):
        rows = []
        todo = pass1_records(run_dir, system)
        for i, rec in enumerate(todo, 1):
            row = {"dataset_hash": dataset_hash, "system": system, "judge": stt.name, "sent": rec["sent"],
                   "text": None, "language": None, "latency_s": None, "error": None, "ts": _now()}
            if rec["audio"] is None or rec["error"] is not None:
                row["error"] = "no audio"
            else:
                try:
                    hyp = stt.transcribe((run_dir / rec["audio"]).read_bytes(), rec["lang"])
                    row.update(text=hyp.text, language=hyp.language, latency_s=hyp.latency_s)
                except BackendError as e:
                    row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            log(f"[{stt.name} <- {system}] {i}/{len(todo)} {rec['sent']}: {row['error'] or repr(row['text'][:60])}")
        written.append(_write(run_dir / "asr" / stt.name / f"{system}.jsonl", rows))
    return written


# --- quality -----------------------------------------------------------------------------------

Mos = Callable[[np.ndarray], float]                  # 16 kHz mono float32 -> predicted MOS
Embed = Callable[[np.ndarray], np.ndarray]           # 16 kHz mono float32 -> speaker embedding


def load_16k(wav: bytes) -> np.ndarray:
    samples, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sr != JUDGE_SR:
        g = math.gcd(sr, JUDGE_SR)
        mono = resample_poly(mono, JUDGE_SR // g, sr // g)
    return np.ascontiguousarray(mono, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def quality(run_dir: Path, sentences_dir: Path, mos: Mos, embed: Embed, cloning: frozenset[str],
            systems: list[str] | None = None, log: Callable[..., None] = print) -> list[Path]:
    """UTMOS for every system; speaker similarity to the reference only for ``cloning`` systems."""
    run_dir = Path(run_dir)
    dataset_hash = _dataset_hash(run_dir)
    ref_emb = embed(load_16k(manifest.reference(sentences_dir).wav))
    written = []
    for system in systems or systems_in(run_dir):
        rows = []
        for rec in pass1_records(run_dir, system):
            row = {"dataset_hash": dataset_hash, "system": system, "sent": rec["sent"],
                   "utmos": None, "spk_sim": None}
            if rec["audio"] is not None and rec["error"] is None:
                x = load_16k((run_dir / rec["audio"]).read_bytes())
                row["utmos"] = float(mos(x))
                if system in cloning:
                    row["spk_sim"] = cosine(embed(x), ref_emb)
            rows.append(row)
        log(f"[quality] {system}: {len(rows)} sentences")
        written.append(_write(run_dir / "quality" / f"{system}.jsonl", rows))
    return written


def load_utmos(device: str = "cpu") -> Mos:
    """UTMOS22 strong learner via SpeechMOS (torch.hub). English-trained: relative signal only."""
    import torch

    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)

    def mos(x: np.ndarray) -> float:
        with torch.inference_mode():
            return float(model(torch.from_numpy(x).unsqueeze(0).to(device), JUDGE_SR).item())

    return mos


SPEAKER_MODEL = "microsoft/wavlm-base-plus-sv"


def load_speaker(device: str = "cpu") -> Embed:
    """WavLM-base-plus x-vector head (transformers), the usual speaker-verification embedding."""
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    extractor = AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL)
    model = WavLMForXVector.from_pretrained(SPEAKER_MODEL).to(device).eval()

    def embed(x: np.ndarray) -> np.ndarray:
        inputs = extractor(x, sampling_rate=JUDGE_SR, return_tensors="pt").to(device)
        with torch.inference_mode():
            return model(**inputs).embeddings[0].cpu().numpy()

    return embed

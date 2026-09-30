"""Time to first audio (TTFB) per TTS system, measured where the model runs.

The REST containers return whole clips, so ``collect`` can only time the complete clip. TTFB
needs the model's own streaming API, so this module runs *inside* each model's serving image
(``scripts/tts_ttfb.py local``: the wrapper loads the weights from the shared cache, then the
runtime's streaming call is timed) and on the laptop for Google (``StreamingSynthesize``,
network included). Systems without a streaming API are timed the same way on one
whole-clip call and marked ``streaming: false``: their TTFB is their full latency.

Every request re-encodes the reference clip, as the REST path does; a server that caches the
voice prompt would start a little sooner.

Imports stay within what the serving images ship (numpy, soundfile, the serving package):
no jiwer, no Google libraries unless the Google streamer is used.
"""
from __future__ import annotations

import io
import json
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from mlflow_registry.bench.tts import manifest
from mlflow_registry.bench.tts.manifest import Reference, Sentence

TTFB_DIR = "ttfb"
RUN_META = "run.json"

Stream = Callable[[Sentence], Iterable[np.ndarray]]      # float32 mono chunks at ``sample_rate``


@dataclass
class Streamer:
    name: str
    languages: frozenset[str]
    sample_rate: int
    streaming: bool
    stream: Stream
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Timing:
    ttfb_s: float
    total_s: float
    audio_s: float
    chunks: int


def time_stream(chunks: Iterable[np.ndarray | bytes], sample_rate: int,
                clock: Callable[[], float] = time.perf_counter) -> Timing:
    """Consume a chunk iterator; the clock starts before the first ``next()``.

    Chunks are float sample arrays, or ``bytes`` of an encoded (MP3) stream: then the first bytes
    count as first audio, as a streaming decoder can play the first frame, and the audio length
    is decoded once the stream ends.
    """
    t0 = clock()
    first = None
    samples = n = 0
    encoded = bytearray()
    for chunk in chunks:
        if chunk is None or len(chunk) == 0:
            continue
        if first is None:
            first = clock() - t0
        if isinstance(chunk, (bytes, bytearray)):
            encoded += chunk
        else:
            samples += len(chunk)
        n += 1
    total = clock() - t0
    if first is None:
        raise RuntimeError("stream produced no audio")
    if encoded:
        info = sf.info(io.BytesIO(bytes(encoded)))
        return Timing(first, total, info.frames / info.samplerate, n)
    return Timing(first, total, samples / sample_rate, n)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(streamer: Streamer, sentences_dir: Path, run_dir: Path, passes: int = 3,
        clock: Callable[[], float] = time.perf_counter, now: Callable[[], str] = _now,
        log: Callable[..., None] = print) -> Path:
    """Time every supported sentence ``passes`` times; write ``ttfb/<system>.jsonl``."""
    sentences_dir, run_dir = Path(sentences_dir), Path(run_dir)
    meta = json.loads((run_dir / RUN_META).read_text(encoding="utf-8"))
    dataset_hash = manifest.dataset_hash(sentences_dir)
    if meta["dataset_hash"] != dataset_hash:
        raise RuntimeError(f"{run_dir} pins {meta['dataset_hash']}, the sentences on disk hash to {dataset_hash}")
    todo = [s for s in manifest.load(sentences_dir) if s.lang in streamer.languages]
    log(f"[{streamer.name}] warm-up")
    time_stream(streamer.stream(todo[0]), streamer.sample_rate, clock)
    out = run_dir / TTFB_DIR / f"{streamer.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for pass_no in range(1, passes + 1):
            for i, s in enumerate(todo, 1):
                row = {"dataset_hash": dataset_hash, "system": streamer.name, "pass": pass_no, "sent": s.id,
                       "lang": s.lang, "streaming": streamer.streaming, "ttfb_s": None, "total_s": None,
                       "audio_s": None, "chunks": None, "error": None, "ts": now(), "backend": streamer.meta}
                try:
                    t = time_stream(streamer.stream(s), streamer.sample_rate, clock)
                    row.update(ttfb_s=t.ttfb_s, total_s=t.total_s, audio_s=t.audio_s, chunks=t.chunks)
                    status = f"first audio {t.ttfb_s:.3f}s, done {t.total_s:.2f}s, {t.chunks} chunks"
                except Exception as e:  # one bad sentence must not end the run
                    row["error"] = f"{type(e).__name__}: {e}"
                    status = row["error"]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                log(f"[{streamer.name}] p{pass_no} {i}/{len(todo)} {s.id}: {status}")
            fh.flush()
    return out


# --- self-hosted: inside the model's serving image ---------------------------------------------

def local_streamer(name: str, weights_version: str, reference: Reference, languages: frozenset[str],
                   voicing: str) -> Streamer:
    """Load ``name`` with its serving wrapper and expose its streaming call (or a whole-clip call).

    Weights come from the serving cache (``MODEL_CACHE``, ``/models`` in the images), downloaded
    by the serving container on its first start.
    """
    from mlflow_registry.serving.base import TtsRequest
    from mlflow_registry.serving.catalog import SERVING
    from mlflow_registry.serving.weights import ensure_weights

    model = SERVING[name].make_wrapper(weights_version)
    model._load(ensure_weights(name, weights_version))
    ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    ref_file.write(reference.wav)
    ref_file.close()
    ref_path = ref_file.name

    if name == "voxcpm2":
        sr = int(model._sample_rate)
        kwargs = {"cfg_value": 2.0, "inference_timesteps": 10, "reference_wav_path": ref_path,
                  "prompt_wav_path": ref_path, "prompt_text": reference.text}
        return Streamer(name, languages, sr, True,
                        lambda s: model._model.generate_streaming(text=s.text, **kwargs),
                        {"kind": "local", "api": "VoxCPM.generate_streaming"})
    if name == "vieneu-tts-v3-turbo":
        return Streamer(name, languages, int(model._sample_rate), True,
                        lambda s: model._tts.infer_stream(s.text, ref_audio=ref_path),
                        {"kind": "local", "api": "Vieneu.infer_stream"})

    ref_audio = sf.read(io.BytesIO(reference.wav), dtype="float32")
    clone = voicing.startswith("clone")

    def request(text: str, lang: str) -> TtsRequest:
        return TtsRequest(text=text, language=lang,
                          voice=None if clone else voicing,
                          ref_audio=ref_audio if clone else None,
                          ref_text=reference.text if voicing == "clone+text" else None)

    def whole(s: Sentence) -> Iterable[np.ndarray]:
        samples, _ = model._synthesize(request(s.text, s.lang))
        yield samples

    # These wrappers only learn the output rate from a synthesis call.
    probe_lang = "en" if "en" in languages else sorted(languages)[0]
    _, sr = model._synthesize(request("Hello." if probe_lang == "en" else "Xin chào.", probe_lang))
    return Streamer(name, languages, int(sr), False, whole, {"kind": "local", "api": "whole clip (no streaming API)"})


# --- Google Cloud TTS StreamingSynthesize (laptop) ---------------------------------------------

GOOGLE_STREAM_SR = 24_000


def google_streamer(voice: str, languages: frozenset[str], lang_codes: dict[str, str], name: str,
                    client=None) -> Streamer:
    from google.cloud import texttospeech as tts

    client = client or tts.TextToSpeechClient()

    def stream(s: Sentence) -> Iterable[np.ndarray]:
        code = lang_codes[s.lang]
        config = tts.StreamingSynthesizeConfig(
            voice=tts.VoiceSelectionParams(language_code=code, name=f"{code}-Chirp3-HD-{voice}"))

        def requests():
            yield tts.StreamingSynthesizeRequest(streaming_config=config)
            yield tts.StreamingSynthesizeRequest(input=tts.StreamingSynthesisInput(text=s.text))

        for response in client.streaming_synthesize(requests()):
            pcm = np.frombuffer(response.audio_content, dtype="<i2")
            yield pcm.astype(np.float32) / 32768.0

    return Streamer(name, languages, GOOGLE_STREAM_SR, True, stream,
                    {"kind": "google", "api": "StreamingSynthesize", "voice": voice})


# --- ElevenLabs /stream (laptop) ---------------------------------------------------------------

def elevenlabs_streamer(backend) -> Streamer:
    """``backend`` is a ``bench.tts.backends.ElevenLabsBackend``; chunks are MP3 bytes."""
    return Streamer(backend.name, backend.languages, 0, True, lambda s: backend.stream_mp3(s.text),
                    {**backend.meta(), "api": "text-to-speech/stream"})

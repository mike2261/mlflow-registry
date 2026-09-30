"""TTS systems under test behind one interface: ``synthesize(text, language) -> Synthesis``.

``TtsServingBackend`` talks to a TTS serving container over the REST contract; every cloning
model gets the same reference clip (and its transcript where the model takes one), the preset
models get their preset voice. ``GoogleTtsBackend`` calls Cloud Text-to-Speech with a Chirp 3 HD
voice. ``ElevenLabsBackend`` calls the ElevenLabs HTTP API with robo-be's model and voice. All
take an injectable transport so tests never touch a network; latency is measured around the
transport call only.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import soundfile as sf

from mlflow_registry.bench.backends import BackendError, Post, _is_quota_error, _urllib_post
from mlflow_registry.bench.tts.manifest import Reference
from mlflow_registry.serving.catalog import SERVING

GOOGLE_NAME = "chirp3-hd"
GOOGLE_VOICE = "Aoede"
GOOGLE_LANG_CODES = {"vi": "vi-VN", "en": "en-US"}
GOOGLE_SAMPLE_RATE = 24_000

# Same model and voice as robo-be's ElevenLabs provider (robo_infer/config.py), one voice for both
# languages. MP3 works on every plan (robo-be's default output format too); PCM needs a paid tier.
ELEVENLABS_NAME = "elevenlabs-v3"
ELEVENLABS_MODEL = "eleven_v3"
ELEVENLABS_VOICE = "0ggMuQ1r9f9jqBu50nJn"
ELEVENLABS_FORMAT = "mp3_44100_128"
ELEVENLABS_KEY_ENV = "ELEVENLABS_API_KEY"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}{stream}?output_format={fmt}"

TTS_MODELS: tuple[str, ...] = tuple(s.name for s in SERVING.values() if s.task == "tts")

SYSTEM_LANGS: dict[str, frozenset[str]] = {
    "voxcpm2": frozenset({"vi", "en"}),
    "vieneu-tts-v3-turbo": frozenset({"vi"}),
    "kokoro-82m": frozenset({"en"}),
    "qwen3-tts-1.7b-base": frozenset({"en"}),   # no Vietnamese in its language list
    GOOGLE_NAME: frozenset({"vi", "en"}),
    ELEVENLABS_NAME: frozenset({"vi", "en"}),
}

# How each self-hosted model is voiced. "clone+text": reference clip and its transcript;
# "clone": reference clip only (the model takes no transcript); anything else is a preset name.
VOICING: dict[str, str] = {
    "voxcpm2": "clone+text",
    "vieneu-tts-v3-turbo": "clone",
    "qwen3-tts-1.7b-base": "clone+text",
    "kokoro-82m": "af_heart",
}
CLONING: frozenset[str] = frozenset(n for n, v in VOICING.items() if v.startswith("clone"))


@dataclass(frozen=True)
class Synthesis:
    wav: bytes              # a complete WAV file
    sample_rate: int
    duration_s: float
    latency_s: float


class TtsBackend(Protocol):
    name: str
    languages: frozenset[str]

    def meta(self) -> dict[str, object]: ...
    def synthesize(self, text: str, language: str) -> Synthesis: ...


def wav_info(wav: bytes) -> tuple[int, float]:
    """(sample_rate, duration_s) of a WAV file; raises BackendError if unreadable."""
    try:
        info = sf.info(io.BytesIO(wav))
    except Exception as e:
        raise BackendError(f"audio is not a readable WAV: {e}") from e
    return int(info.samplerate), info.frames / info.samplerate if info.samplerate else 0.0


class TtsServingBackend:
    def __init__(
        self,
        name: str,
        reference: Reference,
        host: str = "localhost",
        port: int | None = None,
        post: Post | None = None,
        timeout: float = 60.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if name not in TTS_MODELS:
            raise KeyError(f"{name!r} is not a servable TTS model; known: {', '.join(TTS_MODELS)}")
        self.name = name
        self.languages = SYSTEM_LANGS[name]
        self.voicing = VOICING[name]
        self.host = host
        self.port = port if port is not None else SERVING[name].port
        self._ref_b64 = base64.b64encode(reference.wav).decode()
        self._ref_text = reference.text
        self._post = post or _urllib_post
        self._timeout = timeout
        self._clock = clock

    def meta(self) -> dict[str, object]:
        return {"kind": "serving", "host": self.host, "port": self.port, "voice": self.voicing}

    def request(self, text: str, language: str) -> dict[str, str]:
        record = {"text": text, "language": language}
        if self.voicing.startswith("clone"):
            record["ref_audio_b64"] = self._ref_b64
            if self.voicing == "clone+text":
                record["ref_text"] = self._ref_text
        else:
            record["voice"] = self.voicing
        return record

    def synthesize(self, text: str, language: str) -> Synthesis:
        url = f"http://{self.host}:{self.port}/invocations"
        t0 = self._clock()
        try:
            out = self._post(url, {"dataframe_records": [self.request(text, language)]}, self._timeout)
            latency = self._clock() - t0
            wav = base64.b64decode(out["predictions"][0]["audio_b64"])
        except BackendError:
            raise
        except Exception as e:  # connection refused, timeout, bad JSON, unexpected payload
            raise BackendError(f"{type(e).__name__}: {e}") from e
        sample_rate, duration = wav_info(wav)
        return Synthesis(wav=wav, sample_rate=sample_rate, duration_s=duration, latency_s=latency)


# --- Google Cloud Text-to-Speech ------------------------------------------------------------

Speak = Callable[[str, str, str], bytes]   # (text, language_code, voice_name) -> WAV bytes


def google_speaker(client=None) -> Speak:
    """Return a ``speak(text, language_code, voice_name) -> wav`` closure (LINEAR16 is a WAV)."""
    from google.cloud import texttospeech

    client = client or texttospeech.TextToSpeechClient()

    def speak(text: str, language_code: str, voice_name: str) -> bytes:
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                                                  sample_rate_hertz=GOOGLE_SAMPLE_RATE),
        )
        return response.audio_content

    return speak


class GoogleTtsBackend:
    def __init__(
        self,
        voice: str = GOOGLE_VOICE,
        speak: Speak | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        backoff_s: float = 5.0,
    ) -> None:
        self.name = GOOGLE_NAME
        self.languages = SYSTEM_LANGS[GOOGLE_NAME]
        self.voice = voice
        self._speak = speak or google_speaker()
        self._clock = clock
        self._sleep = sleep
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def voice_name(self, language: str) -> str:
        return f"{GOOGLE_LANG_CODES[language]}-Chirp3-HD-{self.voice}"

    def meta(self) -> dict[str, object]:
        return {"kind": "google", "voice": self.voice, "model": "Chirp3-HD"}

    def synthesize(self, text: str, language: str) -> Synthesis:
        code = GOOGLE_LANG_CODES[language]
        # Same quota handling as the Chirp STT backend: retry 429 with backoff, time only the
        # successful call.
        for attempt in range(self._max_retries + 1):
            t0 = self._clock()
            try:
                wav = self._speak(text, code, self.voice_name(language))
                break
            except Exception as e:
                if _is_quota_error(e) and attempt < self._max_retries:
                    self._sleep(self._backoff_s * 2 ** attempt)
                    continue
                raise BackendError(f"{type(e).__name__}: {e}") from e
        latency = self._clock() - t0
        sample_rate, duration = wav_info(wav)
        return Synthesis(wav=wav, sample_rate=sample_rate, duration_s=duration, latency_s=latency)


# --- ElevenLabs --------------------------------------------------------------------------------

Fetch = Callable[[str, dict, bytes], Iterator[bytes]]   # (url, headers, body) -> response body chunks


def _urllib_fetch(url: str, headers: dict, body: bytes, timeout: float = 120.0) -> Iterator[bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while chunk := resp.read1(4096):
                yield chunk
    except urllib.error.HTTPError as e:
        raise BackendError(f"HTTP {e.code} from ElevenLabs: {e.read().decode(errors='replace')[:300]}") from e


def mp3_to_wav(mp3: bytes) -> tuple[bytes, int, float]:
    """Decode an MP3 into a 16-bit WAV; returns (wav, sample_rate, duration_s)."""
    try:
        samples, sr = sf.read(io.BytesIO(mp3), dtype="float32")
    except Exception as e:
        raise BackendError(f"ElevenLabs audio is not a readable MP3: {e}") from e
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), int(sr), len(samples) / sr


class ElevenLabsBackend:
    def __init__(
        self,
        api_key: str | None = None,
        voice: str = ELEVENLABS_VOICE,
        model: str = ELEVENLABS_MODEL,
        fetch: Fetch | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        backoff_s: float = 5.0,
    ) -> None:
        self.name = ELEVENLABS_NAME
        self.languages = SYSTEM_LANGS[ELEVENLABS_NAME]
        self.voice, self.model = voice, model
        self._key = api_key or os.environ.get(ELEVENLABS_KEY_ENV, "")
        if not self._key:
            raise BackendError(f"set {ELEVENLABS_KEY_ENV} to call ElevenLabs")
        self._fetch = fetch or _urllib_fetch
        self._clock, self._sleep = clock, sleep
        self._max_retries, self._backoff_s = max_retries, backoff_s

    def meta(self) -> dict[str, object]:
        return {"kind": "elevenlabs", "model": self.model, "voice": self.voice, "format": ELEVENLABS_FORMAT}

    def request(self, text: str, stream: bool) -> tuple[str, dict, bytes]:
        url = ELEVENLABS_URL.format(voice=self.voice, stream="/stream" if stream else "", fmt=ELEVENLABS_FORMAT)
        headers = {"xi-api-key": self._key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
        return url, headers, json.dumps({"text": text, "model_id": self.model}).encode()

    def stream_mp3(self, text: str) -> Iterator[bytes]:
        """Streaming endpoint; the first MP3 bytes are the first playable audio."""
        return self._fetch(*self.request(text, stream=True))

    def synthesize(self, text: str, language: str) -> Synthesis:  # noqa: ARG002 - v3 detects the language
        # 429 (rate or concurrency limit) is retried with backoff like the Google backends.
        for attempt in range(self._max_retries + 1):
            t0 = self._clock()
            try:
                mp3 = b"".join(self._fetch(*self.request(text, stream=False)))
                break
            except BackendError as e:
                if "HTTP 429" in str(e) and attempt < self._max_retries:
                    self._sleep(self._backoff_s * 2 ** attempt)
                    continue
                raise
            except Exception as e:
                raise BackendError(f"{type(e).__name__}: {e}") from e
        latency = self._clock() - t0
        wav, sr, duration = mp3_to_wav(mp3)
        return Synthesis(wav=wav, sample_rate=sr, duration_s=duration, latency_s=latency)

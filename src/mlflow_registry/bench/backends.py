"""Systems under test behind one interface: ``transcribe(wav_bytes, language) -> Hypothesis``.

``ServingBackend`` talks to a serving container from ``docker-compose.serving.yaml`` over the
REST contract (``POST /invocations`` with ``dataframe_records``). ``ChirpBackend`` calls Google
Speech-to-Text v2 directly. Both take an injectable transport so tests never touch a network.
Latency is measured around the transport call only.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from mlflow_registry.serving.catalog import SERVING

CHIRP_NAME = "chirp_3"
CHIRP_LANG_CODES = {"vi": "vi-VN", "en": "en-US"}
# "auto" condition for Chirp: restricted detection between the two languages the product hears,
# not open-world detection over every supported language (which misreads 1 s Vietnamese clips
# as Korean or Chinese).
CHIRP_AUTO_CODES: tuple[str, ...] = ("vi-VN", "en-US")
_CHIRP_LANG_BACK = {v.lower(): k for k, v in CHIRP_LANG_CODES.items()}

STT_MODELS: tuple[str, ...] = tuple(s.name for s in SERVING.values() if s.task == "stt")

SYSTEM_LANGS: dict[str, frozenset[str]] = {
    "qwen3-asr-1.7b": frozenset({"vi", "en"}),
    "granite-speech-4.1-2b": frozenset({"en"}),
    "gipformer1.5-68m-rnnt": frozenset({"vi"}),
    "parakeet-ctc-0.6b-vietnamese": frozenset({"vi"}),
    "whisper-large-v3": frozenset({"vi", "en"}),
    CHIRP_NAME: frozenset({"vi", "en"}),
}
AUTO_DETECT: frozenset[str] = frozenset({"qwen3-asr-1.7b", "whisper-large-v3", CHIRP_NAME})


class BackendError(RuntimeError):
    """Transport or API failure; collect records it and moves on."""


@dataclass(frozen=True)
class Hypothesis:
    text: str
    language: str | None
    latency_s: float


class Backend(Protocol):
    name: str
    languages: frozenset[str]
    auto_detect: bool

    def meta(self) -> dict[str, object]: ...
    def transcribe(self, wav: bytes, language: str | None) -> Hypothesis: ...


# --- serving containers -----------------------------------------------------------------

Post = Callable[[str, dict, float], dict]


def _urllib_post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise BackendError(f"HTTP {e.code} from {url}: {detail}") from e


class ServingBackend:
    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int | None = None,
        post: Post | None = None,
        timeout: float = 600.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if name not in STT_MODELS:
            raise KeyError(f"{name!r} is not a servable STT model; known: {', '.join(STT_MODELS)}")
        self.name = name
        self.languages = SYSTEM_LANGS[name]
        self.auto_detect = name in AUTO_DETECT
        self.host = host
        self.port = port if port is not None else SERVING[name].port
        self._post = post or _urllib_post
        self._timeout = timeout
        self._clock = clock

    def meta(self) -> dict[str, object]:
        return {"kind": "serving", "host": self.host, "port": self.port}

    def transcribe(self, wav: bytes, language: str | None) -> Hypothesis:
        record: dict[str, str] = {"audio_b64": base64.b64encode(wav).decode()}
        if language:
            record["language"] = language
        url = f"http://{self.host}:{self.port}/invocations"
        t0 = self._clock()
        try:
            out = self._post(url, {"dataframe_records": [record]}, self._timeout)
            latency = self._clock() - t0
            pred = out["predictions"][0]
            text, language = (pred.get("text") or "").strip(), pred.get("language")
        except BackendError:
            raise
        except Exception as e:  # connection refused, timeout, bad JSON, unexpected payload
            raise BackendError(f"{type(e).__name__}: {e}") from e
        return Hypothesis(text=text, language=language, latency_s=latency)


# --- Google Speech-to-Text v2 -------------------------------------------------------------

Recognize = Callable[[list[str], bytes], tuple[list[str], str | None]]


def google_recognizer(project: str, location: str, model: str, client=None) -> Recognize:
    """Return a ``recognize(language_codes, wav_bytes) -> (segments, language_code)`` closure.

    ``client`` is injectable for tests; by default a regional ``SpeechClient`` is built
    (``<location>-speech.googleapis.com``), which is what Chirp 3 in ``us`` / ``eu`` needs.
    """
    from google.cloud.speech_v2.types import cloud_speech

    if client is None:
        from google.api_core.client_options import ClientOptions
        from google.cloud.speech_v2 import SpeechClient

        client = SpeechClient(client_options=ClientOptions(api_endpoint=f"{location}-speech.googleapis.com"))

    recognizer = f"projects/{project}/locations/{location}/recognizers/_"

    def recognize(language_codes: list[str], content: bytes) -> tuple[list[str], str | None]:
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=language_codes,
            model=model,
        )
        request = cloud_speech.RecognizeRequest(recognizer=recognizer, config=config, content=content)
        response = client.recognize(request=request)
        segments = [r.alternatives[0].transcript for r in response.results if r.alternatives]
        code = next((r.language_code for r in response.results if getattr(r, "language_code", "")), None)
        return segments, code

    return recognize


class ChirpBackend:
    def __init__(
        self,
        project: str,
        location: str = "us",
        model: str = "chirp_3",
        recognize: Recognize | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = CHIRP_NAME
        self.languages = SYSTEM_LANGS[CHIRP_NAME]
        self.auto_detect = True
        self.project = project
        self.location = location
        self.model = model
        self._recognize = recognize or google_recognizer(project, location, model)
        self._clock = clock

    def meta(self) -> dict[str, object]:
        return {"kind": "chirp", "project": self.project, "location": self.location, "model": self.model,
                "auto_language_codes": list(CHIRP_AUTO_CODES)}

    def transcribe(self, wav: bytes, language: str | None) -> Hypothesis:
        codes = [CHIRP_LANG_CODES[language]] if language else list(CHIRP_AUTO_CODES)
        t0 = self._clock()
        try:
            segments, code = self._recognize(codes, wav)
        except Exception as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e
        latency = self._clock() - t0
        text = " ".join(s.strip() for s in segments if s and s.strip()).strip()
        lang = _CHIRP_LANG_BACK.get(code.lower()) or code.split("-")[0].lower() if code else None
        return Hypothesis(text=text, language=lang, latency_s=latency)

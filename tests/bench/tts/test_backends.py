import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from mlflow_registry.bench.backends import BackendError
from mlflow_registry.bench.tts import ttfb
from mlflow_registry.bench.tts.backends import (
    CLONING,
    TTS_MODELS,
    ElevenLabsBackend,
    GoogleTtsBackend,
    TtsServingBackend,
)
from mlflow_registry.bench.tts.manifest import Reference

from .conftest import make_wav

REF = Reference(wav=b"REFWAV", text="câu tham chiếu")


def _post(wav: bytes, sink: list):
    def post(url, payload, timeout):
        sink.append((url, payload))
        return {"predictions": [{"audio_b64": base64.b64encode(wav).decode(), "sample_rate": 24000}]}
    return post


def test_tts_models_come_from_the_catalog():
    assert set(TTS_MODELS) == {"voxcpm2", "vieneu-tts-v3-turbo", "kokoro-82m", "qwen3-tts-1.7b-base"}
    assert CLONING == {"voxcpm2", "vieneu-tts-v3-turbo", "qwen3-tts-1.7b-base"}


@pytest.mark.parametrize("name, ref_text", [("voxcpm2", True), ("qwen3-tts-1.7b-base", True),
                                            ("vieneu-tts-v3-turbo", False)])
def test_cloning_models_send_the_reference(name, ref_text):
    calls = []
    b = TtsServingBackend(name, REF, post=_post(make_wav(1.5), calls), clock=iter([0.0, 0.4]).__next__)
    syn = b.synthesize("xin chào", "vi")
    url, payload = calls[0]
    rec = payload["dataframe_records"][0]
    assert url.endswith("/invocations")
    assert rec["ref_audio_b64"] == base64.b64encode(b"REFWAV").decode()
    assert ("ref_text" in rec) is ref_text
    assert "voice" not in rec
    assert syn.sample_rate == 24000 and syn.duration_s == pytest.approx(1.5) and syn.latency_s == pytest.approx(0.4)


def test_preset_model_sends_voice_not_reference():
    calls = []
    TtsServingBackend("kokoro-82m", REF, post=_post(make_wav(1.0), calls)).synthesize("hi", "en")
    rec = calls[0][1]["dataframe_records"][0]
    assert rec["voice"] == "af_heart" and "ref_audio_b64" not in rec


def test_serving_errors_become_backend_errors():
    def post(url, payload, timeout):
        return {"predictions": [{}]}
    with pytest.raises(BackendError):
        TtsServingBackend("voxcpm2", REF, post=post).synthesize("x", "vi")


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        TtsServingBackend("whisper-large-v3", REF)


def test_google_voice_names_and_quota_retry():
    class Quota(Exception):
        code = 429

    calls, sleeps = [], []

    def speak(text, code, voice):
        calls.append((code, voice))
        if len(calls) == 1:
            raise Quota("slow down")
        return make_wav(2.0)

    b = GoogleTtsBackend(speak=speak, sleep=sleeps.append)
    syn = b.synthesize("xin chào", "vi")
    assert calls == [("vi-VN", "vi-VN-Chirp3-HD-Aoede")] * 2
    assert sleeps == [5.0] and syn.duration_s == pytest.approx(2.0)
    assert b.voice_name("en") == "en-US-Chirp3-HD-Aoede"


def test_google_other_errors_are_not_retried():
    def speak(text, code, voice):
        raise ValueError("bad voice")
    with pytest.raises(BackendError, match="bad voice"):
        GoogleTtsBackend(speak=speak, sleep=lambda s: None).synthesize("x", "vi")


def _mp3(seconds: float, sr: int = 44_100) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="MP3")
    return buf.getvalue()


def test_elevenlabs_posts_robo_be_model_and_decodes_mp3():
    seen = []

    def fetch(url, headers, body):
        seen.append((url, headers, body))
        mp3 = _mp3(1.0)
        yield mp3[:1000]
        yield mp3[1000:]

    b = ElevenLabsBackend(api_key="k", fetch=fetch, clock=iter([0.0, 0.8]).__next__)
    syn = b.synthesize("xin chào", "vi")
    url, headers, body = seen[0]
    assert "/v1/text-to-speech/0ggMuQ1r9f9jqBu50nJn?output_format=mp3_44100_128" in url
    assert headers["xi-api-key"] == "k"
    assert json.loads(body) == {"text": "xin chào", "model_id": "eleven_v3"}
    assert syn.wav.startswith(b"RIFF") and syn.sample_rate == 44_100
    assert syn.duration_s == pytest.approx(1.0, abs=0.1) and syn.latency_s == pytest.approx(0.8)


def test_elevenlabs_needs_a_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(BackendError, match="ELEVENLABS_API_KEY"):
        ElevenLabsBackend()


def test_elevenlabs_retries_429():
    calls, sleeps = [], []

    def fetch(url, headers, body):
        calls.append(url)
        if len(calls) == 1:
            raise BackendError("HTTP 429 from ElevenLabs: too_many_concurrent_requests")
        yield _mp3(0.5)

    ElevenLabsBackend(api_key="k", fetch=fetch, sleep=sleeps.append).synthesize("hi", "en")
    assert len(calls) == 2 and sleeps == [5.0]


def test_mp3_stream_timing_uses_first_bytes_and_decoded_length():
    mp3 = _mp3(2.0)
    t = ttfb.time_stream(iter([mp3[:500], mp3[500:]]), 0, clock=iter([0.0, 0.3, 1.0]).__next__)
    assert t.ttfb_s == pytest.approx(0.3) and t.total_s == pytest.approx(1.0)
    assert t.audio_s == pytest.approx(2.0, abs=0.1) and t.chunks == 2

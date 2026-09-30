import base64

import pytest

from mlflow_registry.bench.backends import BackendError
from mlflow_registry.bench.tts.backends import (
    CLONING,
    TTS_MODELS,
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

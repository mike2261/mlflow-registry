import base64

import pytest

from mlflow_registry.bench import backends
from mlflow_registry.bench.backends import BackendError, ChirpBackend, ServingBackend

WAV = b"RIFF....WAVEfmt fake"


def _fake_clock():
    t = [0.0]

    def clock():
        t[0] += 0.25
        return t[0]
    return clock


def test_capability_tables_cover_the_six_systems():
    assert set(backends.SYSTEM_LANGS) == {
        "qwen3-asr-1.7b", "granite-speech-4.1-2b", "gipformer1.5-68m-rnnt",
        "parakeet-ctc-0.6b-vietnamese", "whisper-large-v3", "chirp_3",
    }
    assert backends.SYSTEM_LANGS["gipformer1.5-68m-rnnt"] == frozenset({"vi"})
    assert backends.SYSTEM_LANGS["granite-speech-4.1-2b"] == frozenset({"en"})
    assert backends.AUTO_DETECT == frozenset({"qwen3-asr-1.7b", "whisper-large-v3", "chirp_3"})
    assert backends.STT_MODELS == (
        "qwen3-asr-1.7b", "granite-speech-4.1-2b", "gipformer1.5-68m-rnnt",
        "parakeet-ctc-0.6b-vietnamese", "whisper-large-v3",
    )


def test_serving_backend_posts_dataframe_records_and_parses_prediction():
    seen = {}

    def post(url, payload, timeout):
        seen.update(url=url, payload=payload, timeout=timeout)
        return {"predictions": [{"text": " anh em ", "language": "vi"}]}

    b = ServingBackend("whisper-large-v3", host="devbox", post=post, clock=_fake_clock())
    hyp = b.transcribe(WAV, "vi")

    assert seen["url"] == "http://devbox:5005/invocations"
    rec = seen["payload"]["dataframe_records"][0]
    assert base64.b64decode(rec["audio_b64"]) == WAV
    assert rec["language"] == "vi"
    assert hyp.text == "anh em" and hyp.language == "vi"
    assert hyp.latency_s == pytest.approx(0.25)
    assert b.languages == frozenset({"vi", "en"}) and b.auto_detect is True
    assert b.meta() == {"kind": "serving", "host": "devbox", "port": 5005}


def test_serving_backend_omits_language_when_none_and_accepts_explicit_port():
    seen = {}

    def post(url, payload, timeout):
        seen.update(url=url, payload=payload)
        return {"predictions": [{"text": "cảm ơn", "language": None}]}

    b = ServingBackend("gipformer1.5-68m-rnnt", port=6003, post=post)
    hyp = b.transcribe(WAV, None)
    assert seen["url"] == "http://localhost:6003/invocations"
    assert "language" not in seen["payload"]["dataframe_records"][0]
    assert hyp.language is None and b.auto_detect is False


def test_serving_backend_wraps_transport_errors():
    def post(url, payload, timeout):
        raise OSError("connection refused")

    b = ServingBackend("whisper-large-v3", post=post)
    with pytest.raises(BackendError, match="connection refused"):
        b.transcribe(WAV, "vi")


def test_chirp_backend_maps_hint_to_bcp47_and_auto_to_vi_en_restricted():
    calls = []

    def recognize(codes, content):
        calls.append((codes, content))
        return (["Good ", "morning."], "en-US")

    b = ChirpBackend(project="p", location="us", recognize=recognize, clock=_fake_clock())
    hyp = b.transcribe(WAV, "en")
    assert calls[-1] == (["en-US"], WAV)
    assert hyp.text == "Good morning."          # segments joined, outer whitespace stripped
    assert hyp.language == "en"                 # BCP-47 mapped back to the manifest code
    assert hyp.latency_s == pytest.approx(0.25)

    b.transcribe(WAV, None)
    assert calls[-1][0] == ["vi-VN", "en-US"]   # restricted detection: the product only hears vi/en
    b.transcribe(WAV, "vi")
    assert calls[-1][0] == ["vi-VN"]
    assert b.name == "chirp_3" and b.auto_detect is True
    assert b.meta() == {"kind": "chirp", "project": "p", "location": "us", "model": "chirp_3",
                        "auto_language_codes": ["vi-VN", "en-US"]}


def test_chirp_backend_with_no_results_returns_empty_text_and_unknown_language():
    b = ChirpBackend(project="p", recognize=lambda codes, content: ([], None))
    hyp = b.transcribe(WAV, "vi")
    assert hyp.text == "" and hyp.language is None


def test_chirp_backend_wraps_api_errors():
    def recognize(codes, content):
        raise RuntimeError("403 permission denied")

    b = ChirpBackend(project="p", recognize=recognize)
    with pytest.raises(BackendError, match="permission denied"):
        b.transcribe(WAV, "vi")


def test_google_recognizer_builds_a_regional_v2_request():
    cloud_speech = pytest.importorskip("google.cloud.speech_v2.types").cloud_speech

    class FakeResult:
        def __init__(self, text, code):
            self.alternatives = [type("Alt", (), {"transcript": text})()]
            self.language_code = code

    class FakeClient:
        def __init__(self):
            self.requests = []

        def recognize(self, request):
            self.requests.append(request)
            return type("Resp", (), {"results": [FakeResult("xin chào", "vi-VN")]})()

    client = FakeClient()
    rec = backends.google_recognizer("proj", "us", "chirp_3", client=client)
    segments, code = rec(["vi-VN"], WAV)

    assert segments == ["xin chào"] and code == "vi-VN"
    req = client.requests[0]
    assert isinstance(req, cloud_speech.RecognizeRequest)
    assert req.recognizer == "projects/proj/locations/us/recognizers/_"
    assert list(req.config.language_codes) == ["vi-VN"]
    assert req.config.model == "chirp_3"
    assert req.content == WAV


def test_serving_backend_turns_malformed_prediction_into_backend_error():
    b = ServingBackend("whisper-large-v3", post=lambda url, payload, timeout: {"predictions": ["anh em"]})
    with pytest.raises(BackendError):
        b.transcribe(WAV, "vi")

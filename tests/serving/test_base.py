import base64
import io

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from mlflow_registry.serving import base
from tests.serving._fakes import FakeStt, FakeTts


def _wav_b64(n: int, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def weights(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "weights.txt").write_text("fake-weights")
    return d


def _save_and_load(tmp_path, model, signature):
    path = tmp_path / "pyfunc"
    mlflow.pyfunc.save_model(str(path), python_model=model, signature=signature, pip_requirements=["mlflow"])
    return mlflow.pyfunc.load_model(str(path))


# --- STT -----------------------------------------------------------------------


def test_stt_predict_decodes_audio_and_returns_records(tmp_path, weights):
    loaded = _save_and_load(tmp_path, FakeStt("m", "1", weights_dir=str(weights)), base.STT_SIGNATURE)
    out = loaded.predict(pd.DataFrame({"audio_b64": [_wav_b64(32_000, 16_000)]}))
    assert list(out.columns) == ["text", "language"]
    assert out.iloc[0]["text"] == "32000 samples from fake-weights"
    assert out.iloc[0]["language"] == "auto"


def test_stt_resamples_to_16k_and_passes_language(tmp_path, weights):
    loaded = _save_and_load(tmp_path, FakeStt("m", "1", weights_dir=str(weights)), base.STT_SIGNATURE)
    df = pd.DataFrame({"audio_b64": [_wav_b64(48_000, 48_000)], "language": ["vi"]})
    out = loaded.predict(df)
    assert out.iloc[0]["text"].startswith("16000 samples")
    assert out.iloc[0]["language"] == "vi"


def test_stt_handles_batch_and_null_language(tmp_path, weights):
    loaded = _save_and_load(tmp_path, FakeStt("m", "1", weights_dir=str(weights)), base.STT_SIGNATURE)
    df = pd.DataFrame({"audio_b64": [_wav_b64(16_000, 16_000)] * 2, "language": [None, "en"]})
    out = loaded.predict(df)
    assert list(out["language"]) == ["auto", "en"]


def test_stt_bad_audio_raises_value_error(tmp_path, weights):
    loaded = _save_and_load(tmp_path, FakeStt("m", "1", weights_dir=str(weights)), base.STT_SIGNATURE)
    with pytest.raises(ValueError):
        loaded.predict(pd.DataFrame({"audio_b64": ["bm90IGF1ZGlv"]}))


def test_load_context_uses_ensure_weights_when_no_override(tmp_path, weights, monkeypatch):
    seen = {}

    def fake_ensure(name, version, cache_root=None, download=None):
        seen["args"] = (name, version)
        return weights

    monkeypatch.setattr(base, "ensure_weights", fake_ensure)
    loaded = _save_and_load(tmp_path, FakeStt("whisper-large-v3", "1"), base.STT_SIGNATURE)
    out = loaded.predict(pd.DataFrame({"audio_b64": [_wav_b64(16_000, 16_000)]}))
    assert seen["args"] == ("whisper-large-v3", "1")
    assert "fake-weights" in out.iloc[0]["text"]


# --- TTS -----------------------------------------------------------------------


def test_tts_predict_returns_wav_b64_and_sample_rate(tmp_path, weights):
    loaded = _save_and_load(tmp_path, FakeTts("t", "1", weights_dir=str(weights)), base.TTS_SIGNATURE)
    out = loaded.predict(pd.DataFrame({"text": ["hello"]}))
    assert list(out.columns) == ["audio_b64", "sample_rate"]
    assert int(out.iloc[0]["sample_rate"]) == 8_000
    data, sr = sf.read(io.BytesIO(base64.b64decode(out.iloc[0]["audio_b64"])))
    assert sr == 8_000 and data.shape[0] == 5


def test_tts_optional_columns_default_to_none(tmp_path, weights):
    model = FakeTts("t", "1", weights_dir=str(weights))
    loaded = _save_and_load(tmp_path, model, base.TTS_SIGNATURE)
    loaded.predict(pd.DataFrame({"text": ["hi"]}))
    req = loaded._model_impl.python_model.last_request
    assert (req.voice, req.language, req.ref_audio, req.ref_text) == (None, None, None, None)


def test_tts_reference_audio_is_decoded_at_native_rate(tmp_path, weights):
    model = FakeTts("t", "1", weights_dir=str(weights))
    loaded = _save_and_load(tmp_path, model, base.TTS_SIGNATURE)
    df = pd.DataFrame({
        "text": ["ab"],
        "voice": ["Mai Anh"],
        "ref_audio_b64": [_wav_b64(2_400, 24_000)],
        "ref_text": ["ref transcript"],
    })
    out = loaded.predict(df)
    req = loaded._model_impl.python_model.last_request
    assert req.voice == "Mai Anh" and req.ref_text == "ref transcript"
    assert req.ref_audio[1] == 24_000 and req.ref_audio[0].shape[0] == 2_400
    data, _ = sf.read(io.BytesIO(base64.b64decode(out.iloc[0]["audio_b64"])))
    assert data.shape[0] == 2 + 2_400


def test_tts_empty_text_is_rejected(tmp_path, weights):
    loaded = _save_and_load(tmp_path, FakeTts("t", "1", weights_dir=str(weights)), base.TTS_SIGNATURE)
    with pytest.raises(ValueError):
        loaded.predict(pd.DataFrame({"text": ["   "]}))


# --- request shapes the scoring server may hand us ------------------------------


def test_rows_accepts_dict_of_lists_and_list_of_dicts():
    assert base.rows({"a": [1, 2], "b": ["x", None]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": None}]
    assert base.rows([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]
    assert base.rows({"a": 1}) == [{"a": 1}]
    assert base.rows(pd.DataFrame({"a": [1.0, float("nan")]}))[1]["a"] is None

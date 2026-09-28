import base64
import io

import numpy as np
import pytest
import soundfile as sf

from mlflow_registry.serving import audio


def _wav_b64(samples: np.ndarray, sr: int, subtype: str = "PCM_16") -> str:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype=subtype)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_returns_float32_mono_at_target_rate():
    sr = 16_000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    out = audio.decode_b64(_wav_b64(tone, sr), target_sr=16_000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] == sr
    assert np.abs(out - tone).max() < 1e-3  # PCM_16 quantisation only


def test_decode_downmixes_stereo_and_resamples():
    sr = 48_000
    left = np.linspace(-0.5, 0.5, sr, dtype=np.float32)
    stereo = np.stack([left, left], axis=1)
    out = audio.decode_b64(_wav_b64(stereo, sr), target_sr=16_000)
    assert out.ndim == 1
    assert abs(out.shape[0] - 16_000) <= 2
    assert np.abs(out).max() <= 1.0


def test_decode_accepts_data_url_prefix():
    tone = np.zeros(1600, dtype=np.float32)
    payload = "data:audio/wav;base64," + _wav_b64(tone, 16_000)
    assert audio.decode_b64(payload, target_sr=16_000).shape[0] == 1600


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        audio.decode_b64("not base64 at all!!", target_sr=16_000)
    with pytest.raises(ValueError):
        audio.decode_b64(base64.b64encode(b"hello").decode(), target_sr=16_000)


def test_encode_roundtrip_is_16bit_pcm_wav():
    sr = 24_000
    tone = 0.25 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    b64 = audio.encode_b64(tone, sr)
    data, got_sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    info = sf.info(io.BytesIO(base64.b64decode(b64)))
    assert got_sr == sr
    assert info.subtype == "PCM_16"
    assert np.abs(data - tone).max() < 1e-3


def test_encode_clips_out_of_range_samples():
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    data, _ = sf.read(io.BytesIO(base64.b64decode(audio.encode_b64(loud, 8000))), dtype="float32")
    assert data.max() <= 1.0 and data.min() >= -1.0


def test_encode_accepts_float64_and_torch_like_2d():
    stereo_ish = np.zeros((1, 100), dtype=np.float64)  # (channels, samples) as torch gives
    b64 = audio.encode_b64(stereo_ish, 16_000)
    data, _ = sf.read(io.BytesIO(base64.b64decode(b64)))
    assert data.shape[0] == 100

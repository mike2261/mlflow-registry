import io
import json

import numpy as np
import pytest
import soundfile as sf

from mlflow_registry.bench import manifest, opus

pytest.importorskip("opuslib")          # needs the bench extra and the system libopus


def _wav(samples: np.ndarray, sr: int = 16_000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _read(wav: bytes) -> tuple[np.ndarray, int, sf._SoundFileInfo]:
    info = sf.info(io.BytesIO(wav))
    data, sr = sf.read(io.BytesIO(wav), dtype="int16")
    return data, sr, info


def test_roundtrip_keeps_robo_be_wire_format_and_pads_the_last_frame():
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(330) / 16_000)).astype(np.float32)
    data, sr, info = _read(opus.roundtrip_wav(_wav(tone)))
    assert sr == 16_000 and info.channels == 1 and info.subtype == "PCM_16"
    assert len(data) == 640          # 330 samples -> two 20 ms frames of 320, last one zero-padded


def test_roundtrip_changes_the_audio_but_keeps_its_energy():
    src = manifest.FIXTURES_DIR / "04.vi_short.wav"
    original, _ = sf.read(src, dtype="int16")
    decoded, sr, _ = _read(opus.roundtrip_wav(src.read_bytes()))
    assert sr == 16_000
    assert 0 <= len(decoded) - len(original) < 320
    assert not np.array_equal(decoded[: len(original)], original)      # lossy codec really ran
    rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))  # noqa: E731
    assert rms(decoded) == pytest.approx(rms(original), rel=0.5)


def test_roundtrip_is_deterministic():
    src = (manifest.FIXTURES_DIR / "13.en_short.wav").read_bytes()
    assert opus.roundtrip_wav(src) == opus.roundtrip_wav(src)


def test_roundtrip_rejects_audio_that_is_not_16k_mono():
    with pytest.raises(ValueError, match="16000"):
        opus.roundtrip_wav(_wav(np.zeros(800, dtype=np.float32), sr=8_000))


def test_make_opus_fixtures_mirrors_the_manifest_under_a_new_hash(tmp_path):
    dst = opus.make_opus_fixtures(manifest.FIXTURES_DIR, tmp_path / "opus24")
    src_utts = manifest.load(manifest.FIXTURES_DIR)
    dst_utts = manifest.load(dst)
    assert [(u.id, u.file, u.text, u.lang, u.category, u.en_words) for u in dst_utts] == \
           [(u.id, u.file, u.text, u.lang, u.category, u.en_words) for u in src_utts]
    for s, d in zip(src_utts, dst_utts):
        assert 0 <= d.duration_s - s.duration_s < 0.02 + 1e-6
    assert manifest.dataset_hash(dst) != manifest.dataset_hash(manifest.FIXTURES_DIR)

    source = json.loads((dst / opus.SOURCE_NAME).read_text())
    assert source == {
        "source_dataset": "stt-fixtures",
        "source_dataset_hash": manifest.dataset_hash(manifest.FIXTURES_DIR),
        "codec": "opus", "bitrate": 24000, "application": "voip", "frame_ms": 20, "sample_rate": 16000,
        "note": "encoded and decoded with opuslib as robo-be's STT clients and server do",
    }

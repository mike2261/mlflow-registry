"""Base64 WAV <-> numpy helpers shared by every serving wrapper.

Audio crosses the REST boundary as base64-encoded WAV. Inside the wrapper
everything is float32 in [-1, 1]: STT models get mono at 16 kHz, TTS models
hand back (samples, sample_rate) which we encode as 16-bit PCM WAV.
"""

import base64
import binascii
import contextlib
import io
import math
import os
import tempfile
from collections.abc import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_DATA_URL_MARKER = ";base64,"


def _read(payload: str) -> tuple[np.ndarray, int]:
    """base64 (or data URL) -> (float32 mono, native sample rate)."""
    if _DATA_URL_MARKER in payload[:64]:
        payload = payload.split(_DATA_URL_MARKER, 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"audio is not valid base64: {e}") from e
    try:
        samples, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise ValueError(f"audio bytes are not a readable sound file: {e}") from e
    mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


def decode_b64_native(payload: str) -> tuple[np.ndarray, int]:
    """Decode without resampling; for reference clips whose runtime wants the original rate."""
    return _read(payload)


def decode_b64(payload: str, target_sr: int) -> np.ndarray:
    """Decode a base64 audio file into float32 mono at ``target_sr``.

    Accepts a bare base64 string or a ``data:audio/...;base64,`` URL. Any
    container libsndfile reads (WAV, FLAC, OGG) works; MP3 support depends on
    the libsndfile build. Raises ``ValueError`` on undecodable input.
    """
    mono, sr = _read(payload)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // g, sr // g)
    return np.ascontiguousarray(mono, dtype=np.float32)


def encode_b64(samples: np.ndarray, sample_rate: int) -> str:
    """Encode float samples as base64 16-bit PCM WAV.

    Accepts 1-D ``(n,)`` or 2-D ``(channels, n)`` / ``(n, channels)`` arrays;
    multi-channel input is downmixed to mono. Values are clipped to [-1, 1].
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 2:
        # torch-style (channels, n) has the small dim first; soundfile wants (n, channels)
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.mean(axis=1)
    elif arr.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {arr.shape}")
    arr = np.clip(arr, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, arr, int(sample_rate), format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@contextlib.contextmanager
def temp_wav(samples: np.ndarray, sample_rate: int) -> Iterator[str]:
    """Write ``samples`` to a temporary WAV file and yield its path.

    Several runtimes (NeMo, VoxCPM, VieNeu) only accept file paths for audio.
    The file is removed when the block exits.
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, np.asarray(samples, dtype=np.float32), int(sample_rate), subtype="PCM_16")
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

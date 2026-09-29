"""Opus condition: put fixtures through the codec robo-be's STT path uses, then decode.

robo-be's clients encode 16 kHz mono PCM16 into 20 ms Opus packets at 24 kbps in VOIP
mode, zero-padding the last frame (``benchmarks/stt/bench_wer.py``, ``bench_stt.py``), and
the server decodes each packet with ``opuslib.Decoder(16000, 1)``
(``robo_be/services/audio/opus_codec.py``). ``roundtrip_wav`` does exactly that offline, so
a model sees the same audio it would receive from the robot.

``make_opus_fixtures`` turns a fixture folder into a second one with the same manifest
(ids, texts, languages) and decoded WAVs. It gets its own dataset hash, so its runs can
never be mixed up with the clean ones.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mlflow_registry.bench import manifest

SAMPLE_RATE = 16_000
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2                  # PCM16
SOURCE_NAME = "source.json"


def roundtrip_pcm16(pcm: bytes, bitrate: int = 24_000, application: str = "voip") -> bytes:
    """Encode PCM16 mono 16 kHz into 20 ms Opus packets and decode them back."""
    import opuslib

    app = opuslib.APPLICATION_AUDIO if application == "audio" else opuslib.APPLICATION_VOIP
    encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, app)
    encoder.bitrate = bitrate
    decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)
    out = bytearray()
    for i in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[i:i + FRAME_BYTES]
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        packet = encoder.encode(chunk, FRAME_SAMPLES)
        out += decoder.decode(packet, FRAME_SAMPLES)
    return bytes(out)


def roundtrip_wav(wav: bytes, bitrate: int = 24_000, application: str = "voip") -> bytes:
    """``roundtrip_pcm16`` for a 16 kHz mono WAV; returns a PCM16 WAV."""
    samples, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
    if sr != SAMPLE_RATE or samples.shape[1] != CHANNELS:
        raise ValueError(f"expected {SAMPLE_RATE} Hz mono, got {sr} Hz with {samples.shape[1]} channels")
    decoded = roundtrip_pcm16(samples[:, 0].tobytes(), bitrate, application)
    buf = io.BytesIO()
    sf.write(buf, np.frombuffer(decoded, dtype=np.int16), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_opus_fixtures(src_dir: Path, dst_dir: Path, bitrate: int = 24_000,
                       application: str = "voip") -> Path:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for utt in manifest.load(src_dir):
        wav = roundtrip_wav(manifest.audio_bytes(src_dir, utt), bitrate, application)
        (dst_dir / utt.file).write_bytes(wav)
        duration = sf.info(io.BytesIO(wav)).frames / SAMPLE_RATE
        lines.append(json.dumps({
            "id": utt.id, "file": utt.file, "text": utt.text, "lang": utt.lang,
            "category": utt.category, "en_words": list(utt.en_words), "duration_s": round(duration, 2),
        }, ensure_ascii=False))
    (dst_dir / manifest.MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    source = {
        "source_dataset": src_dir.name,
        "source_dataset_hash": manifest.dataset_hash(src_dir),
        "codec": "opus", "bitrate": bitrate, "application": application,
        "frame_ms": FRAME_MS, "sample_rate": SAMPLE_RATE,
        "note": "encoded and decoded with opuslib as robo-be's STT clients and server do",
    }
    (dst_dir / SOURCE_NAME).write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return dst_dir

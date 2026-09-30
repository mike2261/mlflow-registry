import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mlflow_registry.bench import datasets, manifest


def _wav(seconds: float, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16_000 * seconds), dtype=np.float32), 16_000, format="WAV", subtype=subtype)
    return buf.getvalue()


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _fleurs(tmp: Path, lang: str) -> tuple[Path, Path]:
    tsv = tmp / f"fleurs_{lang}.tsv"
    rows = [
        ("1921", "111.wav", "Mặt khác, băng và tuyết.", "mặt khác băng và tuyết", "m ặ t |", "32000", "FEMALE"),
        ("1921", "222.wav", "Mặt khác, băng và tuyết.", "mặt khác băng và tuyết", "m ặ t |", "16000", "MALE"),
    ]
    tsv.write_text("\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
    # real FLEURS WAVs are 32-bit float
    tar = _tar(tmp / f"fleurs_{lang}.tar.gz", {"test/111.wav": _wav(2.0, "FLOAT"), "test/222.wav": _wav(1.0, "FLOAT"),
                                               "test/999.wav": _wav(0.5, "FLOAT")})
    return tsv, tar


def test_fleurs_rows_use_the_normalized_transcription_and_the_wav_stem_as_id(tmp_path):
    tsv, tar = _fleurs(tmp_path, "vi")
    rows = datasets.build_fleurs(tmp_path / "out", "vi", tsv, tar)
    assert [r["id"] for r in rows] == ["fleurs-vi-111", "fleurs-vi-222"]    # same sentence, two speakers
    assert rows[0] == {"id": "fleurs-vi-111", "file": "fleurs_vi/111.wav", "text": "mặt khác băng và tuyết",
                       "lang": "vi", "category": "fleurs_vi", "en_words": [], "duration_s": 2.0}
    written = sf.info(tmp_path / "out" / "fleurs_vi" / "222.wav")
    assert (written.samplerate, written.channels, written.subtype) == (16_000, 1, "PCM_16")
    assert not (tmp_path / "out" / "fleurs_vi" / "999.wav").exists()      # not in the TSV: not extracted


def test_float_audio_keeps_its_signal_when_stored_as_pcm16(tmp_path):
    t = np.arange(16_000) / 16_000
    tone = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)   # FLEURS-like level, 32-bit float
    buf = io.BytesIO()
    sf.write(buf, tone, 16_000, format="WAV", subtype="FLOAT")
    tsv = tmp_path / "t.tsv"
    tsv.write_text("1\t1.wav\tRaw.\tnorm\tc\t16000\tMALE\n", encoding="utf-8")
    tar = _tar(tmp_path / "t.tar.gz", {"test/1.wav": buf.getvalue()})
    datasets.build_fleurs(tmp_path / "out", "vi", tsv, tar)
    pcm, sr = sf.read(tmp_path / "out" / "fleurs_vi" / "1.wav", dtype="float32")
    assert sf.info(tmp_path / "out" / "fleurs_vi" / "1.wav").subtype == "PCM_16"
    assert np.abs(pcm).max() == pytest.approx(0.25, abs=1e-3)       # not silence, not clipped
    assert np.abs(pcm - tone).max() < 1e-3


def test_vivos_reads_only_the_test_split(tmp_path):
    prompts = "VIVOSDEV01_R002 KHÁCH SẠN\nVIVOSDEV02_R105 TÔI ĐI HỌC\n"
    tar = _tar(tmp_path / "vivos.tar.gz", {
        "vivos/test/prompts.txt": prompts.encode(),
        "vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav": _wav(1.5),
        "vivos/test/waves/VIVOSDEV02/VIVOSDEV02_R105.wav": _wav(3.0),
        "vivos/train/prompts.txt": b"VIVOSSPK01_R001 KHONG\n",
        "vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav": _wav(1.0),
    })
    rows = datasets.build_vivos(tmp_path / "out", tar)
    assert [r["id"] for r in rows] == ["vivos-VIVOSDEV01_R002", "vivos-VIVOSDEV02_R105"]
    assert rows[1] == {"id": "vivos-VIVOSDEV02_R105", "file": "vivos/VIVOSDEV02_R105.wav",
                       "text": "tôi đi học", "lang": "vi", "category": "vivos_vi", "en_words": [],
                       "duration_s": 3.0}
    assert not (tmp_path / "out" / "vivos" / "VIVOSSPK01_R001.wav").exists()


def test_write_dataset_makes_a_loadable_manifest_with_sources(tmp_path):
    tsv, tar = _fleurs(tmp_path, "en")
    out = tmp_path / "public"
    rows = datasets.build_fleurs(out, "en", tsv, tar)
    datasets.write_dataset(out, rows, sources=[{"name": "fleurs_en", "license": "CC-BY-4.0"}])
    utts = manifest.load(out)
    assert [u.id for u in utts] == ["fleurs-en-111", "fleurs-en-222"] and utts[0].lang == "en"
    assert manifest.audio_bytes(out, utts[0]).startswith(b"RIFF")
    src = json.loads((out / datasets.SOURCE_NAME).read_text())
    assert src["utterances"] == 2 and src["sources"] == [{"name": "fleurs_en", "license": "CC-BY-4.0"}]
    assert manifest.dataset_hash(out).startswith("sha256:")

"""Build public benchmark datasets in the harness's manifest format.

* **FLEURS** (google/fleurs on Hugging Face, CC-BY-4.0): read sentences, many speakers.
  ``data/<lang>/test.tsv`` columns: id, file, raw text, normalized text, characters,
  num_samples, gender. We use the normalized text (lowercase, no punctuation) and the WAV
  stem as the utterance id (one sentence id is read by several speakers).
* **VIVOS** (AILAB-VNUHCM/vivos, CC-BY-NC-SA-4.0, non-commercial): Vietnamese read speech.
  Only ``vivos/test`` is used: ``prompts.txt`` ("<ID> <UPPERCASE TEXT>") and
  ``waves/<speaker>/<ID>.wav``.

Audio is 16 kHz mono in both sources and is extracted as-is. The built folder is gitignored
(about 1 GB); ``source.json`` records where it came from and under which licence.
"""
from __future__ import annotations

import io
import json
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

SOURCE_NAME = "source.json"
FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
VIVOS_URL = "https://huggingface.co/datasets/AILAB-VNUHCM/vivos/resolve/main/data/vivos.tar.gz"
_FLEURS_LANG = {"vi": "vi_vn", "en": "en_us"}

SOURCES = {
    "fleurs_vi": {"name": "fleurs_vi", "url": f"{FLEURS_BASE}/vi_vn", "split": "test", "license": "CC-BY-4.0"},
    "fleurs_en": {"name": "fleurs_en", "url": f"{FLEURS_BASE}/en_us", "split": "test", "license": "CC-BY-4.0"},
    "vivos_vi": {"name": "vivos_vi", "url": VIVOS_URL, "split": "test",
                 "license": "CC-BY-NC-SA-4.0 (non-commercial)"},
}


def _row(uid: str, file: str, text: str, lang: str, category: str, wav: bytes) -> dict:
    info = sf.info(io.BytesIO(wav))
    return {"id": uid, "file": file, "text": text, "lang": lang, "category": category,
            "en_words": [], "duration_s": round(info.frames / info.samplerate, 2)}


def build_fleurs(dst: Path, lang: str, tsv: Path, tar: Path) -> list[dict]:
    """Extract the test WAVs listed in ``tsv`` from ``tar`` into ``dst/fleurs_<lang>/``."""
    category = f"fleurs_{lang}"
    wanted: dict[str, str] = {}
    order: list[str] = []
    for line in Path(tsv).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        wanted[f"test/{cols[1]}"] = cols[3].strip()
        order.append(f"test/{cols[1]}")
    out_dir = Path(dst) / category
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    with tarfile.open(tar, "r:*") as tf:
        for member in tf:
            if member.name not in wanted or not member.isfile():
                continue
            wav = tf.extractfile(member).read()
            name = Path(member.name).name
            (out_dir / name).write_bytes(wav)
            rows[member.name] = _row(f"fleurs-{lang}-{Path(name).stem}", f"{category}/{name}",
                                     wanted[member.name], lang, category, wav)
    missing = [m for m in order if m not in rows]
    if missing:
        raise FileNotFoundError(f"{len(missing)} FLEURS files listed in {tsv} are not in {tar}: {missing[:3]}")
    return [rows[m] for m in order]


def build_vivos(dst: Path, tar: Path) -> list[dict]:
    """Extract the VIVOS test split into ``dst/vivos/``."""
    out_dir = Path(dst) / "vivos"
    out_dir.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    wavs: dict[str, bytes] = {}
    with tarfile.open(tar, "r:*") as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith("vivos/test/"):
                continue
            if member.name == "vivos/test/prompts.txt":
                for line in tf.extractfile(member).read().decode("utf-8").splitlines():
                    uid, _, text = line.strip().partition(" ")
                    if uid:
                        texts[uid] = text.strip().lower()
            elif member.name.endswith(".wav"):
                wavs[Path(member.name).stem] = tf.extractfile(member).read()
    rows = []
    for uid in sorted(texts):
        if uid not in wavs:
            raise FileNotFoundError(f"VIVOS prompt {uid} has no wav in {tar}")
        (out_dir / f"{uid}.wav").write_bytes(wavs[uid])
        rows.append(_row(f"vivos-{uid}", f"vivos/{uid}.wav", texts[uid], "vi", "vivos_vi", wavs[uid]))
    return rows


def write_dataset(dst: Path, rows: list[dict], sources: list[dict]) -> Path:
    dst = Path(dst)
    (dst / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (dst / SOURCE_NAME).write_text(json.dumps({"utterances": len(rows), "sources": sources},
                                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def download(url: str, path: Path) -> Path:
    """Fetch ``url`` to ``path`` unless it is already there."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def build_public(dst: Path, cache: Path) -> Path:
    """FLEURS vi + en test and VIVOS test, one manifest, categories fleurs_vi / fleurs_en / vivos_vi."""
    cache = Path(cache)
    rows: list[dict] = []
    for lang in ("vi", "en"):
        code = _FLEURS_LANG[lang]
        tsv = download(f"{FLEURS_BASE}/{code}/test.tsv", cache / f"fleurs_{code}_test.tsv")
        tar = download(f"{FLEURS_BASE}/{code}/audio/test.tar.gz", cache / f"fleurs_{code}_test.tar.gz")
        rows += build_fleurs(dst, lang, tsv, tar)
    rows += build_vivos(dst, download(VIVOS_URL, cache / "vivos.tar.gz"))
    return write_dataset(dst, rows, sources=[SOURCES["fleurs_vi"], SOURCES["fleurs_en"], SOURCES["vivos_vi"]])

import json
from pathlib import Path

from mlflow_registry.bench import manifest


def _write_fixture_set(root: Path, texts: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (utt_id, text) in enumerate(texts.items()):
        wav = root / f"{utt_id}.wav"
        wav.write_bytes(b"RIFF" + bytes([i]) * 8)
        lines.append(json.dumps({
            "id": utt_id, "file": wav.name, "text": text, "lang": "vi",
            "category": "vi_short", "en_words": [], "duration_s": 1.0,
        }, ensure_ascii=False))
    (root / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_load_returns_utterances_in_manifest_order(tmp_path):
    root = _write_fixture_set(tmp_path, {"00": "anh em", "01": "buổi sáng"})
    utts = manifest.load(root)
    assert [u.id for u in utts] == ["00", "01"]
    assert utts[1].text == "buổi sáng"
    assert utts[0].en_words == ()
    assert manifest.audio_bytes(root, utts[0]).startswith(b"RIFF")


def test_dataset_hash_is_stable_and_changes_with_audio_or_text(tmp_path):
    a = _write_fixture_set(tmp_path / "a", {"00": "anh em"})
    b = _write_fixture_set(tmp_path / "b", {"00": "anh em"})
    assert manifest.dataset_hash(a) == manifest.dataset_hash(b)
    assert manifest.dataset_hash(a).startswith("sha256:")

    (b / "00.wav").write_bytes(b"RIFF" + b"\xff" * 8)
    assert manifest.dataset_hash(a) != manifest.dataset_hash(b)

    c = _write_fixture_set(tmp_path / "c", {"00": "anh  em"})
    assert manifest.dataset_hash(a) != manifest.dataset_hash(c)


def test_checked_in_fixtures_load_and_match_spec():
    utts = manifest.load(manifest.FIXTURES_DIR)
    assert len(utts) == 14
    assert sum(1 for u in utts if u.lang == "vi") == 11
    assert sum(1 for u in utts if u.lang == "en") == 3
    assert sum(len(u.text.split()) for u in utts) == 37
    for u in utts:
        assert (manifest.FIXTURES_DIR / u.file).stat().st_size > 1000

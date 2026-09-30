import json

import pytest

from mlflow_registry.bench.tts import manifest


def test_load_reads_alt_texts_and_readings(sentences_dir):
    sents = manifest.load(sentences_dir)
    assert [s.id for s in sents] == ["v1", "m1", "n1", "e1"]
    n1 = sents[2]
    assert n1.readings == ("Con đếm từ 1 đến 10.", "Con đếm từ một đến mười.")
    assert sents[1].en_words == ("dog",)


def test_duplicate_ids_rejected(sentences_dir):
    path = sentences_dir / "manifest.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(
        {"id": "v1", "text": "x", "lang": "vi", "category": "vi_short"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        manifest.load(sentences_dir)


def test_reference_and_hash_cover_the_clip(sentences_dir):
    ref = manifest.reference(sentences_dir)
    assert ref.text == "câu tham chiếu" and ref.wav.startswith(b"RIFF")
    before = manifest.dataset_hash(sentences_dir)
    (sentences_dir / "reference.wav").write_bytes(ref.wav + b"\0\0")
    assert manifest.dataset_hash(sentences_dir) != before


def test_committed_sentence_set_is_valid():
    sents = manifest.load(manifest.SENTENCES_DIR)
    assert len(sents) >= 50
    assert {s.lang for s in sents} == {"vi", "en"}
    assert all(s.en_words for s in sents if s.category == "mix")
    assert manifest.reference(manifest.SENTENCES_DIR).text

import json

import numpy as np
import pytest

from mlflow_registry.bench.collect import ensure_run
from mlflow_registry.bench.tts import manifest, score, ttfb


def _clock(times):
    it = iter(times)
    return lambda: next(it)


def test_time_stream_first_chunk_and_total():
    chunks = [np.zeros(0), np.ones(100, dtype=np.float32), np.ones(300, dtype=np.float32)]
    t = ttfb.time_stream(iter(chunks), 1000, clock=_clock([10.0, 10.2, 10.9]))
    assert t.ttfb_s == pytest.approx(0.2) and t.total_s == pytest.approx(0.9)
    assert t.audio_s == pytest.approx(0.4) and t.chunks == 2       # the empty chunk is not audio


def test_time_stream_without_audio_raises():
    with pytest.raises(RuntimeError):
        ttfb.time_stream(iter([np.zeros(0)]), 1000, clock=_clock([0.0, 0.1]))


def _streamer(fail=()):
    def stream(s):
        if s.id in fail:
            raise ValueError("bad")
        yield np.ones(10, dtype=np.float32)
        yield np.ones(10, dtype=np.float32)
    return ttfb.Streamer("sysa", frozenset({"en"}), 100, True, stream, {"kind": "fake"})


def test_run_writes_records_per_pass(tmp_path, sentences_dir):
    run_dir = tmp_path / "run"
    ensure_run(run_dir, sentences_dir, hasher=manifest.dataset_hash)
    out = ttfb.run(_streamer(), sentences_dir, run_dir, passes=2, now=lambda: "t", log=lambda *a: None)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert out == run_dir / "ttfb" / "sysa.jsonl"
    assert [(r["sent"], r["pass"]) for r in rows] == [("e1", 1), ("e1", 2)]      # English sentences only
    assert rows[0]["chunks"] == 2 and rows[0]["audio_s"] == pytest.approx(0.2) and rows[0]["streaming"]


def test_run_records_errors_and_continues(tmp_path, sentences_dir):
    run_dir = tmp_path / "run"
    ensure_run(run_dir, sentences_dir, hasher=manifest.dataset_hash)
    calls = []

    def stream(sent):
        calls.append(sent.id)
        if sent.id == "v1" and len(calls) > 1:      # the warm-up call (v1) succeeds, the timed one fails
            raise ValueError("bad")
        yield np.ones(10, dtype=np.float32)

    streamer = ttfb.Streamer("sysa", frozenset({"vi"}), 100, True, stream)
    out = ttfb.run(streamer, sentences_dir, run_dir, passes=1, now=lambda: "t", log=lambda *a: None)
    rows = {r["sent"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows["v1"]["error"] == "ValueError: bad" and rows["m1"]["error"] is None


def test_run_refuses_other_dataset(tmp_path, sentences_dir):
    run_dir = tmp_path / "run"
    ensure_run(run_dir, sentences_dir, hasher=manifest.dataset_hash)
    (sentences_dir / "reference.json").write_text(json.dumps({"text": "khác"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pins"):
        ttfb.run(_streamer(), sentences_dir, run_dir, log=lambda *a: None)


def test_summary_medians_over_passes_then_sentences():
    recs = [
        {"sent": "a", "streaming": True, "ttfb_s": 0.1, "total_s": 1.0, "chunks": 4, "error": None},
        {"sent": "a", "streaming": True, "ttfb_s": 0.3, "total_s": 1.2, "chunks": 4, "error": None},
        {"sent": "b", "streaming": True, "ttfb_s": 0.5, "total_s": 2.0, "chunks": 8, "error": None},
        {"sent": "b", "streaming": True, "ttfb_s": None, "total_s": None, "chunks": None, "error": "x"},
    ]
    t = score.ttfb_summary(recs)
    assert t.ttfb_median_s == pytest.approx(0.35)          # median of per-sentence medians 0.2 and 0.5
    assert t.total_median_s == pytest.approx(1.55) and t.errors == 1 and t.streaming

import json
from pathlib import Path

import pytest

from mlflow_registry.bench import collect, manifest
from mlflow_registry.bench.backends import BackendError, Hypothesis


def _fixtures(tmp_path: Path) -> Path:
    root = tmp_path / "fx"
    root.mkdir()
    rows = [
        {"id": "00", "file": "00.wav", "text": "anh em", "lang": "vi", "category": "vi_short", "en_words": [], "duration_s": 1.0},
        {"id": "13", "file": "13.wav", "text": "good morning", "lang": "en", "category": "en_short", "en_words": ["good", "morning"], "duration_s": 0.5},
    ]
    for r in rows:
        (root / r["file"]).write_bytes(b"RIFF" + r["id"].encode())
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root


class FakeBackend:
    def __init__(self, name="whisper-large-v3", auto=True, fail_on=(), empty_on=()):
        self.name = name
        self.languages = frozenset({"vi", "en"})
        self.auto_detect = auto
        self.calls: list[tuple[bytes, str | None]] = []
        self._fail_on = set(fail_on)
        self._empty_on = set(empty_on)

    def meta(self):
        return {"kind": "fake"}

    def transcribe(self, wav, language):
        self.calls.append((wav, language))
        utt = wav[4:].decode()
        if utt in self._fail_on:
            raise BackendError("boom")
        text = "" if utt in self._empty_on else {"00": "anh em", "13": "good morning"}[utt]
        return Hypothesis(text=text, language=language, latency_s=0.1)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ensure_run_writes_meta_then_verifies_hash(tmp_path):
    fx = _fixtures(tmp_path)
    run = tmp_path / "run"
    meta = collect.ensure_run(run, fx, run_id="r1", git_sha="abc")
    assert meta["run_id"] == "r1" and meta["dataset_hash"] == manifest.dataset_hash(fx)
    assert json.loads((run / collect.RUN_META).read_text())["harness_git_sha"] == "abc"

    assert collect.ensure_run(run, fx)["run_id"] == "r1"   # second leg, same fixtures: fine

    (fx / "00.wav").write_bytes(b"RIFFdifferent")
    with pytest.raises(collect.RunMismatch):
        collect.ensure_run(run, fx)


def test_collect_runs_warmup_then_passes_and_conditions(tmp_path):
    fx = _fixtures(tmp_path)
    run = tmp_path / "run"
    b = FakeBackend()
    out = collect.collect(b, fx, run, passes=3, now=lambda: "2026-09-29T00:00:00Z", log=lambda *_: None)

    assert out == run / "whisper-large-v3.jsonl"
    rows = _read(out)
    # 2 utts x 3 passes x 2 conditions = 12 records; warm-up not recorded
    assert len(rows) == 12
    assert len(b.calls) == 13
    assert b.calls[0] == (b"RIFF00", "vi")                       # warm-up: first utt, hinted
    assert {r["condition"] for r in rows} == {"hinted", "auto"}
    assert {r["pass"] for r in rows} == {1, 2, 3}
    hinted_en = next(r for r in rows if r["utt"] == "13" and r["condition"] == "hinted")
    assert hinted_en["lang_hint"] == "en" and hinted_en["text"] == "good morning"
    auto = next(r for r in rows if r["condition"] == "auto")
    assert auto["lang_hint"] is None
    assert rows[0]["system"] == "whisper-large-v3"
    assert rows[0]["backend"] == {"kind": "fake"}
    assert rows[0]["ts"] == "2026-09-29T00:00:00Z"
    assert rows[0]["error"] is None and rows[0]["latency_s"] == pytest.approx(0.1)


def test_collect_skips_auto_for_single_language_backend(tmp_path):
    fx = _fixtures(tmp_path)
    b = FakeBackend(name="gipformer1.5-68m-rnnt", auto=False)
    rows = _read(collect.collect(b, fx, tmp_path / "run", passes=1, log=lambda *_: None))
    assert {r["condition"] for r in rows} == {"hinted"}
    assert len(rows) == 2


def test_collect_records_failures_and_empty_and_keeps_going(tmp_path):
    fx = _fixtures(tmp_path)
    b = FakeBackend(auto=False, fail_on={"00"}, empty_on={"13"})
    rows = _read(collect.collect(b, fx, tmp_path / "run", passes=1, log=lambda *_: None))
    by_utt = {r["utt"]: r for r in rows}
    assert by_utt["00"]["error"] == "BackendError: boom" and by_utt["00"]["text"] is None
    assert by_utt["13"]["error"] == "empty transcript" and by_utt["13"]["text"] == ""


def test_collect_overwrites_previous_file_for_same_system(tmp_path):
    fx = _fixtures(tmp_path)
    run = tmp_path / "run"
    collect.collect(FakeBackend(auto=False), fx, run, passes=1, log=lambda *_: None)
    rows = _read(collect.collect(FakeBackend(auto=False), fx, run, passes=1, log=lambda *_: None))
    assert len(rows) == 2


def test_records_carry_the_dataset_hash_of_the_fixtures_they_were_collected_on(tmp_path):
    fx = _fixtures(tmp_path)
    rows = _read(collect.collect(FakeBackend(auto=False), fx, tmp_path / "run", passes=1, log=lambda *_: None))
    assert {r["dataset_hash"] for r in rows} == {manifest.dataset_hash(fx)}


def test_concurrent_collect_really_overlaps_requests_and_keeps_record_order(tmp_path):
    import threading
    import time

    fx = _fixtures(tmp_path)

    class SlowBackend(FakeBackend):
        def __init__(self):
            super().__init__(auto=True, fail_on={"13"})
            self.lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0

        def transcribe(self, wav, language):
            with self.lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            time.sleep(0.05)
            try:
                return super().transcribe(wav, language)
            finally:
                with self.lock:
                    self.in_flight -= 1

    seq = _read(collect.collect(FakeBackend(auto=True, fail_on={"13"}), fx, tmp_path / "seq", passes=2,
                                log=lambda *_: None))
    slow = SlowBackend()
    par = _read(collect.collect(slow, fx, tmp_path / "par", passes=2, concurrency=4, log=lambda *_: None))

    assert slow.max_in_flight > 1
    key = lambda r: (r["condition"], r["pass"], r["utt"], r["text"], r["error"])  # noqa: E731
    assert [key(r) for r in par] == [key(r) for r in seq]


def test_retry_errors_re_requests_only_failed_records_and_keeps_the_rest(tmp_path):
    fx = _fixtures(tmp_path)
    run = tmp_path / "run"
    first = _read(collect.collect(FakeBackend(auto=True, fail_on={"13"}), fx, run, passes=1, log=lambda *_: None))
    assert sum(1 for r in first if r["error"]) == 2          # utt 13 in hinted and in auto

    healthy = FakeBackend(auto=True)
    out = collect.collect(healthy, fx, run, passes=1, retry_errors=True, now=lambda: "later",
                          log=lambda *_: None)
    again = _read(out)
    assert len(healthy.calls) == 2                            # only the two failed records, no warm-up
    assert [(r["condition"], r["utt"]) for r in again] == [(r["condition"], r["utt"]) for r in first]
    assert all(r["error"] is None for r in again)
    kept = [r for r in again if r["utt"] == "00"]
    assert all(r["ts"] != "later" for r in kept)              # successful records untouched
    assert {r["ts"] for r in again if r["utt"] == "13"} == {"later"}


def test_retry_errors_without_a_previous_file_is_an_error(tmp_path):
    fx = _fixtures(tmp_path)
    with pytest.raises(FileNotFoundError):
        collect.collect(FakeBackend(), fx, tmp_path / "run", retry_errors=True, log=lambda *_: None)

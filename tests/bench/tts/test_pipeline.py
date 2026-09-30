"""collect -> judge asr -> judge quality -> score on a tiny run with fake backends."""
import io
import json

import numpy as np
import pytest
import soundfile as sf

from mlflow_registry.bench.backends import BackendError, Hypothesis
from mlflow_registry.bench.metrics import aggregate
from mlflow_registry.bench.tts import collect, judge, manifest, score
from mlflow_registry.bench.tts.backends import Synthesis

from .conftest import make_wav

# What each fake system "says" (the fake judge returns this text for the system's audio).
SAYS = {
    "sysa": {"v1": "xin chào các bạn", "m1": "bé ơi dog nghĩa là con chó", "n1": "con đếm từ một đến mười",
             "e1": "good morning"},
    "sysb": {"v1": "xin chào", "m1": "bé ơi đóc nghĩa là con chó", "n1": "con đếm từ 1 đến 10"},
}


class FakeTts:
    def __init__(self, name, languages, fail=(), seconds=None):
        self.name, self.languages = name, frozenset(languages)
        self._fail, self._seconds = set(fail), seconds or {}
        self.calls = 0

    def meta(self):
        return {"kind": "fake"}

    def synthesize(self, text, language):
        self.calls += 1
        sid = next(s["id"] for s in _sentences() if s["text"] == text)
        if sid in self._fail:
            raise BackendError("boom")
        secs = self._seconds.get(sid, 1.0)
        # encode system and sentence in the tone so the fake judge can tell them apart
        return Synthesis(wav=make_wav(secs) + f"|{self.name}|{sid}".encode(), sample_rate=24000,
                         duration_s=secs, latency_s=0.1 * secs)


def _sentences():
    from .conftest import SENTENCES
    return SENTENCES


class FakeJudge:
    name = "fakejudge"

    def transcribe(self, wav, language):
        system, sid = wav.rsplit(b"|", 2)[1:]
        return Hypothesis(text=SAYS[system.decode()][sid.decode()], language=language, latency_s=0.01)


@pytest.fixture
def run(tmp_path, sentences_dir):
    run_dir = tmp_path / "run"
    a = FakeTts("sysa", {"vi", "en"}, seconds={"v1": 1.0})
    b = FakeTts("sysb", {"vi"}, fail={"v1"}, seconds={"m1": 9.0})
    for backend in (a, b):
        collect.collect(backend, sentences_dir, run_dir, passes=2, now=lambda: "t", log=lambda *a: None)
    judge.asr(FakeJudge(), run_dir, log=lambda *a: None)
    return run_dir, a, b


def test_collect_writes_pass1_audio_and_timed_passes(run):
    run_dir, a, b = run
    rows = [json.loads(line) for line in (run_dir / "sysa.jsonl").read_text().splitlines()]
    assert len(rows) == 8                                  # 4 sentences x 2 passes
    assert a.calls == 1 + 8                                # warm-up + passes
    assert all(r["audio"] for r in rows if r["pass"] == 1)
    assert not any(r["audio"] for r in rows if r["pass"] == 2)
    assert (run_dir / "audio" / "sysa" / "e1.wav").exists()
    brows = [json.loads(line) for line in (run_dir / "sysb.jsonl").read_text().splitlines()]
    assert {r["sent"] for r in brows} == {"v1", "m1", "n1"}   # English skipped for a vi-only system
    assert [r["error"] for r in brows if r["sent"] == "v1"] == ["BackendError: boom"] * 2


def test_short_audio_is_a_failure(tmp_path, sentences_dir):
    backend = FakeTts("tiny", {"en"}, seconds={"e1": 0.1})
    collect.collect(backend, sentences_dir, tmp_path / "r", passes=1, now=lambda: "t", log=lambda *a: None)
    row = json.loads((tmp_path / "r" / "tiny.jsonl").read_text())
    assert row["error"].startswith("audio shorter")


def test_asr_judge_marks_missing_audio(run):
    run_dir, _, _ = run
    rows = {r["sent"]: r for r in map(json.loads, (run_dir / "asr" / "fakejudge" / "sysb.jsonl").read_text().splitlines())}
    assert rows["v1"]["error"] == "no audio" and rows["m1"]["text"].startswith("bé ơi")


def test_quality_uses_injected_scorers(run, sentences_dir):
    run_dir, _, _ = run
    judge.quality(run_dir, sentences_dir, mos=lambda x: 3.5, embed=lambda x: np.array([1.0, 0.0]),
                  cloning=frozenset({"sysa"}), log=lambda *a: None)
    qa = [json.loads(line) for line in (run_dir / "quality" / "sysa.jsonl").read_text().splitlines()]
    qb = {r["sent"]: r for r in map(json.loads, (run_dir / "quality" / "sysb.jsonl").read_text().splitlines())}
    assert all(r["utmos"] == 3.5 and r["spk_sim"] == pytest.approx(1.0) for r in qa)
    assert qb["v1"]["utmos"] is None and qb["m1"]["spk_sim"] is None      # failed / not cloning


def test_load_16k_resamples():
    x = judge.load_16k(make_wav(1.0, sr=24_000))
    assert len(x) == 16_000 and x.dtype == np.float32


def test_score_counts_failures_as_deletions_and_accepts_alt_readings(run, sentences_dir):
    run_dir, _, _ = run
    sents = manifest.load(sentences_dir)
    results = {r.system: r for r in score.build_results(score.load_run(run_dir), sents)}
    a, b = results["sysa"], results["sysb"]
    assert score.wer(a.rows, ["fakejudge"]) == 0.0                   # numbers spelled out match alt reading
    b_rows = {r.sentence.id: r for r in b.rows}
    assert b_rows["v1"].error and b_rows["v1"].scored["fakejudge"].word_edits == 4   # all 4 words deleted
    assert b_rows["n1"].scored["fakejudge"].word_edits == 0          # digits match the primary reading
    assert b_rows["m1"].scored["fakejudge"].en_hits == 0             # "dog" became "đóc"
    agg = aggregate(r.scored["fakejudge"] for r in b.rows)
    assert agg.failures == 0 and agg.ref_words > 0                   # failures stay in the pooled WER
    assert b_rows["m1"].duration_flag == "9.0x longer"               # 9 s where the other system took 1 s


def test_duration_flag_threshold(run, sentences_dir):
    run_dir, _, _ = run
    results = score.build_results(score.load_run(run_dir), manifest.load(sentences_dir))
    flags = {(r.system, row.sentence.id): row.duration_flag for r in results for row in r.rows}
    assert flags[("sysa", "m1")] == "0.1x shorter"
    assert flags[("sysa", "v1")] is None                              # only system with audio: nothing to compare


def test_report_ranks_per_language_and_lists_winners(run, sentences_dir):
    run_dir, _, _ = run
    sents = manifest.load(sentences_dir)
    loaded = score.load_run(run_dir)
    results = score.build_results(loaded, sents)
    text = score.render_report(loaded, sents, results, "eval/runs/x")
    assert "## Vietnamese" in text and "## English: 1 sentences" in text
    english = text.split("## English")[1].split("##")[0]
    assert "sysa" in english and "sysb" not in english               # vi-only system not ranked in English
    assert "| mix | sysa |" in text                                   # sysa keeps "dog"
    assert "`eval/runs/x/audio/sysb/m1.wav`" in text


def test_score_refuses_mixed_datasets(run):
    run_dir, _, _ = run
    path = run_dir / "asr" / "fakejudge" / "sysa.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["dataset_hash"] = "sha256:other"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(score.DatasetMismatch):
        score.load_run(run_dir)


def test_mlflow_metric_dict(run, sentences_dir):
    from mlflow_registry.bench.tts import mlflow_log

    run_dir, _, _ = run
    results = {r.system: r for r in score.build_results(score.load_run(run_dir), manifest.load(sentences_dir))}
    ma, mb = (mlflow_log.metric_dict(results[s], ["fakejudge"]) for s in ("sysa", "sysb"))
    assert ma["vi.wer"] == 0.0 and ma["en.wer"] == 0.0 and ma["mix.en_recall"] == 1.0
    assert "en.wer" not in mb and mb["vi.failures"] == 1.0 and mb["failed_requests"] == 2.0
    assert "spk_sim" not in ma                       # fake systems are not cloning systems

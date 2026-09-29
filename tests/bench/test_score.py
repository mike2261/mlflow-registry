import json
from pathlib import Path

import pytest

from mlflow_registry.bench import score
from mlflow_registry.bench.manifest import Utterance

UTTS = [
    Utterance("00", "00.wav", "anh em", "vi", "vi_short", (), 1.0),
    Utterance("11", "11.wav", "cộng hoà xã hội chủ nghĩa việt nam", "vi", "vi_medium", (), 6.0),
    Utterance("13", "13.wav", "good morning", "en", "en_short", ("good", "morning"), 0.5),
]


def _rec(system, condition, pass_no, utt, text, latency=0.2, error=None):
    return {"system": system, "condition": condition, "pass": pass_no, "utt": utt.id,
            "lang_hint": None if condition == "auto" else utt.lang, "text": text,
            "language": None, "latency_s": None if error else latency, "error": error,
            "ts": "t", "backend": {"kind": "fake"}}


def _gipformer_records():
    # vi rows perfect; en row garbage. Three passes, latencies 0.1/0.3/0.2.
    rows = []
    for p, lat in zip((1, 2, 3), (0.1, 0.3, 0.2)):
        rows.append(_rec("gipformer1.5-68m-rnnt", "hinted", p, UTTS[0], "anh em", lat))
        rows.append(_rec("gipformer1.5-68m-rnnt", "hinted", p, UTTS[1], "cộng hoà xã hội chủ nghĩa việt nam", lat))
        rows.append(_rec("gipformer1.5-68m-rnnt", "hinted", p, UTTS[2], "gút mó ninh", lat))
    return rows


def _chirp_records():
    rows = []
    for p in (1, 2, 3):
        rows.append(_rec("chirp_3", "hinted", p, UTTS[0], "Anh em.", 0.9))
        rows.append(_rec("chirp_3", "hinted", p, UTTS[1], "cộng hòa xã hội chủ nghĩa Việt Nam", 1.1))
        rows.append(_rec("chirp_3", "hinted", p, UTTS[2], None, error="BackendError: 500"))
    rows.append(_rec("chirp_3", "auto", 1, UTTS[0], "anh em"))
    rows.append(_rec("chirp_3", "auto", 2, UTTS[0], "anh en"))      # non-deterministic
    rows.append(_rec("chirp_3", "auto", 3, UTTS[0], "anh em"))
    return rows


def _write_run(tmp_path: Path, with_chirp=True) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / score.RUN_META).write_text(json.dumps({
        "run_id": "r1", "dataset": "stt-fixtures", "dataset_hash": "sha256:00", "created": "t", "harness_git_sha": "abc"}))
    (run / "gipformer1.5-68m-rnnt.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in _gipformer_records()) + "\n")
    if with_chirp:
        (run / "chirp_3.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in _chirp_records()) + "\n")
    return run


def test_load_run_requires_baseline_unless_allowed(tmp_path):
    run = _write_run(tmp_path, with_chirp=False)
    with pytest.raises(score.MissingBaseline):
        score.load_run(run)
    meta, records = score.load_run(run, allow_missing_baseline=True)
    assert meta["run_id"] == "r1" and len(records) == 9


def test_build_cells_uses_pass_one_text_and_median_latency(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    cells = {(c.system, c.condition): c for c in score.build_cells(records, UTTS)}
    gip = cells[("gipformer1.5-68m-rnnt", "hinted")]
    assert [r.utt.id for r in gip.rows] == ["00", "11", "13"]
    assert gip.rows[0].latency_s == pytest.approx(0.2)         # median of 0.1, 0.3, 0.2
    assert gip.rows[0].hyp == "anh em"
    chirp_auto = cells[("chirp_3", "auto")]
    assert chirp_auto.rows[0].hyp == "anh em"                    # pass 1
    assert len(chirp_auto.rows) == 1                             # only utt 00 was run in auto


def test_in_domain_excludes_out_of_language_rows_but_all_keeps_them(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    gip = next(c for c in score.build_cells(records, UTTS) if c.system == "gipformer1.5-68m-rnnt")
    sl = score.slices(gip)
    assert sl["all"].n == 3 and sl["all"].ref_words == 2 + 8 + 2
    assert sl["all"].word_edits == 3                             # en row: 3 garbage words vs 2 ref -> S,S,I
    assert sl["in_domain"].n == 2 and sl["in_domain"].wer == 0.0
    assert sl["lang:en"].wer == pytest.approx(3 / 2)
    assert sl["cat:vi_medium"].n == 1
    assert sl["all"].en_recall == 0.0


def test_failures_and_case_punctuation_in_chirp_cell(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    chirp = next(c for c in score.build_cells(records, UTTS) if (c.system, c.condition) == ("chirp_3", "hinted"))
    sl = score.slices(chirp)
    assert sl["all"].failures == 1
    assert sl["all"].ref_words == 10                             # failed row excluded
    # "cộng hòa" vs reference "cộng hoà": different tone-mark placement -> 1 word edit
    assert sl["all"].word_edits == 1
    assert sl["all"].exact == 1


def test_nondeterministic_lists_variants(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    nd = score.nondeterministic(records)
    assert nd == [("chirp_3", "auto", "00", ["anh em", "anh en"])]


def test_cost_per_min_only_for_chirp():
    assert score.cost_per_min("chirp_3") == pytest.approx(0.016)
    assert score.cost_per_min("whisper-large-v3") is None


def test_render_report_has_required_sections_and_counts(tmp_path):
    meta, records = score.load_run(_write_run(tmp_path))
    cells = score.build_cells(records, UTTS)
    text = score.render_report(meta, UTTS, cells, score.nondeterministic(records))
    assert "# STT evaluation" in text
    assert "sha256:00" in text and score.NORMALIZER_VERSION in text
    assert "## Summary: hinted" in text and "## Summary: auto" in text
    assert "| chirp_3 |" in text and "| gipformer1.5-68m-rnnt |" in text
    assert "0/10" in text                                         # in-domain edits/words for gipformer
    assert "$0.016" in text
    assert "## Non-deterministic outputs" in text and "anh en" in text
    assert "## Detail: gipformer1.5-68m-rnnt (hinted)" in text
    assert "gút mó ninh" in text
    assert "cannot rank models" in text and "12 reference words" in text

    out = score.write_report(tmp_path / "run", text)
    assert out == tmp_path / "run" / "report.md" and out.read_text(encoding="utf-8") == text

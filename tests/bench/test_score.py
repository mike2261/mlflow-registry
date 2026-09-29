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
            "ts": "t", "backend": {"kind": "fake"}, "dataset_hash": "sha256:00"}


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


def _rewrite(run: Path, name: str, mutate) -> None:
    path = run / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [mutate(r) for r in rows]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


_META = {"run_id": "r", "dataset": "d", "dataset_hash": "sha256:00", "created": "t"}


def test_load_run_refuses_records_collected_on_other_fixtures(tmp_path):
    run = _write_run(tmp_path)
    _rewrite(run, "gipformer1.5-68m-rnnt.jsonl", lambda r: {**r, "dataset_hash": "sha256:ff"})
    with pytest.raises(score.DatasetMismatch, match="gipformer"):
        score.load_run(run)


def test_load_run_refuses_records_without_a_dataset_hash(tmp_path):
    run = _write_run(tmp_path)
    _rewrite(run, "chirp_3.jsonl", lambda r: {k: v for k, v in r.items() if k != "dataset_hash"})
    with pytest.raises(score.DatasetMismatch, match="chirp_3"):
        score.load_run(run)


def test_cell_counts_failed_requests_in_any_pass_and_knows_passes_and_backend(tmp_path):
    run = _write_run(tmp_path)
    # pass 2 of utt 00 fails; pass 1 (the scored text) succeeded
    _rewrite(run, "gipformer1.5-68m-rnnt.jsonl",
             lambda r: {**r, "text": None, "latency_s": None, "error": "BackendError: 500"}
             if (r["utt"], r["pass"]) == ("00", 2) else r)
    _, records = score.load_run(run)
    gip = next(c for c in score.build_cells(records, UTTS) if c.system == "gipformer1.5-68m-rnnt")
    assert gip.requests == 9 and gip.failed_requests == 1
    assert gip.passes == 3
    assert gip.backend == {"kind": "fake"}
    assert score.slices(gip)["all"].failures == 0          # utterance-level: pass 1 was fine
    text = score.render_report(_META, UTTS, [gip], [])
    row = next(line for line in text.splitlines() if line.startswith("| gipformer1.5-68m-rnnt |"))
    assert row.rstrip().endswith("| 1/9 |")                # failed requests is the last column


def test_report_exact_rate_counts_failures_in_the_denominator(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    chirp = next(c for c in score.build_cells(records, UTTS) if (c.system, c.condition) == ("chirp_3", "hinted"))
    text = score.render_report(_META, UTTS, [chirp], [])
    row = next(line for line in text.splitlines() if line.startswith("| chirp_3 |"))
    assert "33.3% (1/3)" in row                             # 1 exact of 3 utterances, one of which failed


def test_detail_table_shows_detected_language(tmp_path):
    run = _write_run(tmp_path)
    _rewrite(run, "chirp_3.jsonl", lambda r: {**r, "language": "ko"} if r["condition"] == "auto" else r)
    _, records = score.load_run(run)
    cells = score.build_cells(records, UTTS)
    chirp_auto = next(c for c in cells if (c.system, c.condition) == ("chirp_3", "auto"))
    assert chirp_auto.languages == {"00": "ko"}
    text = score.render_report(_META, UTTS, cells, [])
    assert "| Utt | Ref | Hyp | Lang | WER | Latency |" in text
    assert "| 00 | anh em | anh em | ko |" in text


# --- robo-be parity ------------------------------------------------------------------------

def _cell_from(system, condition, rows):
    from mlflow_registry.bench.metrics import scored
    return score.Cell(system, condition, [scored(u, h, e, lat) for u, h, e, lat in rows],
                      passes=1, requests=len(rows), failed_requests=sum(1 for _, _, e, _ in rows if e))


EN = Utterance("13", "13.wav", "good morning", "en", "en_short", ("good", "morning"), 0.5)
VI = Utterance("00", "00.wav", "anh em", "vi", "vi_short", (), 1.0)


def test_summary_has_mean_wer_cs_pass_and_p95_columns(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    cells = [c for c in score.build_cells(records, UTTS) if c.condition == "hinted"]
    text = score.render_report(_META, UTTS, cells, [])
    header = next(line for line in text.splitlines() if line.startswith("| System | In-domain WER"))
    for col in ("Mean WER", "CS pass", "Median latency", "p95"):
        assert col in header
    gip = next(line for line in text.splitlines() if line.startswith("| gipformer1.5-68m-rnnt |"))
    assert "66.7% (2/3)" in gip          # CS pass: two vi rows pass vacuously, "gút mó ninh" fails


def test_latency_table_has_robo_be_statistics(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    cells = [c for c in score.build_cells(records, UTTS) if c.condition == "hinted"]
    text = score.render_report(_META, UTTS, cells, [])
    assert "### Latency: hinted" in text
    assert "| System | mean | median | p95 | min | max |" in text


def test_category_winner_code_switch_dominates_then_mean_wer_then_tie():
    a = _cell_from("qwen3-asr-1.7b", "hinted", [(EN, "good morning", None, 0.1), (VI, "anh em", None, 0.1)])
    b = _cell_from("whisper-large-v3", "hinted", [(EN, "good mó ninh", None, 0.1), (VI, "anh em", None, 0.1)])
    c = _cell_from("gipformer1.5-68m-rnnt", "hinted", [(EN, "good morning", None, 0.1), (VI, "anh", None, 0.1)])
    winners = score.category_winners([a, b, c])
    # en_short: gipformer is out of domain (vi only) and excluded; qwen passes code-switch, whisper does not
    assert winners["en_short"] == ["qwen3-asr-1.7b"]
    # vi_short: qwen and whisper both 0 WER -> tie; gipformer 50 % loses
    assert winners["vi_short"] == ["qwen3-asr-1.7b", "whisper-large-v3"]


def test_category_winner_code_switch_beats_lower_wer():
    long_en = Utterance("17", "17.wav", "a rolling stone gathers no moss", "en", "en_medium",
                        ("rolling", "stone"), 2.0)
    passer = _cell_from("qwen3-asr-1.7b", "hinted", [(long_en, "a rolling stone gather know moss", None, 0.1)])
    lower_wer = _cell_from("whisper-large-v3", "hinted", [(long_en, "a rolling stun gathers no moss", None, 0.1)])
    assert score.category_winners([passer, lower_wer])["en_medium"] == ["qwen3-asr-1.7b"]


def test_report_lists_category_winners(tmp_path):
    _, records = score.load_run(_write_run(tmp_path))
    cells = [c for c in score.build_cells(records, UTTS) if c.condition == "hinted"]
    text = score.render_report(_META, UTTS, cells, [])
    assert "### Category winners: hinted" in text
    assert "| vi_short |" in text


def test_comparison_puts_datasets_side_by_side():
    clean = [_cell_from("qwen3-asr-1.7b", "hinted", [(VI, "anh em", None, 0.1), (EN, "good morning", None, 0.1)])]
    opus = [_cell_from("qwen3-asr-1.7b", "hinted", [(VI, "anh", None, 0.2), (EN, "good morning", None, 0.2)])]
    text = score.render_comparison([
        ("clean", {"dataset": "stt-fixtures", "dataset_hash": "sha256:aa"}, clean),
        ("opus24", {"dataset": "stt-fixtures-opus24", "dataset_hash": "sha256:bb"}, opus),
    ])
    assert "# STT evaluation: clean vs opus24" in text
    assert "## hinted" in text
    header = next(line for line in text.splitlines() if line.startswith("| System |"))
    assert "clean WER" in header and "opus24 WER" in header and "Δ WER" in header
    row = next(line for line in text.splitlines() if line.startswith("| qwen3-asr-1.7b |"))
    assert "0.0% (0/4)" in row and "25.0% (1/4)" in row and "+25.0 pp" in row

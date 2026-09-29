import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_stt  # noqa: E402

from mlflow_registry.bench import manifest  # noqa: E402
from mlflow_registry.bench.backends import Hypothesis  # noqa: E402


def test_parse_collect_serving_defaults():
    args = eval_stt.parse(["collect", "--backend", "serving", "--run", "eval/runs/x"])
    assert args.backend == "serving" and args.host == "localhost" and args.passes == 3
    assert args.models is None and args.fixtures == str(manifest.FIXTURES_DIR)


def test_parse_collect_chirp_defaults_location_us():
    args = eval_stt.parse(["collect", "--backend", "chirp", "--run", "r", "--project", "p"])
    assert args.location == "us" and args.project == "p"


def test_build_backends_serving_selects_models():
    args = eval_stt.parse(["collect", "--backend", "serving", "--run", "r",
                           "--models", "whisper-large-v3,gipformer1.5-68m-rnnt", "--host", "h"])
    backends = eval_stt.build_backends(args)
    assert [b.name for b in backends] == ["whisper-large-v3", "gipformer1.5-68m-rnnt"]
    assert backends[0].host == "h"


def test_build_backends_serving_defaults_to_all_five():
    args = eval_stt.parse(["collect", "--backend", "serving", "--run", "r"])
    assert [b.name for b in eval_stt.build_backends(args)] == list(eval_stt.STT_MODELS)


def test_score_end_to_end_on_fake_run(tmp_path):
    fixtures = manifest.FIXTURES_DIR
    run = tmp_path / "run"
    run.mkdir()
    utts = manifest.load(fixtures)
    (run / "run.json").write_text(json.dumps({
        "run_id": "r", "dataset": "stt-fixtures", "dataset_hash": manifest.dataset_hash(fixtures),
        "created": "t", "harness_git_sha": "abc"}))
    rows = [{"system": "chirp_3", "condition": "hinted", "pass": 1, "utt": u.id, "lang_hint": u.lang,
             "text": u.text, "language": u.lang, "latency_s": 0.5, "error": None, "ts": "t",
             "backend": {"kind": "chirp"}, "dataset_hash": manifest.dataset_hash(fixtures)} for u in utts]
    (run / "chirp_3.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    assert eval_stt.main(["score", "--run", str(run)]) == 0
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "| chirp_3 | 0.0% (0/37) |" in report


def test_collect_main_with_injected_backend(tmp_path, monkeypatch):
    class Fake:
        name = "whisper-large-v3"
        languages = frozenset({"vi", "en"})
        auto_detect = False

        def meta(self):
            return {"kind": "fake"}

        def transcribe(self, wav, language):
            return Hypothesis("anh em", language, 0.01)

    monkeypatch.setattr(eval_stt, "build_backends", lambda args: [Fake()])
    run = tmp_path / "run"
    assert eval_stt.main(["collect", "--backend", "serving", "--run", str(run), "--passes", "1"]) == 0
    assert (run / "whisper-large-v3.jsonl").exists() and (run / "run.json").exists()


def test_score_has_no_passes_flag_it_reads_passes_from_the_data():
    import pytest
    with pytest.raises(SystemExit):
        eval_stt.parse(["score", "--run", "r", "--passes", "3"])


def _fake_run(run: Path, fixtures: Path, text_for=lambda u: u.text) -> Path:
    run.mkdir(parents=True)
    h = manifest.dataset_hash(fixtures)
    (run / "run.json").write_text(json.dumps({
        "run_id": run.name, "dataset": fixtures.name, "dataset_hash": h, "created": "t"}))
    rows = [{"system": "chirp_3", "condition": "hinted", "pass": 1, "utt": u.id, "lang_hint": u.lang,
             "text": text_for(u), "language": u.lang, "latency_s": 0.5, "error": None, "ts": "t",
             "backend": {"kind": "chirp"}, "dataset_hash": h} for u in manifest.load(fixtures)]
    (run / "chirp_3.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return run


def test_opus_command_builds_the_opus_fixture_set(tmp_path):
    import pytest
    pytest.importorskip("opuslib")
    dst = tmp_path / "stt-fixtures-opus24"
    assert eval_stt.main(["opus", "--dst", str(dst)]) == 0
    assert len(manifest.load(dst)) == 14 and (dst / "source.json").exists()


def test_compare_command_writes_side_by_side_report(tmp_path):
    clean = _fake_run(tmp_path / "clean", manifest.FIXTURES_DIR)
    other = _fake_run(tmp_path / "opus", manifest.FIXTURES_DIR,
                      text_for=lambda u: "sai" if u.id == "00" else u.text)
    out = tmp_path / "compare.md"
    assert eval_stt.main(["compare", "--run", f"clean={clean}", "--run", f"opus24={other}",
                          "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "# STT evaluation: clean vs opus24" in text
    row = next(line for line in text.splitlines() if line.startswith("| chirp_3 |"))
    assert "0.0% (0/37)" in row and "5.4% (2/37)" in row

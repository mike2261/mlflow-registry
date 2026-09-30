import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_tts  # noqa: E402


def test_collect_defaults():
    args = eval_tts.parse(["collect", "--backend", "serving", "--run", "eval/runs/x"])
    assert args.passes == 3 and args.host == "localhost" and args.models is None
    assert Path(args.sentences).name == "tts-sentences"


def test_asr_rejects_unknown_judge(capsys):
    try:
        eval_tts.main(["asr", "--judge", "nope", "--run", "r"])
    except SystemExit as e:
        assert "unknown judge" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_names_split():
    assert eval_tts._names("a, b") == ["a", "b"] and eval_tts._names(None) is None

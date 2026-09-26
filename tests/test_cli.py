import json
import sys

import pytest

from mlflow_registry.cli import main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "register" in capsys.readouterr().out


def test_register_local_prints_version(model_name, weights_dir, capsys):
    rc = main(["register", model_name, f"local:{weights_dir}", "--tag", "license=MIT"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "1"


def test_register_pretrained_callable(model_name, capsys):
    rc = main(["register", model_name, "pretrained:tests._fake_train:load_model"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "1"


def test_register_missing_source_reports_error(model_name, tmp_path, capsys):
    rc = main(["register", model_name, f"local:{tmp_path / 'nope'}"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_promote_then_resolve(model_name, weights_dir, capsys):
    main(["register", model_name, f"local:{weights_dir}"])
    main(["register", model_name, f"local:{weights_dir}"])
    capsys.readouterr()

    assert main(["promote", model_name, "2"]) == 0
    assert main(["resolve", model_name]) == 0
    out = capsys.readouterr().out
    assert "runs:/" in out

    assert main(["promote", model_name, "1", "--alias", "staging"]) == 0
    assert main(["resolve", model_name, "--alias", "staging", "--version"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def test_download_prints_local_dir(model_name, weights_dir, tmp_path, capsys):
    main(["register", model_name, f"local:{weights_dir}"])
    main(["promote", model_name, "1"])
    capsys.readouterr()

    assert main(["download", model_name, "--dest", str(tmp_path / "dl")]) == 0
    local = capsys.readouterr().out.strip()
    assert (tmp_path / "dl").exists()
    assert local.startswith(str(tmp_path / "dl"))


def test_list_json_includes_tags_and_aliases(model_name, weights_dir, capsys):
    main(["register", model_name, f"local:{weights_dir}", "--tag", "task=stt"])
    main(["promote", model_name, "1"])
    capsys.readouterr()

    assert main(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    mine = [r for r in rows if r["name"] == model_name]
    assert mine == [{
        "name": model_name,
        "version": "1",
        "aliases": ["production"],
        "tags": {"source": f"local:{weights_dir}", "task": "stt"},
    }]


def test_register_pretrained_imports_from_cwd(model_name, monkeypatch, capsys):
    """The console script must import pretrained: targets relative to the cwd."""
    root = str(pytest.importorskip("tests").__path__[0]).rsplit("/tests", 1)[0]
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", root)])
    monkeypatch.delitem(sys.modules, "tests._fake_train", raising=False)
    monkeypatch.delitem(sys.modules, "tests", raising=False)
    monkeypatch.chdir(root)

    rc = main(["register", model_name, "pretrained:tests._fake_train:load_model"])
    assert rc == 0, capsys.readouterr().err

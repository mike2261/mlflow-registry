import sys
import types
from pathlib import Path

import pytest

from mlflow_registry import fetchers


# -- from_local -------------------------------------------------------------

def test_from_local_returns_existing_path(tmp_path):
    assert fetchers.from_local(str(tmp_path)) == str(tmp_path)


def test_from_local_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        fetchers.from_local(str(tmp_path / "missing"))


# -- from_pretrained --------------------------------------------------------

class _FakePretrained:
    """Anything with save_pretrained(dir), like a transformers model/tokenizer."""

    def __init__(self, filename: str):
        self.filename = filename

    def save_pretrained(self, out: str) -> None:
        Path(out, self.filename).write_text("saved")


def test_from_pretrained_saves_model_into_returned_dir():
    out = fetchers.from_pretrained(_FakePretrained("model.bin"))
    assert (Path(out) / "model.bin").read_text() == "saved"


def test_from_pretrained_saves_tokenizer_alongside_model():
    out = fetchers.from_pretrained(
        _FakePretrained("model.bin"), tokenizer=_FakePretrained("tokenizer.json")
    )
    assert (Path(out) / "model.bin").exists()
    assert (Path(out) / "tokenizer.json").exists()


# -- from_huggingface -------------------------------------------------------

@pytest.fixture
def fake_hf_hub(monkeypatch, tmp_path):
    """Stand-in for huggingface_hub so tests need neither the package nor network."""
    calls = []

    def snapshot_download(repo_id, revision=None, **kwargs):
        calls.append({"repo_id": repo_id, "revision": revision, **kwargs})
        snap = tmp_path / "models--org--model" / "snapshots" / "abc123def"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "weights.txt").write_text("hf weights")
        return str(snap)

    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


def test_from_huggingface_returns_snapshot_dir(fake_hf_hub, tmp_path):
    out = fetchers.from_huggingface("org/model")
    assert out.endswith("/snapshots/abc123def")


def test_from_huggingface_passes_repo_and_revision(fake_hf_hub):
    fetchers.from_huggingface("org/model", revision="v1.2")
    assert fake_hf_hub[0]["repo_id"] == "org/model"
    assert fake_hf_hub[0]["revision"] == "v1.2"


def test_from_huggingface_explains_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        fetchers.from_huggingface("org/model")


# -- hardening --------------------------------------------------------------

def test_from_local_rejects_a_file(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        fetchers.from_local(str(f))


def test_from_huggingface_forwards_download_options(fake_hf_hub):
    fetchers.from_huggingface(
        "org/model", token="tok", local_dir="/tmp/x", allow_patterns=["*.safetensors"]
    )
    call = fake_hf_hub[0]
    assert call["repo_id"] == "org/model"
    assert call["revision"] is None
    assert call["token"] == "tok"
    assert call["local_dir"] == "/tmp/x"
    assert call["allow_patterns"] == ["*.safetensors"]


class _BrokenPretrained:
    def save_pretrained(self, out: str) -> None:
        self.out = out
        raise RuntimeError("disk full")


def test_from_pretrained_cleans_up_temp_dir_on_failure():
    broken = _BrokenPretrained()
    with pytest.raises(RuntimeError):
        fetchers.from_pretrained(broken)
    assert not Path(broken.out).exists()


# -- fetch(spec) dispatcher -------------------------------------------------

def test_fetch_local_scheme(tmp_path):
    assert fetchers.fetch(f"local:{tmp_path}") == str(tmp_path)


def test_fetch_bare_path_is_local(tmp_path):
    assert fetchers.fetch(str(tmp_path)) == str(tmp_path)


def test_fetch_hf_returns_snapshot_dir(fake_hf_hub, tmp_path):
    out = fetchers.fetch("hf:org/model")
    assert out == str(tmp_path / "models--org--model" / "snapshots" / "abc123def")


def test_fetch_hf_scheme_with_revision(fake_hf_hub):
    fetchers.fetch("hf:org/model@v1.2")
    assert fake_hf_hub[0]["repo_id"] == "org/model"
    assert fake_hf_hub[0]["revision"] == "v1.2"


def test_fetch_hf_scheme_without_revision(fake_hf_hub):
    fetchers.fetch("hf:org/model")
    assert fake_hf_hub[0]["repo_id"] == "org/model"
    assert fake_hf_hub[0]["revision"] is None


def test_fetch_unknown_scheme_raises():
    with pytest.raises(ValueError, match="s3"):
        fetchers.fetch("s3://bucket/model")


# -- fetch_with_meta --------------------------------------------------------

def test_fetch_with_meta_hf_exposes_resolved_revision(fake_hf_hub):
    got = fetchers.fetch_with_meta("hf:org/model@main")
    assert got.path.endswith("/snapshots/abc123def")
    assert got.revision == "abc123def"
    assert got.source == "hf:org/model@main"


def test_fetch_with_meta_local_has_no_revision(tmp_path):
    got = fetchers.fetch_with_meta(str(tmp_path))
    assert got.path == str(tmp_path)
    assert got.revision is None


def test_fetch_forwards_hf_download_options(fake_hf_hub):
    fetchers.fetch("hf:org/model", allow_patterns=["*.safetensors"], ignore_patterns=["*.bin"])
    assert fake_hf_hub[0]["allow_patterns"] == ["*.safetensors"]
    assert fake_hf_hub[0]["ignore_patterns"] == ["*.bin"]


# -- pretrained: scheme -----------------------------------------------------

def test_fetch_pretrained_imports_callable_and_saves_model():
    out = fetchers.fetch("pretrained:tests._fake_train:load_model")
    assert (Path(out) / "model.bin").exists()


def test_fetch_pretrained_accepts_model_tokenizer_tuple():
    out = fetchers.fetch("pretrained:tests._fake_train:load_model_and_tokenizer")
    assert (Path(out) / "model.bin").exists()
    assert (Path(out) / "tokenizer.json").exists()


def test_fetch_pretrained_rejects_object_without_save_pretrained():
    with pytest.raises(TypeError, match="save_pretrained"):
        fetchers.fetch("pretrained:tests._fake_train:not_a_model")


def test_fetch_pretrained_requires_module_colon_callable():
    with pytest.raises(ValueError, match="module:callable"):
        fetchers.fetch("pretrained:tests._fake_train")


def test_fetch_with_meta_pretrained_keeps_spec_as_source():
    got = fetchers.fetch_with_meta("pretrained:tests._fake_train:load_model")
    assert got.source == "pretrained:tests._fake_train:load_model"
    assert got.revision is None

from pathlib import Path

import pytest

from mlflow.tracking import MlflowClient

from mlflow_registry.config import Config
from mlflow_registry.registry import ModelRegistry


def test_tracking_uri_comes_from_config():
    cfg = Config(
        tracking_uri="http://example.invalid:1234",
        s3_endpoint_url="http://example.invalid:9000",
        aws_access_key_id="k",
        aws_secret_access_key="s",
    )
    reg = ModelRegistry(cfg)
    assert reg.tracking_uri == "http://example.invalid:1234"


def test_register_raises_for_missing_source_dir(registry, model_name, tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.register(model_name, str(tmp_path / "nope"))


def test_register_returns_incrementing_version_strings(registry, model_name, weights_dir):
    v1 = registry.register(model_name, str(weights_dir))
    v2 = registry.register(model_name, str(weights_dir))
    assert isinstance(v1, str) and isinstance(v2, str)
    assert int(v2) == int(v1) + 1


def test_promote_moves_production_alias(registry, model_name, weights_dir):
    v_a = registry.register(model_name, str(weights_dir))
    v_b = registry.register(model_name, str(weights_dir))

    registry.promote(model_name, v_a)
    assert registry.get_current_version(model_name) == v_a

    registry.promote(model_name, v_b)
    assert registry.get_current_version(model_name) == v_b

    registry.promote(model_name, v_a)  # rollback
    assert registry.get_current_version(model_name) == v_a


def test_promote_supports_staging_alias(registry, model_name, weights_dir):
    v = registry.register(model_name, str(weights_dir))
    registry.promote(model_name, v, alias="staging")
    assert registry.get_current_version(model_name, alias="staging") == v


def test_resolve_returns_artifact_uri_of_aliased_version(registry, model_name, weights_dir):
    v = registry.register(model_name, str(weights_dir))
    registry.promote(model_name, v)
    uri = registry.resolve(model_name)
    assert uri.startswith("runs:/")
    assert uri.endswith("/model")


def test_download_pulls_weights_to_local_dir(registry, model_name, weights_dir, tmp_path):
    v = registry.register(model_name, str(weights_dir))
    registry.promote(model_name, v)

    dest = tmp_path / "dl"
    local = registry.download(model_name, dest_dir=str(dest))

    assert Path(local).is_dir()
    assert (Path(local) / "weights.txt").read_text() == "pretend weights"


def test_register_accepts_local_spec(registry, model_name, weights_dir):
    v = registry.register(model_name, f"local:{weights_dir}")
    assert int(v) >= 1


def test_register_accepts_hf_spec(registry, model_name, tmp_path, monkeypatch):
    import sys, types

    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "weights.txt").write_text("hf weights")
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = lambda repo_id, revision=None, **kw: str(snap)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)

    v = registry.register(model_name, "hf:org/model@main")
    registry.promote(model_name, v)
    local = registry.download(model_name, dest_dir=str(tmp_path / "dl"))
    assert (Path(local) / "weights.txt").read_text() == "hf weights"


def test_register_records_provenance_tags(registry, model_name, weights_dir):
    v = registry.register(
        model_name, f"local:{weights_dir}", tags={"license": "MIT", "task": "stt"}
    )
    tags = MlflowClient(tracking_uri=registry.tracking_uri).get_model_version(model_name, v).tags
    assert tags["license"] == "MIT"
    assert tags["task"] == "stt"
    assert tags["source"] == f"local:{weights_dir}"
    assert "hf_revision" not in tags


def test_register_hf_records_revision_tag(registry, model_name, tmp_path, monkeypatch):
    import sys, types

    snap = tmp_path / "models--org--model" / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "weights.txt").write_text("hf weights")
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = lambda repo_id, revision=None, **kw: str(snap)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)

    v = registry.register(model_name, "hf:org/model")
    tags = MlflowClient(tracking_uri=registry.tracking_uri).get_model_version(model_name, v).tags
    assert tags["source"] == "hf:org/model"
    assert tags["hf_revision"] == "deadbeef"


def test_register_large_file_goes_through_proxy(registry, model_name, tmp_path, monkeypatch):
    """Files above the multipart threshold must not use presigned minio URLs.

    Lower the threshold so a 3 MB file triggers the multipart path.
    """
    monkeypatch.setenv("MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE", str(1_000_000))
    monkeypatch.delenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", raising=False)
    big = tmp_path / "big"
    big.mkdir()
    (big / "weights.bin").write_bytes(b"\0" * 3_000_000)

    v = registry.register(model_name, str(big))
    registry.promote(model_name, v)
    local = registry.download(model_name, dest_dir=str(tmp_path / "dl"))
    assert (Path(local) / "weights.bin").stat().st_size == 3_000_000


def test_paginate_drains_every_page():
    from mlflow.store.entities import PagedList

    pages = {None: PagedList(["a", "b"], "p2"), "p2": PagedList(["c"], "p3"), "p3": PagedList(["d"], None)}
    seen_kwargs = []

    def search(page_token=None, **kwargs):
        seen_kwargs.append(kwargs)
        return pages[page_token]

    assert ModelRegistry._paginate(search, filter_string="x") == ["a", "b", "c", "d"]
    assert seen_kwargs == [{"filter_string": "x"}] * 3

from pathlib import Path

from mlflow_registry.serving import weights


def test_ensure_weights_downloads_once_and_marks_complete(tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_download(artifact_uri: str, dst_path: str) -> str:
        calls.append((artifact_uri, dst_path))
        Path(dst_path, "config.json").write_text("{}")
        return dst_path

    first = weights.ensure_weights("whisper-large-v3", "1", tmp_path, download=fake_download)
    second = weights.ensure_weights("whisper-large-v3", "1", tmp_path, download=fake_download)

    assert first == second == tmp_path / "whisper-large-v3" / "1"
    assert (first / "config.json").exists()
    assert calls == [("models:/whisper-large-v3/1", str(first))]


def test_ensure_weights_redownloads_when_previous_attempt_was_partial(tmp_path):
    target = tmp_path / "kokoro-82m" / "1"
    target.mkdir(parents=True)
    (target / "half.bin").write_bytes(b"\x00")  # no marker -> treated as partial

    def fake_download(artifact_uri: str, dst_path: str) -> str:
        Path(dst_path, "full.bin").write_bytes(b"\x01")
        return dst_path

    out = weights.ensure_weights("kokoro-82m", "1", tmp_path, download=fake_download)
    assert (out / "full.bin").exists()
    assert not (out / "half.bin").exists(), "partial download must be wiped first"
    assert (out / weights.MARKER).exists()


def test_ensure_weights_wraps_target_dir_not_parent(tmp_path):
    """download_artifacts drops files straight into dst_path; the wrapper must not nest."""
    def fake_download(artifact_uri: str, dst_path: str) -> str:
        Path(dst_path, "a.txt").write_text("a")
        return dst_path

    out = weights.ensure_weights("m", "3", tmp_path, download=fake_download)
    assert sorted(p.name for p in out.iterdir()) == sorted(["a.txt", weights.MARKER])


def test_cache_root_defaults_to_model_cache_env_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_CACHE", str(tmp_path / "cache"))

    def fake_download(artifact_uri: str, dst_path: str) -> str:
        return dst_path

    out = weights.ensure_weights("m", "1", download=fake_download)
    assert out == tmp_path / "cache" / "m" / "1"

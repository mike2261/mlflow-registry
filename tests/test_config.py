import pytest

from mlflow_registry.config import Config, load_config


def test_load_config_needs_only_tracking_uri(monkeypatch, tmp_path):
    for key in ("MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid:5000")
    empty_dotenv = tmp_path / ".env"
    empty_dotenv.write_text("")

    cfg = load_config(dotenv_path=str(empty_dotenv))

    assert cfg.tracking_uri == "http://example.invalid:5000"
    assert cfg.s3_endpoint_url == ""
    assert cfg.aws_access_key_id == ""
    assert cfg.aws_secret_access_key == ""


def test_load_config_still_requires_tracking_uri(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    empty_dotenv = tmp_path / ".env"
    empty_dotenv.write_text("")

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        load_config(dotenv_path=str(empty_dotenv))


def test_config_dataclass_defaults():
    cfg = Config(tracking_uri="http://x")
    assert cfg.s3_endpoint_url == ""

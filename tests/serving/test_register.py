"""Live-stack test: wrap a throwaway raw-weights model and serve it through the registry."""

import base64
import io

import mlflow.pyfunc
import numpy as np
import pandas as pd
import soundfile as sf
from mlflow.tracking import MlflowClient

from mlflow_registry.serving.catalog import ServingSpec
from mlflow_registry.serving.register import register_serving


def _wav_b64(n: int, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def test_register_serving_creates_v2_alias_and_serves_from_registry(registry, model_name, weights_dir, tmp_path, monkeypatch):
    v1 = registry.register(model_name, str(weights_dir))
    spec = ServingSpec(model_name, "stt", "tests.serving._fakes:FakeStt", port=5999)

    v2 = register_serving(spec, tracking_uri=registry.tracking_uri)

    assert int(v2) == int(v1) + 1
    client = MlflowClient(tracking_uri=registry.tracking_uri)
    mv = client.get_model_version_by_alias(model_name, "serving")
    assert mv.version == v2
    assert mv.tags["flavor"] == "pyfunc"
    assert mv.tags["wraps_version"] == v1
    assert mv.tags["task"] == "stt"

    # A second registration picks the raw weights again, not the pyfunc.
    v3 = register_serving(spec, tracking_uri=registry.tracking_uri)
    assert client.get_model_version(model_name, v3).tags["wraps_version"] == v1
    assert client.get_model_version_by_alias(model_name, "serving").version == v3

    # Loading through the alias downloads the weights into MODEL_CACHE and predicts.
    monkeypatch.setenv("MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@serving")
    out = loaded.predict(pd.DataFrame({"audio_b64": [_wav_b64(16_000, 16_000)]}))
    assert out.iloc[0]["text"] == "16000 samples from pretend weights"
    assert (tmp_path / "cache" / model_name / v1 / "weights.txt").exists()

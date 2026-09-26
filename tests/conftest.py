import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import uuid

import pytest

from mlflow_registry.config import load_config
from mlflow_registry.registry import ModelRegistry


@pytest.fixture(scope="session")
def registry() -> ModelRegistry:
    """A registry talking to the live docker-compose stack (localhost:5000)."""
    return ModelRegistry(load_config())


@pytest.fixture
def model_name(registry: ModelRegistry):
    """Unique per test so runs never collide on the shared MLflow server.

    Deleted afterwards so the server does not accumulate throwaway models.
    """
    name = f"test-{uuid.uuid4().hex[:8]}"
    yield name
    _delete_model(registry, name)


def _delete_model(registry: ModelRegistry, name: str) -> None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=registry.tracking_uri)
    try:
        client.delete_registered_model(name)
    except MlflowException:
        pass  # test never registered anything
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        client.delete_experiment(exp.experiment_id)


@pytest.fixture
def weights_dir(tmp_path):
    d = tmp_path / "weights"
    d.mkdir()
    (d / "weights.txt").write_text("pretend weights")
    return d

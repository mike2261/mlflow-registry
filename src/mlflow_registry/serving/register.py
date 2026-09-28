"""Log a pyfunc wrapper as a new version of a registered model and alias it ``serving``.

The wrapper points at an existing raw-weights version; nothing large is
uploaded. After this, ``mlflow models serve -m models:/<name>@serving`` works
from any environment that has the model's runtime installed.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_registry.serving.catalog import CONTAINER_PORT, SERVING_ALIAS, ServingSpec

PYFUNC_TAG = "flavor"
PYFUNC_TAG_VALUE = "pyfunc"
WRAPS_TAG = "wraps_version"


def _quiet_env() -> None:
    # The serving images install the package themselves (--env-manager local), so the
    # pyfunc's own environment files are noise; keep the artifact to a few kilobytes.
    os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")
    os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def latest_weights_version(client: MlflowClient, name: str) -> str:
    """Highest version that is raw weights (not a pyfunc wrapper)."""
    versions = client.search_model_versions(f"name='{name}'")
    raw = [v for v in versions if v.tags.get(PYFUNC_TAG) != PYFUNC_TAG_VALUE]
    if not raw:
        raise LookupError(f"{name}: no raw-weights version to wrap")
    return str(max(int(v.version) for v in raw))


def register_serving(
    spec: ServingSpec,
    weights_version: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Create the pyfunc version for ``spec`` and point ``@serving`` at it. Returns the version."""
    _quiet_env()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    weights_version = weights_version or latest_weights_version(client, spec.name)
    client.get_model_version(spec.name, weights_version)  # fail early if it does not exist

    experiment = client.get_experiment_by_name(spec.name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(spec.name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"serving-{spec.name}"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=spec.make_wrapper(weights_version),
            signature=spec.signature,
            pip_requirements=["mlflow", "mlflow-registry"],
            registered_model_name=spec.name,
            metadata={"task": spec.task, "wraps": f"models:/{spec.name}/{weights_version}"},
        )
    version = str(info.registered_model_version)

    tags = {
        PYFUNC_TAG: PYFUNC_TAG_VALUE,
        WRAPS_TAG: weights_version,
        "task": spec.task,
        "wrapper": spec.wrapper,
        "serving_port": str(spec.port),
        "container_port": str(CONTAINER_PORT),
    }
    for key, value in tags.items():
        client.set_model_version_tag(spec.name, version, key, value)
    client.set_registered_model_alias(spec.name, SERVING_ALIAS, version)
    return version

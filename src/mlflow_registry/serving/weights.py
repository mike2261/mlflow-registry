"""Fetch a registered model's raw weights once into a local cache directory.

The pyfunc wrapper stores only ``(name, version)``. At load time it calls
``ensure_weights`` which downloads ``models:/<name>/<version>`` into
``<cache_root>/<name>/<version>/`` and drops a marker file when the copy is
complete. A directory without the marker is treated as a partial download and
wiped, so a container killed mid-download recovers on restart.
"""

import os
import shutil
from collections.abc import Callable
from pathlib import Path

MARKER = ".mlflow-registry-complete"
CACHE_ENV = "MODEL_CACHE"
DEFAULT_CACHE_ROOT = "/models"

Downloader = Callable[[str, str], str]  # (artifact_uri, dst_path) -> local path


def _mlflow_download(artifact_uri: str, dst_path: str) -> str:
    from mlflow.artifacts import download_artifacts

    return download_artifacts(artifact_uri=artifact_uri, dst_path=dst_path)


def default_cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV, DEFAULT_CACHE_ROOT))


def ensure_weights(
    name: str,
    version: str,
    cache_root: str | os.PathLike[str] | None = None,
    download: Downloader = _mlflow_download,
) -> Path:
    """Return the local directory holding ``models:/<name>/<version>``.

    Downloads on first call (or after an incomplete attempt); afterwards the
    cached copy is returned immediately. ``cache_root`` defaults to
    ``$MODEL_CACHE`` (``/models`` inside the serving images). ``download`` is
    injectable for tests.
    """
    root = Path(cache_root) if cache_root is not None else default_cache_root()
    target = root / name / str(version)
    if (target / MARKER).exists():
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    download(f"models:/{name}/{version}", str(target))
    (target / MARKER).touch()
    return target

"""Model registry client: register weights, promote by alias, resolve/download.

Consumers address a model by ``name@alias`` (e.g. ``stt-model@production``),
never by version number. Moving the alias is the deploy; moving it back is
the rollback.
"""

import os
from pathlib import Path
from typing import Any, Literal

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from mlflow_registry import fetchers
from mlflow_registry.config import Config, load_config

type Alias = Literal["production", "staging"]
DEFAULT_ALIAS: Alias = "production"

_ARTIFACT_PATH = "model"


def _prefer_proxied_transfers() -> None:
    """Route large-file transfers through the MLflow server, not presigned URLs.

    With ``--serve-artifacts`` the server hands out presigned URLs on the
    docker-internal MinIO host for multipart uploads/downloads. From outside the
    compose network that host does not resolve, so uploads of files above
    ``MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE`` (500 MB) fail and downloads
    hang. Defaults respect an explicit opt-in by the caller.
    """
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")


class ModelRegistry:
    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self._client = MlflowClient(tracking_uri=self._config.tracking_uri)

    @property
    def tracking_uri(self) -> str:
        return self._config.tracking_uri

    # -- write side ---------------------------------------------------------

    def register(
        self,
        model_name: str,
        source: str,
        tags: dict[str, str] | None = None,
        **fetch_options: Any,
    ) -> str:
        """Upload the weights at ``source`` as a new version of ``model_name``.

        ``source`` is a fetcher spec (see ``mlflow_registry.fetchers.fetch``):
        a local directory path, ``local:/path``, or ``hf:org/model[@revision]``.
        ``fetch_options`` go to the fetcher (e.g. ``allow_patterns`` for HF).

        The version is tagged with ``source`` and, for HF, ``hf_revision``,
        plus any caller ``tags``. Returns the new version number as a string.
        """
        fetched = fetchers.fetch_with_meta(source, **fetch_options)
        source_dir = Path(fetched.path)
        _prefer_proxied_transfers()

        experiment_id = self._experiment_id(model_name)
        run = self._client.create_run(
            experiment_id, run_name=f"register-{model_name}"
        )
        run_id = run.info.run_id
        try:
            self._client.log_artifacts(run_id, str(source_dir), _ARTIFACT_PATH)
            self._client.set_terminated(run_id, status="FINISHED")
        except Exception:
            self._client.set_terminated(run_id, status="FAILED")
            raise

        self._ensure_registered_model(model_name)
        version = self._client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/{_ARTIFACT_PATH}",
            run_id=run_id,
        )

        all_tags = {"source": fetched.source, **(tags or {})}
        if fetched.revision:
            all_tags["hf_revision"] = fetched.revision
        for key, value in all_tags.items():
            self._client.set_model_version_tag(model_name, version.version, key, value)
        return str(version.version)

    def promote(
        self, model_name: str, version: str, alias: Alias = DEFAULT_ALIAS
    ) -> None:
        """Point ``alias`` at ``version``. Same call performs a rollback."""
        self._client.set_registered_model_alias(model_name, alias, version)

    # -- read side ----------------------------------------------------------

    def get_current_version(
        self, model_name: str, alias: Alias = DEFAULT_ALIAS
    ) -> str:
        return str(self._version_by_alias(model_name, alias).version)

    def resolve(self, model_name: str, alias: Alias = DEFAULT_ALIAS) -> str:
        """Artifact URI (``runs:/<run_id>/model``) behind ``model_name@alias``."""
        return self._version_by_alias(model_name, alias).source

    def download(
        self,
        model_name: str,
        alias: Alias = DEFAULT_ALIAS,
        dest_dir: str | None = None,
    ) -> str:
        """Fetch the weights behind ``model_name@alias`` and return the local dir."""
        _prefer_proxied_transfers()
        return download_artifacts(
            artifact_uri=self.resolve(model_name, alias),
            dst_path=dest_dir,
            tracking_uri=self.tracking_uri,
        )

    def list_versions(self) -> list[dict[str, Any]]:
        """Every version of every registered model, with aliases and tags."""
        rows: list[dict[str, Any]] = []
        for model in self._paginate(self._client.search_registered_models):
            by_version: dict[str, list[str]] = {}
            for alias, version in (model.aliases or {}).items():
                by_version.setdefault(str(version), []).append(alias)
            versions = self._paginate(
                self._client.search_model_versions, filter_string=f"name='{model.name}'"
            )
            for v in versions:
                rows.append({
                    "name": model.name,
                    "version": str(v.version),
                    "aliases": sorted(by_version.get(str(v.version), [])),
                    "tags": dict(v.tags),
                })
        rows.sort(key=lambda r: (r["name"], int(r["version"])))
        return rows

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _paginate(search: Any, **kwargs: Any) -> list[Any]:
        """Drain an MLflow paged search (100 items per page by default)."""
        items: list[Any] = []
        token = None
        while True:
            page = search(page_token=token, **kwargs)
            items.extend(page)
            token = page.token
            if not token:
                return items

    def _version_by_alias(self, model_name: str, alias: str):
        return self._client.get_model_version_by_alias(model_name, alias)

    def _experiment_id(self, model_name: str) -> str:
        exp = self._client.get_experiment_by_name(model_name)
        if exp is not None:
            return exp.experiment_id
        return self._client.create_experiment(model_name)

    def _ensure_registered_model(self, model_name: str) -> None:
        try:
            self._client.create_registered_model(model_name)
        except RestException as e:
            if e.error_code != "RESOURCE_ALREADY_EXISTS":
                raise


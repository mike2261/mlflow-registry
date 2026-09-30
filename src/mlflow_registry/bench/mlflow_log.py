"""Log one MLflow run per (system, condition) so results are queryable next to the registry."""
from __future__ import annotations

from pathlib import Path

import mlflow

from mlflow_registry.bench.normalize import NORMALIZER_VERSION
from mlflow_registry.bench.score import Cell, cost_per_min, slices

EXPERIMENT = "eval-stt"

_AGG_FIELDS = ("wer", "cer", "wer_mean", "exact_rate", "en_recall", "cs_pass_rate", "rtf",
               "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_min_s", "latency_max_s",
               "failures", "n", "ref_words", "word_edits")


def metric_dict(cell: Cell) -> dict[str, float]:
    out: dict[str, float] = {}
    sl = slices(cell)
    nn = sl["in_domain_no_numbers"]
    for field_name in ("wer", "cer", "ref_words", "n"):
        value = getattr(nn, field_name)
        if value is not None:
            out[f"in_domain_no_numbers.{field_name}"] = float(value)
    for scope in ("in_domain", "all"):
        for field in _AGG_FIELDS:
            value = getattr(sl[scope], field)
            if value is not None:
                out[f"{scope}.{field}"] = float(value)
    for key, agg in sl.items():
        if key.startswith("lang:") and agg.wer is not None:
            out[f"wer_{key.split(':', 1)[1]}"] = float(agg.wer)
    cost = cost_per_min(cell.system)
    if cost is not None:
        out["cost_usd_per_min"] = cost
    out["requests"] = float(cell.requests)
    out["failed_requests"] = float(cell.failed_requests)
    return out


def _backend_params(backend: dict) -> dict[str, str]:
    return {f"backend.{k}": ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
            for k, v in backend.items()}


def log_run(meta: dict, cell: Cell, run_dir: Path, tracking_uri: str | None = None,
            experiment: str = EXPERIMENT) -> str:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    run_dir = Path(run_dir)
    with mlflow.start_run(run_name=f"{cell.system}/{cell.condition}") as run:
        mlflow.set_tags({"run_id": meta["run_id"], "system": cell.system, "condition": cell.condition})
        mlflow.log_params({
            "system": cell.system,
            "condition": cell.condition,
            "dataset": meta["dataset"],
            "dataset_hash": meta["dataset_hash"],
            "normalizer_version": NORMALIZER_VERSION,
            "passes": cell.passes,
            "harness_git_sha": meta.get("harness_git_sha") or "unknown",
            **_backend_params(cell.backend),
        })
        mlflow.log_metrics(metric_dict(cell))
        for name in (f"{cell.system}.jsonl", "report.md"):
            path = run_dir / name
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id


def log_all(meta: dict, cells: list[Cell], run_dir: Path, tracking_uri: str | None = None,
            experiment: str = EXPERIMENT) -> list[str]:
    return [log_run(meta, c, run_dir, tracking_uri, experiment) for c in cells]

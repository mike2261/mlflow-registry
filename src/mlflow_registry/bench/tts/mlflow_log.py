"""Log one MLflow run per TTS system to the ``eval-tts`` experiment. Audio stays in the run dir."""
from __future__ import annotations

from pathlib import Path

import mlflow

from mlflow_registry.bench.normalize import NORMALIZER_VERSION
from mlflow_registry.bench.tts.score import (
    CLONING,
    Result,
    Run,
    agg,
    cost_per_1k,
    en_recall,
    mean_of,
    rows_for,
    speaks,
    ttfb_summary,
    wer,
)

EXPERIMENT = "eval-tts"


def metric_dict(res: Result, judges: list[str], ttfb_records: list[dict] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}

    def put(key: str, value: float | None) -> None:
        if value is not None:
            out[key] = float(value)

    for lang in ("vi", "en"):
        rows = rows_for(res, lang)
        if not speaks(res, lang):
            continue
        put(f"{lang}.wer", wer(rows, judges))
        for j in judges:
            put(f"{lang}.wer.{j}", agg(rows, j).wer)
        put(f"{lang}.cer", mean_of([agg(rows, j).cer for j in judges]))
        put(f"{lang}.utmos", mean_of([r.utmos for r in rows]))
        put(f"{lang}.failures", sum(1 for r in rows if r.error))
    mix = rows_for(res, category="mix")
    if mix:
        put("mix.en_recall", en_recall(mix, judges))
        put("mix.wer", wer(mix, judges))
    if res.system in CLONING:
        put("spk_sim", mean_of([r.spk_sim for r in res.rows]))
    put("duration_flags", sum(1 for r in res.rows if r.duration_flag))
    if judges:
        a = agg(res.rows, judges[0])
        for f in ("latency_mean_s", "latency_median_s", "latency_p95_s", "latency_min_s", "latency_max_s", "rtf"):
            put(f, getattr(a, f))
    put("cost_usd_per_1k_sentences", cost_per_1k(res.system, [r.sentence for r in res.rows]))
    if ttfb_records:
        t = ttfb_summary(ttfb_records)
        put("ttfb_median_s", t.ttfb_median_s)
        put("ttfb_p95_s", t.ttfb_p95_s)
        put("ttfb_stream_total_median_s", t.total_median_s)
    put("requests", res.requests)
    put("failed_requests", res.failed_requests)
    return out


def log_all(run: Run, results: list[Result], run_dir: Path, tracking_uri: str | None = None,
            experiment: str = EXPERIMENT) -> list[str]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    run_dir = Path(run_dir)
    ids = []
    for res in results:
        with mlflow.start_run(run_name=res.system) as mr:
            mlflow.set_tags({"run_id": run.meta["run_id"], "system": res.system})
            mlflow.log_params({
                "system": res.system,
                "dataset": run.meta["dataset"],
                "dataset_hash": run.meta["dataset_hash"],
                "normalizer_version": NORMALIZER_VERSION,
                "judges": ",".join(run.judges),
                "passes": res.passes,
                "harness_git_sha": run.meta.get("harness_git_sha") or "unknown",
                **{f"backend.{k}": str(v) for k, v in res.backend.items()},
            })
            mlflow.log_metrics(metric_dict(res, run.judges, run.ttfb.get(res.system)))
            for path in [run_dir / f"{res.system}.jsonl", run_dir / "report.md",
                         run_dir / "quality" / f"{res.system}.jsonl", run_dir / "ttfb" / f"{res.system}.jsonl",
                         *sorted((run_dir / "asr").glob(f"*/{res.system}.jsonl"))]:
                if path.exists():
                    sub = str(path.parent.relative_to(run_dir)) if path.parent != run_dir else None
                    mlflow.log_artifact(str(path), artifact_path=sub)
            ids.append(mr.info.run_id)
    return ids

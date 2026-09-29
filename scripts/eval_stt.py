"""Collect STT hypotheses from the serving containers or Google Chirp 3, then score them.

    # Google leg (laptop, uses Application Default Credentials; project from gcloud config)
    uv run python scripts/eval_stt.py collect --backend chirp --run eval/runs/2026-09-29-fixtures

    # self-hosted leg (devserver, models started with scripts/serve.sh up ...)
    uv run python scripts/eval_stt.py collect --backend serving --run eval/runs/2026-09-29-fixtures

    # score (needs both legs in the run dir; --mlflow logs to MLFLOW_TRACKING_URI from .env)
    uv run python scripts/eval_stt.py score --run eval/runs/2026-09-29-fixtures --mlflow

See docs/superpowers/specs/2026-09-29-stt-eval-design.md.
"""
from __future__ import annotations

import argparse
import configparser
import os
import sys
from pathlib import Path

from mlflow_registry.bench import collect, datasets, manifest, score
from mlflow_registry.bench.backends import STT_MODELS, Backend, ChirpBackend, ServingBackend


def _gcloud_project() -> str | None:
    """$GOOGLE_CLOUD_PROJECT, else the active gcloud configuration's core/project."""
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ["GOOGLE_CLOUD_PROJECT"]
    config_dir = Path(os.environ.get("CLOUDSDK_CONFIG", Path.home() / ".config" / "gcloud"))
    try:
        active = (config_dir / "active_config").read_text().strip() or "default"
        parser = configparser.ConfigParser()
        parser.read(config_dir / "configurations" / f"config_{active}")
        return parser.get("core", "project", fallback=None)
    except OSError:
        return None


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="call one backend over the manifest and write JSONL")
    c.add_argument("--backend", choices=["serving", "chirp"], required=True)
    c.add_argument("--run", required=True, help="run directory, e.g. eval/runs/2026-09-29-fixtures")
    c.add_argument("--fixtures", default=str(manifest.FIXTURES_DIR))
    c.add_argument("--passes", type=int, default=3)
    c.add_argument("--concurrency", type=int, default=1,
                   help="parallel requests; keep 1 for self-hosted models so latency is not measured "
                        "under load, raise it for chirp on large datasets")
    c.add_argument("--models", default=None, help="serving: comma-separated names (default: all five STT)")
    c.add_argument("--host", default="localhost", help="serving: host of the containers")
    c.add_argument("--project", default=None,
                   help="chirp: GCP project (default: $GOOGLE_CLOUD_PROJECT or the active gcloud config)")
    c.add_argument("--location", default="us", help="chirp: Speech v2 location; chirp_3 is GA in us and eu")

    s = sub.add_parser("score", help="score a run directory, write report.md, optionally log to MLflow")
    s.add_argument("--run", required=True)
    s.add_argument("--fixtures", default=str(manifest.FIXTURES_DIR))
    s.add_argument("--mlflow", action="store_true")
    s.add_argument("--allow-missing-baseline", action="store_true")

    o = sub.add_parser("opus", help="build an Opus-degraded copy of a fixture set (robo-be wire codec)")
    o.add_argument("--src", default=str(manifest.FIXTURES_DIR))
    o.add_argument("--dst", default=str(manifest.FIXTURES_DIR.parent / "stt-fixtures-opus24"))
    o.add_argument("--bitrate", type=int, default=24_000)
    o.add_argument("--application", choices=["voip", "audio"], default="voip")

    d = sub.add_parser("datasets", help="download and build the public set: FLEURS vi+en test, VIVOS test")
    d.add_argument("--dst", default=str(manifest.FIXTURES_DIR.parent / "datasets" / "public-v1"))
    d.add_argument("--cache", default=str(manifest.FIXTURES_DIR.parent / "datasets" / ".cache"))

    m = sub.add_parser("compare", help="side-by-side report of several scored runs (e.g. clean vs opus)")
    m.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                   help="repeat; the first run is the baseline for the delta column")
    m.add_argument("--fixtures-root", default=str(manifest.FIXTURES_DIR.parent),
                   help="folder holding each run's dataset (run.json 'dataset' is the subfolder)")
    m.add_argument("--out", required=True)
    m.add_argument("--allow-missing-baseline", action="store_true")
    return p.parse_args(argv)


def build_backends(args: argparse.Namespace) -> list[Backend]:
    if args.backend == "serving":
        names = [n.strip() for n in args.models.split(",")] if args.models else list(STT_MODELS)
        return [ServingBackend(n, host=args.host) for n in names]
    project = args.project or _gcloud_project()
    if not project:
        sys.exit("chirp: pass --project or set GOOGLE_CLOUD_PROJECT (and: uv sync --extra bench)")
    return [ChirpBackend(project=project, location=args.location)]


def cmd_collect(args: argparse.Namespace) -> int:
    for backend in build_backends(args):
        out = collect.collect(backend, Path(args.fixtures), Path(args.run), passes=args.passes,
                              concurrency=args.concurrency)
        print(f"wrote {out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    run_dir, fixtures = Path(args.run), Path(args.fixtures)
    meta, records = score.load_run(run_dir, allow_missing_baseline=args.allow_missing_baseline)
    if meta["dataset_hash"] != manifest.dataset_hash(fixtures):
        sys.exit(f"{run_dir} was collected on dataset {meta['dataset_hash']}; the fixtures on disk differ")
    utts = manifest.load(fixtures)
    cells = score.build_cells(records, utts)
    report = score.render_report(meta, utts, cells, score.nondeterministic(records))
    path = score.write_report(run_dir, report)
    print(f"wrote {path}")
    if args.mlflow:
        from mlflow_registry.bench import mlflow_log
        from mlflow_registry.config import load_config

        ids = mlflow_log.log_all(meta, cells, run_dir, tracking_uri=load_config().tracking_uri)
        print(f"logged {len(ids)} MLflow runs to experiment {mlflow_log.EXPERIMENT}")
    return 0


def cmd_opus(args: argparse.Namespace) -> int:
    from mlflow_registry.bench import opus

    dst = opus.make_opus_fixtures(Path(args.src), Path(args.dst), args.bitrate, args.application)
    print(f"wrote {dst} ({len(manifest.load(dst))} utterances, {manifest.dataset_hash(dst)})")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    runs = []
    for spec in args.run:
        label, sep, path = spec.partition("=")
        if not sep:
            sys.exit(f"--run expects LABEL=DIR, got {spec!r}")
        meta, records = score.load_run(Path(path), allow_missing_baseline=args.allow_missing_baseline)
        fixtures = Path(args.fixtures_root) / meta["dataset"]
        if manifest.dataset_hash(fixtures) != meta["dataset_hash"]:
            sys.exit(f"{path}: dataset {meta['dataset_hash']} does not match {fixtures} on disk")
        runs.append((label, meta, score.build_cells(records, manifest.load(fixtures))))
    out = Path(args.out)
    out.write_text(score.render_comparison(runs), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    dst = datasets.build_public(Path(args.dst), Path(args.cache))
    print(f"wrote {dst} ({len(manifest.load(dst))} utterances, {manifest.dataset_hash(dst)})")
    return 0


COMMANDS = {"collect": cmd_collect, "score": cmd_score, "opus": cmd_opus, "compare": cmd_compare,
            "datasets": cmd_datasets}


def main(argv: list[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

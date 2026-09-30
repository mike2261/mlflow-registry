"""Synthesize the TTS sentence set, judge the audio, score it.

    # Google leg (laptop, Application Default Credentials)
    uv run python scripts/eval_tts.py collect --backend google --run eval/runs/2026-09-30-tts

    # self-hosted leg (devserver: scripts/serve.sh up voxcpm2 vieneu-tts-v3-turbo kokoro-82m qwen3-tts-1.7b-base)
    uv run python scripts/eval_tts.py collect --backend serving --run eval/runs/2026-09-30-tts

    # judges: ASR round trip (a serving STT container, or chirp_3), then quality (devserver, torch)
    uv run python scripts/eval_tts.py asr --judge qwen3-asr-1.7b --run eval/runs/2026-09-30-tts
    uv run python scripts/eval_tts.py asr --judge chirp_3 --run eval/runs/2026-09-30-tts
    uv run python scripts/eval_tts.py quality --run eval/runs/2026-09-30-tts --device cuda

    # score (--mlflow logs to MLFLOW_TRACKING_URI from .env)
    uv run python scripts/eval_tts.py score --run eval/runs/2026-09-30-tts --mlflow

See docs/superpowers/specs/2026-09-30-tts-eval-design.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_stt import _gcloud_project

from mlflow_registry.bench.backends import CHIRP_NAME, STT_MODELS, ChirpBackend, ServingBackend
from mlflow_registry.bench.tts import collect, judge, manifest, score
from mlflow_registry.bench.tts.backends import (
    CLONING,
    TTS_MODELS,
    ElevenLabsBackend,
    GoogleTtsBackend,
    TtsServingBackend,
)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="synthesize every supported sentence; write JSONL + pass-1 WAVs")
    c.add_argument("--backend", choices=["serving", "google", "elevenlabs"], required=True)
    c.add_argument("--run", required=True, help="run directory, e.g. eval/runs/2026-09-30-tts")
    c.add_argument("--sentences", default=str(manifest.SENTENCES_DIR))
    c.add_argument("--passes", type=int, default=3, help="pass 1 is judged, the rest are timed only")
    c.add_argument("--models", default=None, help="serving: comma-separated names (default: all four TTS)")
    c.add_argument("--host", default="localhost", help="serving: host of the containers")

    a = sub.add_parser("asr", help="transcribe pass-1 audio with an STT judge")
    a.add_argument("--judge", required=True, help=f"a serving STT model or {CHIRP_NAME}")
    a.add_argument("--run", required=True)
    a.add_argument("--systems", default=None, help="comma-separated (default: every system in the run)")
    a.add_argument("--host", default="localhost", help="serving judge: host of the container")
    a.add_argument("--project", default=None, help=f"{CHIRP_NAME}: GCP project")
    a.add_argument("--location", default="us", help=f"{CHIRP_NAME}: Speech v2 location")

    q = sub.add_parser("quality", help="UTMOS + speaker similarity (needs: uv sync --extra tts-judge)")
    q.add_argument("--run", required=True)
    q.add_argument("--sentences", default=str(manifest.SENTENCES_DIR))
    q.add_argument("--systems", default=None)
    q.add_argument("--device", default="cpu")

    s = sub.add_parser("score", help="score a run directory, write report.md, optionally log to MLflow")
    s.add_argument("--run", required=True)
    s.add_argument("--sentences", default=str(manifest.SENTENCES_DIR))
    s.add_argument("--mlflow", action="store_true")
    return p.parse_args(argv)


def _names(value: str | None) -> list[str] | None:
    return [n.strip() for n in value.split(",")] if value else None


def cmd_collect(args: argparse.Namespace) -> int:
    if args.backend == "serving":
        ref = manifest.reference(Path(args.sentences))
        backends = [TtsServingBackend(n, ref, host=args.host) for n in _names(args.models) or TTS_MODELS]
    else:
        backends = [GoogleTtsBackend() if args.backend == "google" else ElevenLabsBackend()]
    for backend in backends:
        out = collect.collect(backend, Path(args.sentences), Path(args.run), passes=args.passes)
        print(f"wrote {out}")
    return 0


def cmd_asr(args: argparse.Namespace) -> int:
    if args.judge == CHIRP_NAME:
        project = args.project or _gcloud_project()
        if not project:
            sys.exit(f"{CHIRP_NAME}: pass --project or set GOOGLE_CLOUD_PROJECT")
        stt = ChirpBackend(project=project, location=args.location)
    elif args.judge in STT_MODELS:
        stt = ServingBackend(args.judge, host=args.host)
    else:
        sys.exit(f"unknown judge {args.judge!r}; use {CHIRP_NAME} or one of {', '.join(STT_MODELS)}")
    for path in judge.asr(stt, Path(args.run), _names(args.systems)):
        print(f"wrote {path}")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    mos, embed = judge.load_utmos(args.device), judge.load_speaker(args.device)
    for path in judge.quality(Path(args.run), Path(args.sentences), mos, embed, CLONING, _names(args.systems)):
        print(f"wrote {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    run_dir, sentences_dir = Path(args.run), Path(args.sentences)
    run = score.load_run(run_dir)
    if run.meta["dataset_hash"] != manifest.dataset_hash(sentences_dir):
        sys.exit(f"{run_dir} was collected on dataset {run.meta['dataset_hash']}; the sentences on disk differ")
    sentences = manifest.load(sentences_dir)
    results = score.build_results(run, sentences)
    path = score.write_report(run_dir, score.render_report(run, sentences, results, str(run_dir)))
    print(f"wrote {path}")
    if args.mlflow:
        from mlflow_registry.bench.tts import mlflow_log
        from mlflow_registry.config import load_config

        ids = mlflow_log.log_all(run, results, run_dir, tracking_uri=load_config().tracking_uri)
        print(f"logged {len(ids)} MLflow runs to experiment {mlflow_log.EXPERIMENT}")
    return 0


COMMANDS = {"collect": cmd_collect, "asr": cmd_asr, "quality": cmd_quality, "score": cmd_score}


def main(argv: list[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

"""Measure time to first audio (TTFB) for one TTS system; writes <run>/ttfb/<system>.jsonl.

Self-hosted systems run inside their serving image (weights come from the serving cache, so
start the container once first), with the repo mounted at /work:

    docker compose -f docker-compose.yaml -f docker-compose.serving.yaml run --rm --no-deps \\
        -v "$PWD":/work -w /work -e PYTHONPATH=/work/src --entrypoint python \\
        serve-voxcpm2 scripts/tts_ttfb.py local --system voxcpm2 --run eval/runs/2026-09-30-tts

Google streams from the laptop (network included):

    uv run python scripts/tts_ttfb.py google --run eval/runs/2026-09-30-tts

Kept apart from eval_tts.py because the serving images do not ship the scoring dependencies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlflow_registry.bench.tts import manifest, ttfb
from mlflow_registry.bench.tts.backends import (
    GOOGLE_LANG_CODES,
    GOOGLE_NAME,
    GOOGLE_VOICE,
    SYSTEM_LANGS,
    TTS_MODELS,
    VOICING,
)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("where", choices=["local", "google"])
    p.add_argument("--system", choices=TTS_MODELS, help="local: the model this image serves")
    p.add_argument("--weights-version", default="1", help="local: raw weights version the wrapper wraps")
    p.add_argument("--run", required=True)
    p.add_argument("--sentences", default=str(manifest.SENTENCES_DIR))
    p.add_argument("--passes", type=int, default=3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    sentences = Path(args.sentences)
    if args.where == "google":
        streamer = ttfb.google_streamer(GOOGLE_VOICE, SYSTEM_LANGS[GOOGLE_NAME], GOOGLE_LANG_CODES, GOOGLE_NAME)
    else:
        if not args.system:
            sys.exit("local: pass --system")
        streamer = ttfb.local_streamer(args.system, args.weights_version, manifest.reference(sentences),
                                       SYSTEM_LANGS[args.system], VOICING[args.system])
    print(f"wrote {ttfb.run(streamer, sentences, Path(args.run), passes=args.passes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

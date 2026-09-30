"""Create the ``@serving`` pyfunc version for the shortlisted models.

    uv run python scripts/register_serving.py                 # all nine
    uv run python scripts/register_serving.py kokoro-82m      # a subset
    uv run python scripts/register_serving.py --weights-version 1 whisper-large-v3

Each run logs a new pyfunc version (a few KB) that points at the raw weights
version, tags it ``flavor=pyfunc wraps_version=N`` and moves the ``serving``
alias to it. Weights are never re-uploaded. Settings come from ``.env`` /
the environment like every other command in this repo.
"""

import argparse
import logging
import sys

from mlflow_registry.config import load_config
from mlflow_registry.serving.catalog import SERVING, spec
from mlflow_registry.serving.register import register_serving

log = logging.getLogger("register-serving")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="registered model names (default: all nine)")
    p.add_argument("--weights-version", help="wrap this version instead of the latest raw one")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    names = args.names or list(SERVING)
    failures = 0
    for name in names:
        try:
            version = register_serving(spec(name), args.weights_version, tracking_uri=config.tracking_uri)
        except Exception:
            failures += 1
            log.exception("FAILED %s", name)
            continue
        log.info("%s -> version %s, alias serving (port %d)", name, version, spec(name).port)
    log.info("finished: %d ok, %d failed", len(names) - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

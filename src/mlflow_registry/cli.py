"""Command-line front door for the registry.

    mlflow-registry register NAME SOURCE [--tag k=v ...] [--allow PAT] [--ignore PAT]
    mlflow-registry promote  NAME VERSION [--alias production]
    mlflow-registry resolve  NAME [--alias production] [--version]
    mlflow-registry download NAME [--alias production] [--dest DIR]
    mlflow-registry list [--json]

SOURCE is a fetcher spec: a local dir, ``local:/path``, ``hf:org/model[@rev]``
or ``pretrained:pkg.module:callable``. Settings come from ``.env`` / the
environment (MLFLOW_TRACKING_URI etc.), so the same commands work against the
laptop stack or a dev server.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence

from mlflow_registry.registry import DEFAULT_ALIAS, ModelRegistry


def _parse_tag(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlflow-registry",
        description="Register, promote, resolve and download models in the MLflow registry.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="upload weights as a new version")
    reg.add_argument("name")
    reg.add_argument("source", help="local dir | local:PATH | hf:ORG/MODEL[@REV] | pretrained:MODULE:CALLABLE")
    reg.add_argument("--tag", action="append", type=_parse_tag, default=[], metavar="KEY=VALUE")
    reg.add_argument("--allow", action="append", metavar="PATTERN", help="hf: only, keep matching files")
    reg.add_argument("--ignore", action="append", metavar="PATTERN", help="hf: only, skip matching files")

    pro = sub.add_parser("promote", help="point an alias at a version (also rollback)")
    pro.add_argument("name")
    pro.add_argument("version")
    pro.add_argument("--alias", default=DEFAULT_ALIAS)

    res = sub.add_parser("resolve", help="print the artifact URI behind NAME@ALIAS")
    res.add_argument("name")
    res.add_argument("--alias", default=DEFAULT_ALIAS)
    res.add_argument("--version", action="store_true", help="print the version number instead")

    dl = sub.add_parser("download", help="fetch weights behind NAME@ALIAS, print the local dir")
    dl.add_argument("name")
    dl.add_argument("--alias", default=DEFAULT_ALIAS)
    dl.add_argument("--dest", help="target directory (default: temp dir)")

    ls = sub.add_parser("list", help="list registered models, versions, aliases and tags")
    ls.add_argument("--json", action="store_true")
    return p


def _cmd_register(reg: ModelRegistry, a: argparse.Namespace) -> int:
    opts = {}
    if a.allow:
        opts["allow_patterns"] = a.allow
    if a.ignore:
        opts["ignore_patterns"] = a.ignore
    print(reg.register(a.name, a.source, tags=dict(a.tag) or None, **opts))
    return 0


def _cmd_promote(reg: ModelRegistry, a: argparse.Namespace) -> int:
    reg.promote(a.name, a.version, alias=a.alias)
    return 0


def _cmd_resolve(reg: ModelRegistry, a: argparse.Namespace) -> int:
    if a.version:
        print(reg.get_current_version(a.name, alias=a.alias))
    else:
        print(reg.resolve(a.name, alias=a.alias))
    return 0


def _cmd_download(reg: ModelRegistry, a: argparse.Namespace) -> int:
    if a.dest:
        os.makedirs(a.dest, exist_ok=True)
    print(reg.download(a.name, alias=a.alias, dest_dir=a.dest))
    return 0


def _cmd_list(reg: ModelRegistry, a: argparse.Namespace) -> int:
    rows = reg.list_versions()
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no registered models")
        return 0
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        aliases = ",".join(r["aliases"]) or "-"
        tags = " ".join(f"{k}={v}" for k, v in sorted(r["tags"].items()))
        print(f"{r['name']:<{width}}  v{r['version']:<3} {aliases:<12} {tags}")
    return 0


_COMMANDS = {
    "register": _cmd_register,
    "promote": _cmd_promote,
    "resolve": _cmd_resolve,
    "download": _cmd_download,
    "list": _cmd_list,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # stdout is the CLI's return value (a version, a URI, a path); keep it clean.
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    # Like `python -m`: let pretrained:module:callable resolve against the cwd.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        registry = ModelRegistry()
        return _COMMANDS[args.command](registry, args)
    except Exception as e:  # surface a one-line error, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

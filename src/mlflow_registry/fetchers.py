"""Source adapters: turn "where the model comes from" into a local directory.

Every fetcher returns a local dir path, which is the only thing
``ModelRegistry.register`` needs. ``fetch(spec)`` is the single entry point
that picks the right fetcher from a source spec string:

    "local:/path/to/weights"   or a bare path            -> from_local
    "hf:org/model"             or "hf:org/model@revision" -> from_huggingface
    "pretrained:pkg.module:callable"                       -> from_pretrained

The ``pretrained:`` form imports ``callable``, calls it with no arguments and
expects a model or a ``(model, tokenizer)`` tuple, so a finetune script can be
registered from the command line.
"""

import importlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class SavesPretrained(Protocol):
    """Duck type for transformers-style models and tokenizers."""

    def save_pretrained(self, save_directory: str) -> object: ...


def from_local(path: str) -> str:
    """Already on disk. Validate it is a directory and hand it back."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"expected a directory of weights, got a file: {p}")
    return str(p)


def from_huggingface(
    repo_id: str,
    revision: str | None = None,
    *,
    token: str | None = None,
    local_dir: str | None = None,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    """Download a Hugging Face Hub snapshot and return its local dir.

    Requires the ``hf`` extra: ``uv sync --extra hf``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "from_huggingface needs the 'huggingface_hub' package; "
            "install with `uv sync --extra hf`"
        ) from e
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def from_pretrained(
    model: SavesPretrained, tokenizer: SavesPretrained | None = None
) -> str:
    """Save an in-memory model (e.g. after finetuning) to a temp dir."""
    out = tempfile.mkdtemp(prefix="mlflow-registry-")
    try:
        model.save_pretrained(out)
        if tokenizer is not None:
            tokenizer.save_pretrained(out)
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out


# -- dispatcher -------------------------------------------------------------

_SCHEMES = ("local", "hf", "pretrained")


@dataclass(frozen=True)
class Fetched:
    """Where the weights landed, plus what we know about where they came from."""

    path: str
    source: str
    revision: str | None = None  # resolved HF commit hash, when known


def fetch_with_meta(spec: str, **hf_options: Any) -> Fetched:
    """Resolve a source spec to a local directory, keeping provenance.

    A spec without ``:`` is a bare local path. A local path that itself
    contains ``:`` must be written as ``local:<path>``. ``hf_options`` are
    passed through to ``from_huggingface`` (``allow_patterns``, ``token``...).
    """
    scheme, sep, rest = spec.partition(":")
    if not sep:
        return Fetched(path=from_local(spec), source=spec)
    if scheme == "local":
        return Fetched(path=from_local(rest), source=spec)
    if scheme == "hf":
        repo_id, _, revision = rest.partition("@")
        path = from_huggingface(repo_id, revision or None, **hf_options)
        return Fetched(path=path, source=spec, revision=_hf_snapshot_revision(path))
    if scheme == "pretrained":
        return Fetched(path=_from_pretrained_spec(rest), source=spec)
    raise ValueError(
        f"unsupported source scheme {scheme!r} in {spec!r}; "
        f"expected one of {', '.join(_SCHEMES)} or a bare local path"
    )


def fetch(spec: str, **hf_options: Any) -> str:
    """Resolve a source spec to a local directory of weights."""
    return fetch_with_meta(spec, **hf_options).path


def _hf_snapshot_revision(path: str) -> str | None:
    """The HF cache stores snapshots as ``.../snapshots/<commit>``."""
    p = Path(path)
    return p.name if p.parent.name == "snapshots" else None


def _from_pretrained_spec(target: str) -> str:
    """``pkg.module:callable`` -> import, call, save with ``from_pretrained``."""
    module_name, sep, attr = target.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"pretrained spec must be 'module:callable', got {target!r}"
        )
    loader = getattr(importlib.import_module(module_name), attr)
    result = loader()
    model, tokenizer = result if isinstance(result, tuple) else (result, None)
    for obj in (model, tokenizer):
        if obj is not None and not hasattr(obj, "save_pretrained"):
            raise TypeError(
                f"{target} returned {type(obj).__name__}, which has no save_pretrained()"
            )
    return from_pretrained(model, tokenizer)

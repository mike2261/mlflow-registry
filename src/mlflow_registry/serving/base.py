"""pyfunc base classes and the uniform REST contract for STT and TTS models.

Request payloads follow MLflow's scoring server conventions (``dataframe_records``,
``dataframe_split`` or ``inputs``). The server hands ``predict`` a pandas
DataFrame with these columns:

STT input   audio_b64 (str, required) · language (str, optional: "vi", "en", ...)
STT output  text (str) · language (str or null: detected, or the request value)

TTS input   text (str, required) · voice (optional) · language (optional)
            · ref_audio_b64 (optional, cloning) · ref_text (optional, cloning)
TTS output  audio_b64 (16-bit PCM WAV) · sample_rate (int)

A concrete wrapper subclasses ``SttModel`` or ``TtsModel`` and implements
``_load(weights_dir)`` plus ``_transcribe`` / ``_synthesize``. Runtime imports
belong inside ``_load`` so the class stays importable without torch et al.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel
from mlflow.types import ColSpec, Schema

from mlflow_registry.serving.audio import decode_b64, decode_b64_native, encode_b64
from mlflow_registry.serving.weights import ensure_weights

STT_SIGNATURE = ModelSignature(
    inputs=Schema([
        ColSpec("string", "audio_b64"),
        ColSpec("string", "language", required=False),
    ]),
    outputs=Schema([
        ColSpec("string", "text"),
        ColSpec("string", "language", required=False),
    ]),
)

TTS_SIGNATURE = ModelSignature(
    inputs=Schema([
        ColSpec("string", "text"),
        ColSpec("string", "voice", required=False),
        ColSpec("string", "language", required=False),
        ColSpec("string", "ref_audio_b64", required=False),
        ColSpec("string", "ref_text", required=False),
    ]),
    outputs=Schema([
        ColSpec("string", "audio_b64"),
        ColSpec("long", "sample_rate"),
    ]),
)


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None


@dataclass(frozen=True)
class TtsRequest:
    text: str
    voice: str | None = None
    language: str | None = None
    ref_audio: tuple[np.ndarray, int] | None = None  # (float32 mono, native sample rate)
    ref_text: str | None = None


def rows(model_input: Any) -> list[dict[str, Any]]:
    """Normalise whatever the scoring server passes into a list of row dicts.

    NaN (pandas' null for object columns) becomes ``None`` so optional string
    fields are simply absent rather than ``float('nan')``.
    """
    if isinstance(model_input, pd.DataFrame):
        records = model_input.to_dict(orient="records")
    elif isinstance(model_input, list):
        records = list(model_input)
    elif isinstance(model_input, dict):
        if model_input and all(isinstance(v, (list, tuple, np.ndarray)) for v in model_input.values()):
            records = pd.DataFrame(model_input).to_dict(orient="records")
        else:
            records = [model_input]
    else:
        raise TypeError(f"unsupported model input type {type(model_input).__name__}")
    return [{k: _clean(v) for k, v in r.items()} for r in records]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_str(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


class _RegistryModel(PythonModel):
    """Common plumbing: locate the weights, then hand off to the runtime."""

    task: str = ""

    def __init__(self, name: str, version: str, weights_dir: str | None = None) -> None:
        self.name = name
        self.version = str(version)
        # Explicit override for local runs and tests; None means fetch from the registry.
        self.weights_dir = weights_dir

    def load_context(self, context) -> None:  # noqa: ARG002 - MLflow API
        weights = Path(self.weights_dir) if self.weights_dir else ensure_weights(self.name, self.version)
        self._load(weights)

    def _load(self, weights_dir: Path) -> None:
        raise NotImplementedError

    # Keep the pickled payload small and independent of runtime handles.
    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "weights_dir": self.weights_dir}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class SttModel(_RegistryModel):
    task = "stt"
    SAMPLE_RATE = 16_000

    def predict(self, context, model_input, params=None) -> pd.DataFrame:  # noqa: ARG002
        out = []
        for row in rows(model_input):
            payload = row.get("audio_b64")
            if not payload:
                raise ValueError("audio_b64 is required")
            audio = decode_b64(payload, target_sr=self.SAMPLE_RATE)
            result = self._transcribe(audio, _optional_str(row, "language"))
            out.append({"text": result.text, "language": result.language})
        return pd.DataFrame(out, columns=["text", "language"])

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        raise NotImplementedError


class TtsModel(_RegistryModel):
    task = "tts"

    def predict(self, context, model_input, params=None) -> pd.DataFrame:  # noqa: ARG002
        out = []
        for row in rows(model_input):
            text = _optional_str(row, "text")
            if not text:
                raise ValueError("text is required")
            ref_b64 = _optional_str(row, "ref_audio_b64")
            req = TtsRequest(
                text=text,
                voice=_optional_str(row, "voice"),
                language=_optional_str(row, "language"),
                ref_audio=decode_b64_native(ref_b64) if ref_b64 else None,
                ref_text=_optional_str(row, "ref_text"),
            )
            samples, sample_rate = self._synthesize(req)
            out.append({"audio_b64": encode_b64(samples, sample_rate), "sample_rate": int(sample_rate)})
        return pd.DataFrame(out, columns=["audio_b64", "sample_rate"])

    def _synthesize(self, req: TtsRequest) -> tuple[np.ndarray, int]:
        raise NotImplementedError

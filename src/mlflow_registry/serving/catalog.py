"""The nine servable models: which wrapper, which port, GPU or not.

Single source of truth for ``scripts/register_serving.py`` (what to log),
``docker-compose.serving.yaml`` (ports, GPU) and the README table. Order
matches ``scripts/register_shortlist.py``; host ports follow that order.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel

from mlflow_registry.serving.base import STT_SIGNATURE, TTS_SIGNATURE

Task = Literal["stt", "tts"]
SERVING_ALIAS = "serving"
CONTAINER_PORT = 8080


@dataclass(frozen=True)
class ServingSpec:
    name: str                 # registered model name
    task: Task
    wrapper: str              # "package.module:ClassName"
    port: int                 # host port the container is published on
    gpu: bool = True
    note: str = ""            # one-line hint for the README / invoke help

    @property
    def signature(self) -> ModelSignature:
        return STT_SIGNATURE if self.task == "stt" else TTS_SIGNATURE

    def wrapper_cls(self) -> type[PythonModel]:
        module_name, _, cls_name = self.wrapper.partition(":")
        return getattr(importlib.import_module(module_name), cls_name)

    def make_wrapper(self, weights_version: str, weights_dir: str | None = None) -> PythonModel:
        return self.wrapper_cls()(self.name, weights_version, weights_dir=weights_dir)


_W = "mlflow_registry.serving"

SERVING: dict[str, ServingSpec] = {
    s.name: s
    for s in [
        # --- STT -------------------------------------------------------------
        ServingSpec("qwen3-asr-1.7b", "stt", f"{_W}.stt_qwen3:Qwen3AsrStt", 5001,
                    note="language vi/en or omit for auto-detect"),
        ServingSpec("granite-speech-4.1-2b", "stt", f"{_W}.stt_granite:GraniteStt", 5002,
                    note="English only"),
        ServingSpec("gipformer1.5-68m-rnnt", "stt", f"{_W}.stt_gipformer:GipformerStt", 5003, gpu=False,
                    note="Vietnamese only, CPU via sherpa-onnx"),
        ServingSpec("parakeet-ctc-0.6b-vietnamese", "stt", f"{_W}.stt_parakeet:ParakeetStt", 5004,
                    note="Vietnamese only, NeMo"),
        ServingSpec("whisper-large-v3", "stt", f"{_W}.stt_whisper:WhisperStt", 5005,
                    note="multilingual baseline"),
        # --- TTS -------------------------------------------------------------
        ServingSpec("voxcpm2", "tts", f"{_W}.tts_voxcpm:VoxCpmTts", 5006,
                    note="voice = free-text description; ref_audio_b64 (+ref_text) clones"),
        ServingSpec("vieneu-tts-v3-turbo", "tts", f"{_W}.tts_vieneu:VieneuTts", 5007,
                    note="voice = preset name e.g. 'Mai Anh'; ref_audio_b64 clones"),
        ServingSpec("kokoro-82m", "tts", f"{_W}.tts_kokoro:KokoroTts", 5008,
                    note="voice = e.g. af_heart, bm_george (files in voices/)"),
        ServingSpec("qwen3-tts-1.7b-base", "tts", f"{_W}.tts_qwen3:Qwen3TtsTts", 5009,
                    note="base clone model: ref_audio_b64 required, ref_text recommended"),
        # --- added after the first shortlist ----------------------------------
        ServingSpec("phowhisper-large", "stt", f"{_W}.stt_whisper:WhisperStt", 5010,
                    note="Vietnamese only (Whisper large fine-tuned by VinAI)"),
        ServingSpec("cohere-transcribe-03-2026", "stt", f"{_W}.stt_cohere:CohereTranscribeStt", 5011,
                    note="language vi/en (defaults to vi); robo-be's English ASR"),
        ServingSpec("fun-asr-mlt-nano-2512", "stt", f"{_W}.stt_funasr:FunAsrStt", 5012,
                    note="language vi/en or omit for auto-detect incl. code-switching"),
    ]
}


def spec(name: str) -> ServingSpec:
    try:
        return SERVING[name]
    except KeyError:
        raise KeyError(f"{name!r} is not servable; known: {', '.join(SERVING)}") from None

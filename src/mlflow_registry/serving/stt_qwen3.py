"""qwen3-asr-1.7b via the ``qwen-asr`` package (transformers backend)."""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import cuda_available
from mlflow_registry.serving.base import SttModel, Transcript

# qwen-asr wants human-readable language names, not ISO codes.
_TO_QWEN = {"vi": "Vietnamese", "en": "English", "zh": "Chinese", "ja": "Japanese",
            "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish"}
_FROM_QWEN = {v: k for k, v in _TO_QWEN.items()}


class Qwen3AsrStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        cuda = cuda_available()
        self._model = Qwen3ASRModel.from_pretrained(
            str(weights_dir),
            dtype=torch.bfloat16 if cuda else torch.float32,
            device_map="cuda:0" if cuda else "cpu",
            max_inference_batch_size=8,
            max_new_tokens=512,
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        qwen_lang = _TO_QWEN.get(language.lower(), language) if language else None
        result = self._model.transcribe(audio=(audio, self.SAMPLE_RATE), language=qwen_lang)[0]
        detected = getattr(result, "language", None) or ""
        # "Vietnamese,English" for code-switched clips; keep the first as the row language
        first = detected.split(",")[0].strip()
        return Transcript(text=result.text.strip(), language=_FROM_QWEN.get(first, first or None) or language)

"""Whisper-architecture models (whisper-large-v3, phowhisper-large) via the transformers ASR pipeline."""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import cuda_available
from mlflow_registry.serving.base import SttModel, Transcript


class WhisperStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        import torch
        from transformers import pipeline

        cuda = cuda_available()
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=str(weights_dir),
            torch_dtype=torch.float16 if cuda else torch.float32,
            device=0 if cuda else -1,
            chunk_length_s=30,  # long-form audio is chunked instead of truncated
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language  # ISO code ("vi") or name ("vietnamese")
        result = self._pipe(
            {"raw": audio, "sampling_rate": self.SAMPLE_RATE},
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )
        return Transcript(text=result["text"].strip(), language=language)

"""cohere-transcribe-03-2026 via transformers' native ``CohereAsrForConditionalGeneration``.

Same call as robo-be's Cohere sidecar (``robo_infer/models/asr_cohere.py``): processor with the
language, ``generate``, decode with ``audio_chunk_index`` so long clips are stitched back. The
model needs a language; without a hint the wrapper falls back to Vietnamese, the product's
default, and reports that as the language.
"""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import cuda_available
from mlflow_registry.serving.base import SttModel, Transcript

DEFAULT_LANGUAGE = "vi"


class CohereTranscribeStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        cuda = cuda_available()
        self._processor = AutoProcessor.from_pretrained(str(weights_dir))
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            str(weights_dir),
            torch_dtype=torch.bfloat16 if cuda else torch.float32,
            device_map="cuda" if cuda else "cpu",
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        lang = language or DEFAULT_LANGUAGE
        inputs = self._processor([audio], sampling_rate=self.SAMPLE_RATE, return_tensors="pt", language=lang)
        chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self._model.device, dtype=self._model.dtype)
        outputs = self._model.generate(**inputs, max_new_tokens=512)
        decoded = self._processor.decode(outputs, skip_special_tokens=True,
                                         audio_chunk_index=chunk_index, language=lang)
        text = decoded if isinstance(decoded, str) else decoded[0]
        return Transcript(text=text.strip(), language=lang)

"""qwen3-tts-1.7b-base via the ``qwen-tts`` package.

The *Base* checkpoint only supports voice cloning: every request needs
``ref_audio_b64`` (a 3-10 s clip of the target speaker). ``ref_text`` (its
transcript) gives the best quality; without it the model falls back to
speaker-embedding-only cloning. ``language`` is ``vi``/``en``/... or omitted
for Qwen's automatic choice.
"""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import cuda_available
from mlflow_registry.serving.base import TtsModel, TtsRequest

_TO_QWEN = {"vi": "Vietnamese", "en": "English", "zh": "Chinese", "ja": "Japanese",
            "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish"}


class Qwen3TtsTts(TtsModel):
    def _load(self, weights_dir: Path) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        cuda = cuda_available()
        self._model = Qwen3TTSModel.from_pretrained(
            str(weights_dir),
            device_map="cuda:0" if cuda else "cpu",
            dtype=torch.bfloat16 if cuda else torch.float32,
        )

    def _synthesize(self, req: TtsRequest) -> tuple[np.ndarray, int]:
        if req.ref_audio is None:
            raise ValueError(
                "qwen3-tts-1.7b-base is a voice-clone base model: send ref_audio_b64 "
                "(3-10 s of the target speaker) and ideally ref_text (its transcript)"
            )
        language = _TO_QWEN.get(req.language.lower(), req.language) if req.language else "Auto"
        wavs, sample_rate = self._model.generate_voice_clone(
            text=req.text,
            language=language,
            ref_audio=req.ref_audio,          # (float32 mono, native sample rate)
            ref_text=req.ref_text,
            x_vector_only_mode=req.ref_text is None,
        )
        return np.asarray(wavs[0], dtype=np.float32), int(sample_rate)

"""kokoro-82m via the ``kokoro`` package, loading config, weights and voices from the snapshot.

``voice`` is a preset name from the snapshot's ``voices/`` folder (``af_heart``,
``bm_george``, ...). The first letter of the voice picks the G2P language when
``language`` is not given. Needs ``espeak-ng`` on the host image.
"""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import torch_device
from mlflow_registry.serving.base import TtsModel, TtsRequest

_REPO_ID = "hexgrad/Kokoro-82M"  # only used by kokoro for logging / voice-name warnings
_SAMPLE_RATE = 24_000
_DEFAULT_VOICE = "af_heart"
_LANG_TO_CODE = {"en": "a", "en-us": "a", "en-gb": "b", "es": "e", "fr": "f", "hi": "h",
                 "it": "i", "pt": "p", "pt-br": "p", "ja": "j", "zh": "z"}


class KokoroTts(TtsModel):
    def _load(self, weights_dir: Path) -> None:
        from kokoro import KModel

        self._dir = weights_dir
        self._model = KModel(
            repo_id=_REPO_ID,
            config=str(weights_dir / "config.json"),
            model=str(weights_dir / "kokoro-v1_0.pth"),
        ).to(torch_device()).eval()
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, lang_code: str):
        from kokoro import KPipeline

        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=_REPO_ID, model=self._model)
        return self._pipelines[lang_code]

    def _synthesize(self, req: TtsRequest) -> tuple[np.ndarray, int]:
        voice = req.voice or _DEFAULT_VOICE
        voice_path = self._dir / "voices" / f"{voice}.pt"
        if not voice_path.exists():
            available = sorted(p.stem for p in (self._dir / "voices").glob("*.pt"))
            raise ValueError(f"unknown kokoro voice {voice!r}; available: {', '.join(available)}")
        lang_code = _LANG_TO_CODE.get(req.language.lower(), voice[0]) if req.language else voice[0]

        chunks = []
        for result in self._pipeline(lang_code)(req.text, voice=str(voice_path)):
            audio = result.audio
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        if not chunks:
            raise ValueError("kokoro produced no audio for the given text")
        return np.concatenate(chunks).astype(np.float32), _SAMPLE_RATE

"""fun-asr-mlt-nano-2512 via ``funasr`` plus the FunAudioLLM/Fun-ASR repo's model code.

The weights repo holds only checkpoints; the ``FunASRNano`` model class lives in the GitHub repo
(``model.py``, ``ctc.py``, ``tools/``). The image unpacks a pinned commit of that repo into
``FUN_ASR_CODE`` (see ``serving/requirements/fun-asr-mlt-nano-2512.sh``); importing its
``model`` module registers the class with funasr, exactly as robo-be's Fun-ASR sidecar does.

Without a language hint the model runs its own auto-detection, which also handles
Vietnamese-English code-switching. ITN is on (numbers come back as digits), as in robo-be.
"""

import os
import sys
from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import torch_device
from mlflow_registry.serving.base import SttModel, Transcript

CODE_ENV = "FUN_ASR_CODE"
DEFAULT_CODE_DIR = "/opt/fun-asr"
# Fun-ASR takes Chinese language names.
_LANGUAGES = {"vi": "越南语", "en": "English", "zh": "中文", "ja": "日文", "ko": "韩文"}


class FunAsrStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        code_dir = os.environ.get(CODE_ENV, DEFAULT_CODE_DIR)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        import model  # noqa: F401  (registers FunASRNano with funasr's tables)
        from funasr import AutoModel

        self._model = AutoModel(model=str(weights_dir), trust_remote_code=True,
                                device=torch_device(), disable_update=True)

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        import torch

        kwargs = {"input": [torch.from_numpy(audio.astype(np.float32))], "cache": {}, "batch_size": 1, "itn": True}
        if language:
            if language not in _LANGUAGES:
                raise ValueError(f"unsupported language {language!r}; supported: {sorted(_LANGUAGES)} or none")
            kwargs["language"] = _LANGUAGES[language]
        result = self._model.generate(**kwargs)
        return Transcript(text=result[0].get("text", "").strip(), language=language)

"""vieneu-tts-v3-turbo via the ``vieneu`` package.

GPU: PyTorch backbone from ``<snapshot>/update``. CPU: ONNX graphs from
``<snapshot>/onnx_update``. The MOSS audio tokenizer is a separate Hugging Face
repo that vieneu downloads on first start (cached in the shared model volume
through ``HF_HOME``). ``voice`` is a preset name (``tts.list_preset_voices()``);
``ref_audio_b64`` clones a 3-8 s clip instead.
"""

import contextlib
from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import cuda_available
from mlflow_registry.serving.audio import temp_wav
from mlflow_registry.serving.base import TtsModel, TtsRequest


class VieneuTts(TtsModel):
    def _load(self, weights_dir: Path) -> None:
        from vieneu import Vieneu

        cuda = cuda_available()
        self._tts = Vieneu(
            mode="v3turbo",
            backbone_repo=str(weights_dir),
            device="cuda" if cuda else "cpu",
            backend="pytorch" if cuda else "onnx",
        )
        self._sample_rate = int(self._tts.sample_rate)

    def _synthesize(self, req: TtsRequest) -> tuple[np.ndarray, int]:
        kwargs = {"show_progress": False} if "show_progress" in self._tts.infer.__code__.co_varnames else {}
        with contextlib.ExitStack() as stack:
            if req.ref_audio is not None:
                kwargs["ref_audio"] = stack.enter_context(temp_wav(*req.ref_audio))
            elif req.voice:
                kwargs["voice"] = req.voice
            wav = self._tts.infer(req.text, **kwargs)
        return np.asarray(wav, dtype=np.float32), self._sample_rate

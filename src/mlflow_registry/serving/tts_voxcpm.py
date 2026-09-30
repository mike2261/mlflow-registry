"""voxcpm2 via the ``voxcpm`` package.

``voice`` is a free-text voice description that VoxCPM2 reads from a
parenthesised prefix, e.g. ``"A young woman, gentle and sweet voice"``.
``ref_audio_b64`` switches to cloning; adding ``ref_text`` (the transcript of
the reference clip) enables VoxCPM's higher-fidelity prompt mode.
Set ``VOXCPM_COMPILE=true`` to enable torch.compile (slower start, faster steps).
"""

import contextlib
import os
from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import torch_device
from mlflow_registry.serving.audio import temp_wav
from mlflow_registry.serving.base import TtsModel, TtsRequest


class VoxCpmTts(TtsModel):
    def _load(self, weights_dir: Path) -> None:
        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(
            str(weights_dir),
            load_denoiser=False,
            optimize=os.environ.get("VOXCPM_COMPILE", "false").lower() == "true",
            device=torch_device(),
        )
        self._sample_rate = int(self._model.tts_model.sample_rate)

    def _synthesize(self, req: TtsRequest) -> tuple[np.ndarray, int]:
        text = req.text
        kwargs = {"cfg_value": 2.0, "inference_timesteps": 10}
        with contextlib.ExitStack() as stack:
            if req.ref_audio is not None:
                path = stack.enter_context(temp_wav(*req.ref_audio))
                kwargs["reference_wav_path"] = path
                if req.ref_text:
                    kwargs["prompt_wav_path"] = path
                    kwargs["prompt_text"] = req.ref_text
            elif req.voice:
                text = f"({req.voice}){text}"
            wav = self._model.generate(text=text, **kwargs)
        return np.asarray(wav, dtype=np.float32), self._sample_rate

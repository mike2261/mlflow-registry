"""parakeet-ctc-0.6b-vietnamese via NVIDIA NeMo (``nemo_toolkit[asr]``)."""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import torch_device
from mlflow_registry.serving.audio import temp_wav
from mlflow_registry.serving.base import SttModel, Transcript


class ParakeetStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        import nemo.collections.asr as nemo_asr

        nemo_file = next(weights_dir.glob("*.nemo"))
        self._model = nemo_asr.models.ASRModel.restore_from(str(nemo_file), map_location=torch_device())
        self._model.eval()

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:  # noqa: ARG002
        # NeMo's transcribe() wants file paths (or a manifest).
        with temp_wav(audio, self.SAMPLE_RATE) as path:
            outputs = self._model.transcribe([path])
        hyp = outputs[0]
        if isinstance(hyp, list):  # some NeMo versions nest per-decoder results
            hyp = hyp[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        return Transcript(text=text.strip(), language="vi")

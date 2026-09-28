"""gipformer1.5-68m-rnnt (Zipformer transducer, ONNX) via sherpa-onnx on CPU.

Env knobs: ``GIPFORMER_QUANTIZE=int8`` picks the int8 graphs (smaller, ~2x
faster, slightly worse WER); ``GIPFORMER_THREADS`` sets onnxruntime threads.
"""

import os
from pathlib import Path

import numpy as np

from mlflow_registry.serving.base import SttModel, Transcript


class GipformerStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        import sherpa_onnx

        suffix = ".int8" if os.environ.get("GIPFORMER_QUANTIZE", "fp32") == "int8" else ""
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(weights_dir / f"encoder{suffix}.onnx"),
            decoder=str(weights_dir / f"decoder{suffix}.onnx"),
            joiner=str(weights_dir / f"joiner{suffix}.onnx"),
            tokens=str(weights_dir / "tokens.txt"),
            num_threads=int(os.environ.get("GIPFORMER_THREADS", "4")),
            sample_rate=self.SAMPLE_RATE,
            feature_dim=80,
            decoding_method="modified_beam_search",
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:  # noqa: ARG002
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.SAMPLE_RATE, audio.astype(np.float32))
        self._recognizer.decode_streams([stream])
        return Transcript(text=stream.result.text.strip(), language="vi")

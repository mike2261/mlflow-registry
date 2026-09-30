"""Runtime-free wrappers used to exercise the pyfunc base classes end to end.

They live in an importable module (not the test file) because cloudpickle stores
classes from importable modules by reference, exactly like the real wrappers
that get installed inside each Docker image.
"""

from pathlib import Path

import numpy as np

from mlflow_registry.serving.base import SttModel, Transcript, TtsModel, TtsRequest


class FakeStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir
        self.marker = (weights_dir / "weights.txt").read_text().strip()

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        return Transcript(
            text=f"{audio.shape[0]} samples from {self.marker}",
            language=language or "auto",
        )


class FakeTts(TtsModel):
    def _load(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def _synthesize(self, req: TtsRequest) -> tuple[np.ndarray, int]:
        # One sample per character so tests can check text reached the runtime,
        # and echo the reference clip length so cloning inputs are observable.
        n = len(req.text) + (req.ref_audio[0].shape[0] if req.ref_audio else 0)
        self.last_request = req
        return np.zeros(n, dtype=np.float32), 8_000

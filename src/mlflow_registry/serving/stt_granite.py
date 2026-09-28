"""granite-speech-4.1-2b via native transformers (AutoModelForSpeechSeq2Seq)."""

from pathlib import Path

import numpy as np

from mlflow_registry.serving._device import cuda_available
from mlflow_registry.serving.base import SttModel, Transcript

_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."


class GraniteStt(SttModel):
    def _load(self, weights_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        cuda = cuda_available()
        self._device = "cuda" if cuda else "cpu"
        self._processor = AutoProcessor.from_pretrained(str(weights_dir))
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(weights_dir),
            device_map=self._device,
            torch_dtype=torch.bfloat16 if cuda else torch.float32,
        )
        chat = [{"role": "user", "content": _PROMPT}]
        self._prompt = self._processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> Transcript:
        import torch

        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, samples), mono 16 kHz
        inputs = self._processor(self._prompt, wav, device=self._device, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=448, do_sample=False, num_beams=1)
        prompt_len = inputs["input_ids"].shape[-1]
        text = self._processor.tokenizer.batch_decode(
            out[:, prompt_len:], add_special_tokens=False, skip_special_tokens=True
        )[0]
        return Transcript(text=text.strip(), language=language or "en")

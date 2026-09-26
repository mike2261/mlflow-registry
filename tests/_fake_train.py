"""Import target for pretrained: specs in tests. Mimics a finetune script."""

from pathlib import Path


class _Saves:
    def __init__(self, filename: str):
        self.filename = filename

    def save_pretrained(self, out: str) -> None:
        Path(out, self.filename).write_text("saved")


def load_model():
    return _Saves("model.bin")


def load_model_and_tokenizer():
    return _Saves("model.bin"), _Saves("tokenizer.json")


def not_a_model():
    return "nope"

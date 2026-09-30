import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SENTENCES = [
    {"id": "v1", "text": "Xin chào các bạn.", "lang": "vi", "category": "vi_short", "en_words": []},
    {"id": "m1", "text": "Bé ơi, dog nghĩa là con chó.", "lang": "vi", "category": "mix", "en_words": ["dog"]},
    {"id": "n1", "text": "Con đếm từ 1 đến 10.", "lang": "vi", "category": "vi_numbers", "en_words": [],
     "alt_texts": ["Con đếm từ một đến mười."]},
    {"id": "e1", "text": "Good morning!", "lang": "en", "category": "en_short", "en_words": []},
]


def make_wav(seconds: float, sr: int = 24_000, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def sentences_dir(tmp_path: Path) -> Path:
    root = tmp_path / "sentences"
    root.mkdir()
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in SENTENCES) + "\n",
                                         encoding="utf-8")
    (root / "reference.wav").write_bytes(make_wav(1.0, sr=16_000))
    (root / "reference.json").write_text(json.dumps({"text": "câu tham chiếu"}, ensure_ascii=False), encoding="utf-8")
    return root

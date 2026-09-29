"""The one text normalizer applied to references and hypotheses alike.

Steps, in order: Unicode NFC, lowercase, replace anything that is not a word
character, whitespace or a Vietnamese letter with a space (robo-be's pattern),
collapse whitespace. Bump ``NORMALIZER_VERSION`` whenever the output can change;
it is stamped on every scored result.
"""
from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "nfc-lower-vi-1"

_VI_LETTERS = "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
_NOT_WORD = re.compile(rf"[^\w\s{_VI_LETTERS}]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = _NOT_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []

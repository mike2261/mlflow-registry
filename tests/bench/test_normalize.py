import unicodedata

from mlflow_registry.bench.normalize import NORMALIZER_VERSION, normalize, tokens


def test_version_is_pinned():
    assert NORMALIZER_VERSION == "nfc-lower-vi-1"


def test_lowercase_punctuation_and_whitespace():
    assert normalize("  Cảm ơn!  Bác sĩ, ok?? ") == "cảm ơn bác sĩ ok"


def test_vietnamese_letters_survive():
    text = "cộng hoà xã hội chủ nghĩa việt nam đ ư ơ"
    assert normalize(text) == text


def test_nfd_and_nfc_forms_normalize_to_the_same_string():
    nfc = "cơ hội"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc  # the test is meaningless otherwise
    assert normalize(nfd) == normalize(nfc) == "cơ hội"


def test_digits_and_ascii_words_kept():
    assert normalize("A rolling STONE gathers 2 mosses.") == "a rolling stone gathers 2 mosses"


def test_tokens_splits_normalized_text():
    assert tokens("Good, morning!") == ["good", "morning"]
    assert tokens("   ") == []

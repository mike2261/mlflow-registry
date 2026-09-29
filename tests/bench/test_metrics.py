import unicodedata

import pytest

from mlflow_registry.bench import metrics
from mlflow_registry.bench.manifest import Utterance


def _utt(i: int, text: str, lang: str = "vi", en_words=(), dur: float = 1.0) -> Utterance:
    return Utterance(id=f"{i:02d}", file=f"{i:02d}.wav", text=text, lang=lang,
                     category=f"{lang}_short", en_words=tuple(en_words), duration_s=dur)


def test_score_text_counts_word_and_char_edits():
    word_edits, ref_words, char_edits, ref_chars, exact = metrics.score_text("cảm ơn", "cảm on")
    assert (word_edits, ref_words) == (1, 2)
    assert ref_chars == len("cảmơn")
    assert char_edits == 1
    assert exact is False


def test_score_text_ignores_punctuation_case_and_composition():
    hyp = unicodedata.normalize("NFD", "Cơ hội!")
    assert metrics.score_text("cơ hội", hyp) == (0, 2, 0, 5, True)


def test_score_text_insertions_count_as_edits():
    word_edits, ref_words, *_ = metrics.score_text("anh em", "anh em ơi")
    assert (word_edits, ref_words) == (1, 2)


def test_en_recall_counts():
    assert metrics.en_recall_counts(["good", "morning"], "Good morning!") == (2, 2)
    assert metrics.en_recall_counts(["good", "morning"], "gút mó ninh") == (0, 2)
    assert metrics.en_recall_counts([], "anything") == (0, 0)


def test_pooled_wer_is_not_mean_of_per_utterance_wer():
    # 13 perfect two-word utterances + one six-word utterance with two errors:
    # pooled = 2 / 32 = 6.25 %, averaged would be (33 % + 0 * 13) / 14 = 2.4 %
    rows = [metrics.scored(_utt(i, "anh em"), "anh em", None, 0.5) for i in range(13)]
    rows.append(metrics.scored(_utt(13, "a rolling stone gathers no moss", "en"),
                               "a rolling stone gathers", None, 0.5))
    agg = metrics.aggregate(rows)
    assert agg.ref_words == 32
    assert agg.word_edits == 2
    assert agg.wer == pytest.approx(2 / 32)
    assert agg.exact == 13
    assert agg.exact_rate == pytest.approx(13 / 14)


def test_failures_are_counted_not_scored():
    ok = metrics.scored(_utt(0, "anh em"), "anh em", None, 0.4)
    empty = metrics.scored(_utt(1, "cảm ơn"), None, "empty transcript", None)
    boom = metrics.scored(_utt(2, "bánh mì"), None, "HTTP 500", None)
    agg = metrics.aggregate([ok, empty, boom])
    assert agg.n == 3
    assert agg.failures == 2
    assert agg.ref_words == 2       # only the scored row contributes
    assert agg.wer == 0.0
    assert agg.latency_median_s == pytest.approx(0.4)


def test_latency_stats_and_rtf():
    rows = [metrics.scored(_utt(i, "anh em", dur=2.0), "anh em", None, lat)
            for i, lat in enumerate([0.1, 0.2, 0.3, 0.4, 1.0])]
    agg = metrics.aggregate(rows)
    assert agg.latency_median_s == pytest.approx(0.3)
    assert agg.latency_p90_s == pytest.approx(1.0)
    assert agg.audio_s == pytest.approx(10.0)
    assert agg.rtf == pytest.approx(2.0 / 10.0)


def test_aggregate_of_nothing_has_none_rates():
    agg = metrics.aggregate([])
    assert agg.n == 0 and agg.wer is None and agg.cer is None and agg.rtf is None
    assert agg.latency_median_s is None and agg.en_recall is None

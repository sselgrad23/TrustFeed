"""Tests for the WER/CER metrics."""

from __future__ import annotations

from trustfeed.eval.wer import character_error_rate, word_error_rate


def test_wer_zero_for_identical_text() -> None:
    assert word_error_rate("the cat sat", "the cat sat") == 0.0


def test_wer_ignores_case_and_punctuation() -> None:
    assert word_error_rate("The cat sat.", "the cat sat") == 0.0


def test_wer_counts_one_substitution() -> None:
    # one of three words wrong -> WER 1/3
    assert abs(word_error_rate("the cat sat", "the dog sat") - 1 / 3) < 1e-6


def test_cer_zero_for_identical_text() -> None:
    assert character_error_rate("bonjour", "bonjour") == 0.0

"""Word- and character-error-rate metrics for ASR evaluation.

Thin wrapper over `jiwer <https://github.com/jitsi/jiwer>`_ with a shared text
normalisation (lower-case, strip punctuation, collapse whitespace) so scores
reflect recognition quality rather than formatting differences - the standard way
WER is reported on benchmarks like FLEURS.
"""

from __future__ import annotations

import jiwer

_WORD_NORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return the normalised word error rate between two strings (0.0 = perfect)."""
    out = jiwer.process_words(
        reference, hypothesis,
        reference_transform=_WORD_NORM, hypothesis_transform=_WORD_NORM,
    )
    return float(out.wer)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return the normalised character error rate between two strings."""
    return float(jiwer.cer(reference.lower().strip(), hypothesis.lower().strip()))

"""Build a real, legally-usable sample broadcast from the FLEURS corpus.

We need real speech that is (a) free to redistribute, (b) multilingual, and (c)
comes with a gold transcript so we can compute a *real* word error rate. `FLEURS
<https://huggingface.co/datasets/google/fleurs>`_ fits all three: ~2k parallel
read sentences in 102 languages, licensed **CC-BY 4.0**. The sentences come from
FLoRes (Wikipedia-derived informational text), so they read as news-style prose.

This script streams a handful of same-language sentences and concatenates them
into one multi-minute "broadcast" clip, writing alongside it:

* ``<name>.wav``          - the audio,
* ``<name>.ref.txt``      - the concatenated gold transcript (for WER), and
* ``<name>.segments.json``- per-sentence ``{start,end,text}`` (real timestamps),
  which is what the extraction gold labels are hand-annotated against.

Run as a module::

    python -m trustfeed.media.download_sample --lang fr_fr --n 14
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from trustfeed import config

logger = logging.getLogger(__name__)

# A few FLEURS language configs. Whisper auto-detects, so any of these "just work".
LANGUAGES = {"fr_fr": "French", "de_de": "German", "es_419": "Spanish", "en_us": "English"}


def build_fleurs_sample(
    lang: str = "fr_fr", n: int = 14, gap_s: float = 0.4, split: str = "test"
) -> Path:
    """Stream ``n`` FLEURS sentences for ``lang`` and write a concatenated clip.

    Args:
        lang: A FLEURS language config (e.g. ``fr_fr``, ``de_de``, ``es_419``).
        n: Number of sentences to stitch together.
        gap_s: Silence inserted between sentences, in seconds.
        split: Dataset split to pull from.

    Returns:
        Path to the written ``.wav`` file.
    """
    import io

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    config.ensure_dirs()
    logger.info("Streaming %d FLEURS '%s' sentences...", n, lang)
    ds = load_dataset("google/fleurs", lang, split=split, streaming=True)
    # Decode audio bytes ourselves with soundfile - avoids the torchcodec
    # dependency newer `datasets` versions require for server-side decoding.
    ds = ds.cast_column("audio", Audio(decode=False))

    chunks: list = []
    segments: list[dict] = []
    refs: list[str] = []
    sr = 16000
    cursor = 0.0
    for i, ex in enumerate(ds):
        if i >= n:
            break
        audio = ex["audio"]
        data = audio.get("bytes")
        if data is None:
            with open(audio["path"], "rb") as fh:
                data = fh.read()
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # mixdown to mono
        text = (ex.get("raw_transcription") or ex.get("transcription") or "").strip()
        dur = len(arr) / sr
        segments.append({"start": round(cursor, 2), "end": round(cursor + dur, 2), "text": text})
        refs.append(text)
        chunks.append(arr)
        chunks.append(np.zeros(int(gap_s * sr), dtype="float32"))
        cursor += dur + gap_s

    if not chunks:
        raise RuntimeError(f"No samples returned for FLEURS '{lang}'.")

    audio_out = np.concatenate(chunks)
    stem = config.SAMPLE_DIR / f"fleurs_{lang}"
    wav_path = stem.with_suffix(".wav")
    sf.write(wav_path, audio_out, sr)
    stem.with_suffix(".ref.txt").write_text(" ".join(refs), encoding="utf-8")
    stem.with_suffix(".segments.json").write_text(
        json.dumps({"language": lang, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%.1fs, %d sentences)", wav_path, cursor, len(refs))
    return wav_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="fr_fr", choices=sorted(LANGUAGES))
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = build_fleurs_sample(args.lang, args.n, split=args.split)
    print(path)


if __name__ == "__main__":
    _main()

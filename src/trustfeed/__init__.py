"""TrustFeed - turn long-form foreign-language news audio into review-ready social clips.

The package is organised as one vertical slice of an editorial pipeline:

* :mod:`trustfeed.asr` - Whisper transcription with word/segment timestamps,
* :mod:`trustfeed.extract` - LLM extraction of high-value clip candidates,
* :mod:`trustfeed.eval` - WER and clip-selection evaluation,
* :mod:`trustfeed.monitoring` - input-stream drift monitoring,
* :mod:`trustfeed.api` - the FastAPI service that fronts it all.
"""

from __future__ import annotations

__version__ = "0.1.0"

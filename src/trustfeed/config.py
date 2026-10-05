"""Central configuration for TrustFeed.

Every path resolves relative to the repository root so the code behaves the same
on a laptop, in a Docker container, or in CI. Anything that differs between local
development and a deployed Space is overridable with an environment variable.

The pipeline makes **no** assumptions about a single language: Whisper detects the
spoken language and the extractor is prompted to respond in that same language,
so the same code handles French, German, Spanish, Arabic, etc.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# ``config.py`` lives at ``<root>/src/trustfeed/config.py`` -> three parents up.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.getenv("TRUSTFEED_DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR: Path = Path(os.getenv("TRUSTFEED_MODEL_DIR", ROOT_DIR / "models"))
REPORT_DIR: Path = Path(os.getenv("TRUSTFEED_REPORT_DIR", ROOT_DIR / "reports"))
SAMPLE_DIR: Path = DATA_DIR / "samples"

# --- ASR (faster-whisper) ---------------------------------------------------
# ``faster-whisper`` runs the CTranslate2 Whisper build: fast, free, fully local.
# ``small`` is the sweet spot on an 8 GB GPU or a CPU laptop; override for quality.
ASR_MODEL: str = os.getenv("TRUSTFEED_ASR_MODEL", "small")
ASR_DEVICE: str = os.getenv("TRUSTFEED_ASR_DEVICE", "auto")  # auto|cuda|cpu
# int8_float16 on GPU, int8 on CPU - both keep VRAM/RAM low without hurting WER much.
ASR_COMPUTE_TYPE: str = os.getenv("TRUSTFEED_ASR_COMPUTE_TYPE", "default")
ASR_BEAM_SIZE: int = int(os.getenv("TRUSTFEED_ASR_BEAM_SIZE", "5"))

# --- Extraction LLM ---------------------------------------------------------
# Backend selection: "transformers" (local instruct model), "heuristic" (no LLM,
# used in CI / as a graceful fallback and the eval floor), or "auto" (transformers
# when a CUDA GPU + the model are available, else heuristic).
LLM_BACKEND: str = os.getenv("TRUSTFEED_LLM_BACKEND", "auto")
# Qwen2.5-3B-Instruct: Apache-2.0, strong multilingual instruction-following, and
# small enough to run in 4-bit next to Whisper on an 8 GB card. See docs/PROPOSAL.md
# for the accuracy/latency/cost trade-off behind this choice.
LLM_MODEL: str = os.getenv("TRUSTFEED_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LLM_LOAD_4BIT: bool = os.getenv("TRUSTFEED_LLM_4BIT", "1") == "1"
LLM_MAX_NEW_TOKENS: int = int(os.getenv("TRUSTFEED_LLM_MAX_NEW_TOKENS", "640"))

# Long-context handling: transcripts of a 20-40 min broadcast overflow a small
# model's usable context, so we window the transcript (map) and merge the clip
# candidates (reduce). These bound the window in characters with an overlap so a
# clip straddling a boundary is still seen whole in one window.
WINDOW_CHARS: int = int(os.getenv("TRUSTFEED_WINDOW_CHARS", "6000"))
WINDOW_OVERLAP_CHARS: int = int(os.getenv("TRUSTFEED_WINDOW_OVERLAP_CHARS", "600"))
MAX_CLIPS: int = int(os.getenv("TRUSTFEED_MAX_CLIPS", "6"))

# --- MLflow -----------------------------------------------------------------
# Local SQLite backend by default - no server, no paid service. Override with a
# remote tracking-server URI if one is available.
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "trustfeed")

# --- Editorial workflow -----------------------------------------------------
# Nothing is "published" until a human approves it. Distribution channels are
# illustrative - the demo produces a ready-to-post package rather than calling a
# paid WhatsApp/Facebook Business API (which would bill and needs a verified org).
DISTRIBUTION_CHANNELS: tuple[str, ...] = ("X/Twitter", "Instagram Reels", "TikTok", "WhatsApp")


def ensure_dirs() -> None:
    """Create the data/model/report/sample directories if they do not exist."""
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR, SAMPLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

"""Gradio demo for TrustFeed - free Hugging Face Space entry point.

The hosted demo runs the *real* `trustfeed` pipeline (bundled under ``src/``) on
Hugging Face's **free ZeroGPU tier**: Whisper (faster-whisper) transcription plus the
deterministic heuristic extractor. The heuristic backend is chosen for the public
demo because it is instant and reliable within a ZeroGPU time slice; the local LLM
backend (Qwen2.5-3B-Instruct) is what the repository runs - see the README.

The demo keeps the product's signature feature: a **human-validation gate**. You
transcribe, tick the clips you would actually publish, and only those come back as
a ready-to-post package. Nothing is "published" automatically.
"""

from __future__ import annotations

import os
import sys

# Route all writes to /tmp (a Space's app dir can be read-only) and pick the fast,
# reliable demo configuration. Must be set before trustfeed.config is imported.
os.environ.setdefault("TRUSTFEED_DATA_DIR", "/tmp/trustfeed/data")
os.environ.setdefault("TRUSTFEED_MODEL_DIR", "/tmp/trustfeed/models")
os.environ.setdefault("TRUSTFEED_REPORT_DIR", "/tmp/trustfeed/reports")
os.environ.setdefault("HF_HOME", "/tmp/trustfeed/hf")
os.environ.setdefault("TRUSTFEED_LLM_BACKEND", "heuristic")
# Run Whisper CPU-side with a small, fast model. CTranslate2's CUDA build is
# fragile on the ZeroGPU worker, so we transcribe on CPU (reliable) within the
# GPU slice rather than chase cuDNN issues - plenty fast for short demo clips.
os.environ.setdefault("TRUSTFEED_ASR_MODEL", "base")
os.environ.setdefault("TRUSTFEED_ASR_DEVICE", "cpu")
os.environ.setdefault("TRUSTFEED_ASR_COMPUTE_TYPE", "int8")

# The trustfeed package is bundled under ./src in the Space; make it importable
# without a full editable install (Spaces just run `python app.py`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr  # noqa: E402

from trustfeed.extract.schema import Extraction, format_timestamp  # noqa: E402
from trustfeed.extract.pipeline import process  # noqa: E402

# Free HF Spaces for a non-PRO account run on ZeroGPU, which (a) needs at least one
# @spaces.GPU-decorated function to start and (b) only provisions a GPU inside it -
# which is exactly where we want Whisper to run. Locally the `spaces` package isn't
# installed, so fall back to a no-op decorator that supports both @gpu and @gpu(...).
try:
    import spaces  # provided on ZeroGPU Spaces

    gpu = spaces.GPU
except ImportError:
    def gpu(func=None, **kwargs):  # type: ignore[no-redef]
        """No-op stand-in for spaces.GPU when the package is unavailable."""
        if func is None:
            return lambda f: f
        return func

CHANNELS = ["X/Twitter", "Instagram Reels", "TikTok", "WhatsApp"]


@gpu(duration=60)  # base/int8 ASR does a ~3-min clip in ~20s CPU-side; 60s is ample
def analyze(audio_path: str | None):
    """Transcribe + extract, returning UI updates and the extraction state."""
    if not audio_path:
        return ("Upload or record some news audio first.", "", gr.update(choices=[], value=[]), None)

    extraction = process(audio_path)
    header = (
        f"**Language:** {extraction.language} | **Duration:** "
        f"{format_timestamp(extraction.duration)} | **Clip candidates:** "
        f"{len(extraction.clips)} | _asr {extraction.asr_seconds}s_"
    )
    summary = f"**Summary.** {extraction.summary}" if extraction.summary else ""

    choices, cards = [], []
    for i, clip in enumerate(extraction.clips):
        label = f"[{clip.timespan}] {clip.caption}"
        choices.append((label, i))
        cards.append(
            f"**{clip.timespan} - {clip.title}**\n\n"
            f'"{clip.caption}"\n\n'
            + (f"> Quote: {clip.quote}\n\n" if clip.quote else "")
            + (f"_Why: {clip.rationale}_" if clip.rationale else "")
        )
    body = header + "\n\n" + summary + "\n\n---\n\n" + "\n\n---\n\n".join(cards)
    return (body, "", gr.update(choices=choices, value=[]), extraction.model_dump())


def publish(selected: list[int], state: dict | None):
    """The human-validation gate: package only the editor-approved clips."""
    if not state:
        return "Analyze some audio first."
    extraction = Extraction.model_validate(state)
    if not selected:
        return "No clips approved - nothing is published. (That's the gate working.)"
    lines = [f"### Approved for distribution ({len(selected)})", ""]
    for i in selected:
        clip = extraction.clips[i]
        lines += [
            f'**{clip.timespan}** - "{clip.caption}"',
            f"-> {' | '.join(CHANNELS)}",
            "",
        ]
    return "\n".join(lines)


with gr.Blocks(title="TrustFeed") as demo:
    gr.Markdown(
        "# TrustFeed\n"
        "Foreign-language news audio -> transcript -> **review-ready social clips**. "
        "Bad actors move fast; newsrooms should move faster - but with a human on the "
        "trust-critical step.\n\n"
        "_Hosted demo: Whisper ASR + heuristic extractor on the free ZeroGPU tier. The repo "
        "also runs a local LLM (Qwen2.5-3B) extractor on a GPU._"
    )
    state = gr.State(None)
    audio = gr.Audio(type="filepath", label="News audio (upload or record, any language)")
    analyze_btn = gr.Button("Transcribe & find clips", variant="primary")
    output = gr.Markdown()
    approve = gr.CheckboxGroup(choices=[], label="Approve the clips you would publish")
    publish_btn = gr.Button("Publish approved clips")
    published = gr.Markdown()

    analyze_btn.click(analyze, inputs=audio, outputs=[output, published, approve, state])
    publish_btn.click(publish, inputs=[approve, state], outputs=published)

if __name__ == "__main__":
    demo.launch()

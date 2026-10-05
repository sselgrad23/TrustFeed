# TrustFeed

Long-form foreign-language news audio goes in; short, captioned, timestamped clips come out, ready for an editor to approve and post. Whisper handles the transcription, a language model picks the moments worth clipping and writes the captions, and nothing leaves the building until a human signs off. The premise is simple: misinformation travels fast because the people spreading it aren't waiting for anyone's sign-off, and a newsroom loses ground when the accurate version is still buried in a 30-minute broadcast. TrustFeed closes that gap without dropping the trust check.

Live demo: https://huggingface.co/spaces/sophiesd/trustfeed

This is a full slice rather than a notebook: a REST API, a React review console, the local ASR+LLM pipeline, an evaluation suite, and the usual MLOps plumbing (MLflow tracking, CI, drift monitoring). It runs free and offline, with no paid APIs and no key that can bill you.

## What it does

```
audio/video --> Whisper ASR (timestamps) --> LLM extraction --> review gate --> ready-to-post package
              faster-whisper               chapters + clip      editor         approved clips only
                                           candidates + captions  approves
```

A few design choices are worth calling out. Nothing hardcodes a language - Whisper detects it, so French, German and Spanish all go through the same path. The model chooses clips by segment id rather than by raw timestamp, so a clip can't point at a time that doesn't exist; the transcript supplies the actual seconds. Whatever the model returns is validated against a typed schema and repaired once if it comes back malformed. And the human-validation gate is carried over from the original hackathon build, because it is the point of the product, not a bolt-on.

## Real numbers

From `make eval` on the FLEURS samples:

| Metric | Result |
|---|---|
| ASR word error rate (Whisper `small` vs FLEURS gold) | 0.121 mean; French 0.164, German 0.078 |
| ASR speed | 7-12x real-time on an 8 GB GPU |
| Clip-selection F1 (hand-labelled gold) | heuristic 0.48, LLM (Qwen2.5-3B) 0.29 |
| Structured-output validity (valid JSON, first try) | 100% |
| LLM extraction latency | ~26 s for a 2.7-min clip (warm, 8 GB GPU) |

There is a result here I did not expect, and I left it in. The 3B model is more precise than the heuristic baseline but much more cautious, so it proposes fewer clips and loses on F1. Where it earns its keep is caption and summary quality, which this metric doesn't measure. Tuning the extraction prompt moved it from 0.18 to 0.29, and that is the whole reason the eval exists: you can't fix what you can't see. The longer version is in [docs/PROPOSAL.md](docs/PROPOSAL.md), section 7.

## Requirements

- Python 3.10+
- A GPU is recommended for the LLM extractor (an 8 GB card runs Qwen2.5-3B in 4-bit); everything also runs CPU-only on the heuristic backend.
- Node 20+ (only for the React console)
- `ffmpeg` (for decoding arbitrary audio/video)

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm,eval]"      # core + dev tools + local LLM + eval extras
```

Drop `llm` to stay torch-free and use the heuristic extractor (the CI/CPU path).

## 2. Get a sample & run the evaluation

```bash
make sample                     # stream a FLEURS French sample -> data/samples/
make eval                       # ASR WER + clip-selection F1 -> reports/, logged to MLflow
```

Or step by step:

```bash
python -m trustfeed.media.download_sample --lang fr_fr   # or de_de, es_419, en_us
python -m trustfeed.eval.asr_eval --langs fr_fr de_de
python -m trustfeed.eval.extraction_eval --backends heuristic transformers
python -m trustfeed.monitoring.drift                     # input-stream drift report
```

Process one file end to end:

```bash
python -m trustfeed.extract.pipeline data/samples/fleurs_fr_fr.wav
```

## 3. Run the service + console

Locally, in two terminals:

```bash
make api        # uvicorn trustfeed.api.main:app --port 8000 --app-dir src  (docs at /docs)
make frontend   # React console on http://localhost:5173
```

With Docker, the API (`:8000`) and console (`:8080`) come up together:

```bash
docker compose up --build
```

The API defaults to the heuristic extractor (CPU, no model download); set `TRUSTFEED_LLM_BACKEND=transformers` on a GPU host to use the LLM.

## 4. Usage

```bash
# Extract from an already-transcribed document (no ASR):
curl -X POST localhost:8000/extract -H 'Content-Type: application/json' -d '{
  "language":"en",
  "segments":[
    {"start":0,"end":6,"text":"The central bank raised interest rates by half a point today."},
    {"start":6,"end":12,"text":"Officials said inflation of 7 percent remains their top concern."}
  ]}'

# Or upload a media file for the full pipeline:
curl -F "file=@data/samples/fleurs_fr_fr.wav" localhost:8000/process
```

The response is an `Extraction` with `status:"pending_review"`; approve clips via `POST /jobs/{id}/review`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/process` | Upload audio/video -> transcript + clip candidates |
| `POST` | `/extract` | Same, from an already-transcribed document (no ASR) |
| `GET`  | `/jobs/{id}` | Fetch a job |
| `POST` | `/jobs/{id}/review` | Human-validation gate: approve/reject clips, publish |
| `GET`  | `/health`, `/config` | Probe / operative configuration |

## Make targets

| Shortcut | Runs |
|---|---|
| `make install` | `pip install -e ".[dev,llm,eval]"` |
| `make sample` | download a FLEURS sample (`LANG=fr_fr`) |
| `make asr-eval` / `extract-eval` / `eval` | WER / clip-selection F1 / both |
| `make drift` | input-stream drift report |
| `make api` / `frontend` | serve the API / the console |
| `make lint` / `typecheck` / `test` | `ruff` / `mypy` / `pytest` |
| `make docker` | full stack via docker compose |

## Layout

```
src/trustfeed/
  asr/transcribe.py        faster-whisper ASR -> timestamped segments
  extract/schema.py        typed structured-output contract (Pydantic)
  extract/prompts.py       extraction prompts (grounded, JSON-only)
  extract/llm.py           LLM + heuristic backends; validate/repair; windowed map-reduce
  extract/pipeline.py      audio -> transcript -> extraction (observability + MLflow)
  eval/                    WER + clip-selection F1 against hand-labelled gold
  monitoring/drift.py      PSI drift on language / duration / clip-yield
  api/                     FastAPI service (upload, extract, review gate)
  media/download_sample.py build a FLEURS sample broadcast (CC-BY)
frontend/                  React + Vite review console
tests/                     pytest suite (heuristic backend, no GPU/network)
deploy/huggingface-gradio/ free Gradio Space (live demo)
docs/PROPOSAL.md           mission case, architecture decisions, honest limitations
legacy/                    the original 2024 hackathon n8n workflow, notebook and audio, for provenance
```

## Dataset & audio sample

The sample audio and the WER numbers both come from [FLEURS](https://huggingface.co/datasets/google/fleurs) (`google/fleurs`, CC-BY 4.0), a multilingual read-speech corpus that ships gold transcripts - which is what makes a real error rate possible in the first place. It is Wikipedia-style prose read aloud, so it stands in for broadcast news rather than being it; `trustfeed.media.download_sample` stitches same-language sentences into a few-minute clip. For genuine broadcast audio, [Voice of America](https://learningenglish.voanews.com/) output is a U.S. government work and therefore public domain, and drops straight in.

## Deploy

The live demo is a Gradio Space on Hugging Face's free ZeroGPU tier, running the real pipeline with Whisper and the heuristic extractor; the source is in `deploy/huggingface-gradio/`. One wrinkle worth knowing: HF now gates free *CPU* Gradio Spaces behind PRO, so the demo uses the free ZeroGPU hardware instead. Beyond that, the `Dockerfile` (API) and `frontend/Dockerfile` build images that run on any container platform.

## License

MIT (see `LICENSE`). Bundled sample derived from FLEURS (CC-BY 4.0).

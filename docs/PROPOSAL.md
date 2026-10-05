# Project Proposal - TrustFeed

*Turn long-form foreign-language news audio into review-ready social clips - fast, and with a human on the trust-critical step.*

## 1. The mission problem

Misinformation wins on speed. A fabricated clip can be cut, captioned and posted in minutes, while the accurate version is still sitting inside a 30-minute foreign-language broadcast that nobody has had time to watch. By the time a newsroom has found the newsworthy 20 seconds by hand, transcribed it, and written a caption, the false version has already travelled.

The bottleneck isn't reporting. It's repackaging: finding the moments worth sharing in long audio or video and turning them into short, captioned clips. That work is slow, repetitive and language-bound, which is the shape of task a narrow AI system is good at - as long as a human still signs off before anything goes public, because on a trust product a misleading clip is worse than a slow one.

TrustFeed is the original PeaceTech-Hackathon idea ("Most Innovative Idea", 2024), rebuilt as a real, free, self-hostable service rather than a no-code demo.

## 2. What TrustFeed does

Give it a news audio or video file in more or less any language and it returns a timestamped transcript, chapters for navigation, and a set of clip candidates - the segments worth distributing, each with an attention-catching caption and the verbatim key quote. None of them count as published until an editor approves them, at which point the approved clips come back as a ready-to-post package.

It ships as three parts a newsroom could actually operate: a FastAPI service any existing tool can call, a React console where an editor works through the clips, and the MLOps/eval backbone behind them - reproducible evaluation, MLflow tracking, CI, and drift monitoring on the incoming stream.

## 3. Reused from the hackathon original (not started from zero)

Two things from the original were worth keeping. The human-validation gate is the first - a genuine editorial-trust feature, now central to the API and the UI. The mission framing is the second: countering misinformation by matching bad actors on speed and reach. The overall shape of the old n8n flow (transcribe, extract, review, distribute) survives too, re-implemented as typed Python. What went: n8n itself, and the paid Argo LLM endpoint it called, since a hard requirement this time was that nothing can bill me.

## 4. Data and the audio sample

The pipeline needs real speech, and for the evaluation to mean anything that speech has to be free to redistribute and come with a gold transcript.

FLEURS ([`google/fleurs`](https://huggingface.co/datasets/google/fleurs), CC-BY 4.0) fits both: around 2k parallel read sentences across 102 languages, each with a reference transcript. `trustfeed.media.download_sample` streams same-language sentences and stitches them into a multi-minute "broadcast" (French and German are used here). The sentences come from FLoRes, which is Wikipedia-derived, so they read as informational prose rather than a live bulletin - a stand-in, and the README says as much. When actual broadcast audio is wanted, [Voice of America](https://learningenglish.voanews.com/) output is a U.S. government work in the public domain and drops in unchanged; FLEURS just happens to be scriptable and to carry the transcripts that WER needs. In production the input is the client's own stream, and nothing in the code is FLEURS-specific.

## 5. Architecture decisions - and why

The skills that matter for AI-engineering roles right now are structured outputs, evaluation, observability, and keeping non-deterministic systems reliable, more than any particular framework. Each choice below maps to one of those.

| Decision | Why |
|---|---|
| Whisper via `faster-whisper` for ASR | Transformer ASR is the task; the CTranslate2 build is 4x lighter, runs on CPU or GPU, and is free - 7-12x real-time on the 8 GB GPU here. |
| Local Qwen2.5-3B-Instruct for extraction | Free with zero billing risk (the hard constraint), strong multilingual instruction-following, Apache-2.0; trades latency for no per-call cost (section 6). |
| A heuristic extractor as second backend | Deterministic and GPU-free: the CI/offline fallback, and the evaluation floor the LLM has to beat - which turned out to matter (section 7). |
| Structured outputs (Pydantic + validate/repair) | Output is validated against a typed schema and repaired once if malformed, rather than shipped as-is - the reliability story for a non-deterministic component. |
| Grounded timestamps (model returns segment ids) | The model chooses which segment; the data supplies when. A clip can't land on a hallucinated time. |
| Long context via windowing (map/reduce) | A 30-minute transcript overflows a small model's context, so we window with overlap and merge the clips - the whole document is processed, without RAG (section 8). |
| Evaluation (WER + clip-selection F1) | The only way to know whether it works, and to have numbers to quote (section 7). |
| Observability (timing, JSON-validity, MLflow) | Latency and output validity are logged per run - the operational signals that decide whether an LLM system survives real use. |
| Drift monitoring (PSI) | Flags when the incoming stream has shifted enough - new languages, longer files, collapsing clip yield - to re-validate the pipeline. |
| FastAPI + React + Docker + CI | Packaging a client could actually run, not a notebook. |

## 6. The LLM trade-off (accuracy / latency / cost)

There were three free options for the extraction step.

A local small instruct model (Qwen2.5-3B-Instruct, 4-bit) is what I chose. It has zero marginal cost, needs no key, and involves no vendor that can bill, and it runs beside Whisper on an 8 GB card. The cost is latency - roughly 26 s of extraction for a 2.7-minute clip once the model is warm, plus a one-time load, against milliseconds for the heuristic - and the ops burden of self-hosting.

A free-tier hosted API (Groq or Gemini, say) would be faster and better, but every free tier has usage caps and a live key that could bill, which is exactly the line I couldn't cross. It's the obvious upgrade the day that constraint relaxes.

The heuristic extractive baseline is instant and free but produces nothing worth publishing on its own. It stays as the baseline, not the product.

So the choice is the "zero billing risk, self-hosted, live with the latency" corner of the accuracy/latency/cost trade-off.

## 7. Does it actually work? (evaluation)

The numbers come from a real run on the FLEURS samples and regenerate with `make eval`.

For ASR, Whisper `small` scores a mean WER of 0.121 against the FLEURS gold transcripts - 0.164 on French, 0.078 on German - at 7-12x real-time. For clip selection, I hand-labelled a small gold set where each transcript's newsworthy spans are marked and the encyclopedic sentences are left as distractors; the eval then measures how well the predicted clip spans overlap the gold ones (precision, recall, F1), alongside the structured-output validity rate.

The clip-selection result is the interesting one. The first prompt was too conservative - F1 0.18, recall 0.11, barely proposing any clips at all. Loosening it toward coverage brought that to 0.29 (recall 0.22). Even so, the heuristic baseline still wins at 0.48, simply because it puts forward more candidates. The honest reading is that for pure segment selection a 3B model isn't worth its latency here; its value is in caption and summary quality, which this metric doesn't capture. That points at the production design I'd actually ship - heuristic candidate generation with LLM captioning on top - and it's the kind of non-obvious conclusion the eval is there to produce. JSON-schema validity, for what it's worth, was 100% first-try in every run.

## 8. Why not RAG

I considered retrieval and left it out on purpose. The task is extraction from one document already in hand, not search across a corpus, so there's nothing to retrieve. The real difficulty - a transcript longer than the model's usable context - is a windowing problem, and windowed map/reduce solves it by processing the whole document instead of fetching fragments of it. Retrieval would only make sense at a different scope, for instance checking a claim against a fact-check archive, and that would be added deliberately rather than by reflex.

## 9. Scope and cost

Everything runs free and local: no paid APIs, no key that can bill, no billed cloud. The live demo is a free Hugging Face Gradio Space on ZeroGPU, and the Docker images go to any container platform. Two free-tier limits are worth stating plainly - the Space sleeps after inactivity, and ZeroGPU has a per-session GPU-time quota, so the first hit after a nap is slow and a very long file can exceed a quota window. That's fine for a demo. The scope is one vertical slice, ingest through transcribe, extract, review and (simulated) distribute, rather than a broad platform.

## 10. Honest limitations

- On clip-selection F1, the 3B LLM loses to the heuristic (section 7). That isn't buried; it's the headline result and the argument for a hybrid design.
- The small model tends to translate captions to English and occasionally mis-copies the `quote` field. A larger model, or an explicit target-language and verbatim-check pass, would fix it.
- The gold set is small: two transcripts, labelled by me. It's a real eval, not a benchmark, so read the numbers as directional.
- FLEURS is read Wikipedia-style speech, not live broadcast news - a licensed stand-in for the target domain.
- Distribution is simulated. Real WhatsApp or Facebook posting needs a paid, verified Business API, so the demo returns a ready-to-post package instead of calling out.
- Jobs live in memory. Fine for a demo; a real deployment needs a database or queue, and auth.
- This is an independent rebuild of the hackathon prototype. It has not been deployed at a newsroom.

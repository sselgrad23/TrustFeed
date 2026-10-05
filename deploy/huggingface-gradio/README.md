---
title: TrustFeed
emoji: 📰
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# TrustFeed - live demo

Foreign-language news audio -> transcript -> **review-ready social clips**, with a human-validation gate before anything is "published".

Upload or record some news audio (any language). TrustFeed transcribes it with Whisper (`faster-whisper`), proposes timestamped clip candidates with captions, and lets you approve the ones you would actually distribute - only those come back as a ready-to-post package.

Runs on Hugging Face's **free ZeroGPU tier** with a small Whisper model and the fast, reliable heuristic extractor; the Space sleeps after inactivity and wakes on the next visit, and ZeroGPU applies a per-session time quota. The full project - REST API, React review console, the local LLM (Qwen2.5-3B) extractor, evaluation, Docker stack - lives in the source repository.

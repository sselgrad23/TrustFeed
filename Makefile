# Everything here runs locally and is free of charge.
.PHONY: help install sample asr-eval extract-eval eval drift api frontend lint typecheck test docker

help:
	@echo "install      Install the package + dev/LLM/eval extras (editable)"
	@echo "sample       Download a FLEURS sample broadcast (LANG=fr_fr)"
	@echo "asr-eval     Transcribe the samples and report WER (FLEURS gold)"
	@echo "extract-eval Score clip selection vs the hand-labelled gold set"
	@echo "eval         asr-eval + extract-eval (heuristic + transformers)"
	@echo "drift        Generate an input-stream drift report"
	@echo "api          Serve the FastAPI service on :8000"
	@echo "frontend     Run the React review console on :5173"
	@echo "lint         Ruff | typecheck  mypy | test  pytest"
	@echo "docker       Build & run the full stack via docker compose"

LANG ?= fr_fr

install:
	pip install -e ".[dev,llm,eval]"

sample:
	python -m trustfeed.media.download_sample --lang $(LANG)

asr-eval:
	python -m trustfeed.eval.asr_eval --langs fr_fr de_de

extract-eval:
	python -m trustfeed.eval.extraction_eval --backends heuristic transformers

eval: asr-eval extract-eval

drift:
	python -m trustfeed.monitoring.drift

api:
	uvicorn trustfeed.api.main:app --reload --port 8000 --app-dir src

frontend:
	cd frontend && npm install && npm run dev

lint:
	ruff check src tests

typecheck:
	mypy src

test:
	pytest -q

docker:
	docker compose up --build
